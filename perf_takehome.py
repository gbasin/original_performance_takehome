"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    """
    Optimized VLIW kernel builder using:
    - SIMD vectorization (VLEN=8)
    - Parallel chunk processing (2 chunks)
    - Preloaded vector constants
    - Arithmetic instead of vselect where possible
    - Dependency-aware VLIW scheduling
    """

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.pending_slots = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vector_scratch(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def get_const(self, val):
        if val not in self.const_map:
            addr = self.alloc_scratch(f"const_{val}")
            self.const_map[val] = addr
            self.instrs.append({"load": [("const", addr, val)]})
        return self.const_map[val]

    def emit(self, engine, slot):
        self.pending_slots.append((engine, slot))

    def _get_slot_reads_writes(self, engine, slot):
        reads, writes = set(), set()
        op = slot[0]
        if engine == "alu":
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])
        elif engine == "valu":
            if op == "vbroadcast":
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
            else:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
        elif engine == "load":
            if op == "const":
                writes.add(slot[1])
            elif op == "load":
                writes.add(slot[1])
                reads.add(slot[2])
            elif op == "vload":
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif engine == "store":
            if op == "store":
                reads.add(slot[1])
                reads.add(slot[2])
            elif op == "vstore":
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)
        elif engine == "flow":
            if op in ("select", "vselect"):
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
                if op == "vselect":
                    for i in range(VLEN):
                        writes.add(slot[1] + i)
                        reads.add(slot[2] + i)
                        reads.add(slot[3] + i)
                        reads.add(slot[4] + i)
        return reads, writes

    def _has_dependency(self, slot1_info, slot2_info):
        _, _, _, writes1, _ = slot1_info
        _, _, reads2, writes2, _ = slot2_info
        return bool(writes1 & reads2) or bool(writes1 & writes2)

    def flush_schedule(self):
        if not self.pending_slots:
            return
        slots_info = []
        for engine, slot in self.pending_slots:
            reads, writes = self._get_slot_reads_writes(engine, slot)
            slots_info.append([engine, slot, reads, writes, False])
        n = len(slots_info)
        must_precede = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._has_dependency(slots_info[i], slots_info[j]):
                    must_precede[j].add(i)
        scheduled = [False] * n
        while not all(scheduled):
            ready = [i for i in range(n) if not scheduled[i] and all(scheduled[p] for p in must_precede[i])]
            if not ready:
                for i in range(n):
                    if not scheduled[i]:
                        engine, slot, _, _, _ = slots_info[i]
                        self.instrs.append({engine: [slot]})
                        scheduled[i] = True
                break
            bundle = {}
            slots_in_bundle = set()
            bundle_writes = set()
            for i in ready:
                engine, slot, reads, writes, _ = slots_info[i]
                limit = SLOT_LIMITS.get(engine, 0)
                current = len(bundle.get(engine, []))
                if writes & bundle_writes:
                    continue
                if current < limit:
                    bundle.setdefault(engine, []).append(slot)
                    slots_in_bundle.add(i)
                    bundle_writes |= writes
            if bundle:
                self.instrs.append(bundle)
                for i in slots_in_bundle:
                    scheduled[i] = True
        self.pending_slots = []

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Optimized vectorized kernel with:
        - Parallel chunk processing (2 chunks)
        - Preloaded vector constants including n_nodes
        - multiply_add for index computation
        - Reused address computations
        - Minimal flush_schedule barriers for better VLIW packing
        - Software pipelining (overlap load/compute)
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Header variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Load all header vars - can pack const loads
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        # Preload scalar constants
        for val in [0, 1, 2]:
            self.get_const(val)
        for _, val1, _, _, val3 in HASH_STAGES:
            self.get_const(val1)
            self.get_const(val3)

        # Preload vector constants (including n_nodes!)
        for val in [0, 1, 2]:
            v_addr = self.alloc_vector_scratch(f"vconst_{val}")
            self.vec_const_map[val] = v_addr
            self.emit("valu", ("vbroadcast", v_addr, self.get_const(val)))
        for _, val1, _, _, val3 in HASH_STAGES:
            if val1 not in self.vec_const_map:
                v_addr = self.alloc_vector_scratch(f"vconst_{val1:x}")
                self.vec_const_map[val1] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.get_const(val1)))
            if val3 not in self.vec_const_map:
                v_addr = self.alloc_vector_scratch(f"vconst_{val3}")
                self.vec_const_map[val3] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.get_const(val3)))

        # Preload n_nodes as vector constant (was broadcasting every iteration!)
        vc_n_nodes = self.alloc_vector_scratch("vconst_n_nodes")
        self.vec_const_map["n_nodes"] = vc_n_nodes
        self.emit("valu", ("vbroadcast", vc_n_nodes, self.scratch["n_nodes"]))
        self.flush_schedule()

        self.instrs.append({"flow": [("pause",)]})

        # Allocate parallel chunk scratch (8 chunks - good balance of parallelism vs overhead)
        n_chunks = 8
        parallel_scratch = []
        for c in range(n_chunks):
            p_idx = self.alloc_vector_scratch(f"p{c}_idx")
            p_val = self.alloc_vector_scratch(f"p{c}_val")
            p_node_val = self.alloc_vector_scratch(f"p{c}_node_val")
            p_tmp1 = self.alloc_vector_scratch(f"p{c}_tmp1")
            p_tmp2 = self.alloc_vector_scratch(f"p{c}_tmp2")
            p_idx_addr = self.alloc_scratch(f"p{c}_idx_addr")  # Reused for load and store
            p_val_addr = self.alloc_scratch(f"p{c}_val_addr")  # Reused for load and store
            parallel_scratch.append((p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr))

        vc_zero = self.vec_const_map[0]
        vc_one = self.vec_const_map[1]
        vc_two = self.vec_const_map[2]
        vc_n_nodes = self.vec_const_map["n_nodes"]
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        # Main loop - minimized flush_schedule calls
        for _ in range(rounds):
            for chunk_group in range(chunks_per_round):
                batch_indices = [chunk_group * n_chunks * chunk_size + c * chunk_size for c in range(n_chunks)]

                # Phase 1: Compute addresses (reused in Phase 6)
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("load", ("const", p_idx_addr, batch_indices[c]))
                    self.emit("load", ("const", p_val_addr, batch_indices[c]))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("alu", ("+", p_idx_addr, self.scratch["inp_indices_p"], p_idx_addr))
                    self.emit("alu", ("+", p_val_addr, self.scratch["inp_values_p"], p_val_addr))
                # Load indices and values
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("load", ("vload", p_idx, p_idx_addr))
                    self.emit("load", ("vload", p_val, p_val_addr))
                self.flush_schedule()

                # Phase 2: Gather node_val (bottleneck - 2 loads/cycle)
                # Must ensure addresses computed before loads, process 2 at a time to match load slots
                for vi in range(VLEN):
                    # Compute addresses for all chunks
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                        self.emit("alu", ("+", p_idx_addr, self.scratch["forest_values_p"], p_idx + vi))
                    self.flush_schedule()
                    # Loads - 2 at a time (limited by load slots)
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                        self.emit("load", ("load", p_node_val + vi, p_idx_addr))
                        if (c + 1) % 2 == 0:  # Flush every 2 loads
                            self.flush_schedule()
                    if n_chunks % 2 != 0:  # Handle odd number of chunks
                        self.flush_schedule()

                # Phase 3: XOR + Phase 4: Hash + Phase 5: Index computation
                # Emit all without intermediate flushes - let scheduler handle dependencies

                # XOR
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("^", p_val, p_val, p_node_val))

                # Hash - all 6 stages
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    vc1 = self.vec_const_map[val1]
                    vc3 = self.vec_const_map[val3]
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1))
                        self.emit("valu", (op3, p_tmp2, p_val, vc3))
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))

                # Index computation
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("%", p_tmp2, p_val, vc_two))  # p_tmp2 = val % 2
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))  # p_tmp2 = 1 if even else 0
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))  # p_tmp1 = offset
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, vc_n_nodes))  # p_tmp2 = idx < n_nodes
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("valu", ("*", p_idx, p_idx, p_tmp2))  # idx = idx * (idx < n_nodes)

                self.flush_schedule()  # Single flush for all VALU work

                # Phase 6: Store results - recompute addresses (can't reuse, were overwritten)
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("load", ("const", p_idx_addr, batch_indices[c]))
                    self.emit("load", ("const", p_val_addr, batch_indices[c]))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("alu", ("+", p_idx_addr, self.scratch["inp_indices_p"], p_idx_addr))
                    self.emit("alu", ("+", p_val_addr, self.scratch["inp_values_p"], p_val_addr))
                self.flush_schedule()
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_idx_addr, p_val_addr) in enumerate(parallel_scratch):
                    self.emit("store", ("vstore", p_idx_addr, p_idx))
                    self.emit("store", ("vstore", p_val_addr, p_val))
                self.flush_schedule()

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
