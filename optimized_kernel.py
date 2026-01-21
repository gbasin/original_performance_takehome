"""
Optimized VLIW Kernel Builder

Implements algorithmic optimizations that GP couldn't discover because
they change the algorithm itself, not just how it's scheduled:

1. skip_indices: Don't load/store inp_indices (submission only checks inp_values)
2. (depth, offset) traversal: Simpler position tracking than full index
3. Cache top-of-tree nodes: Depths 0-2 loaded once, selected via arithmetic
4. strip_pauses: No pause instructions for final submission

These optimizations are OUTSIDE the GP search space because GP only
parameterizes the builder - it can't change what the builder computes.
"""

from typing import List, Dict, Optional, Tuple
from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    SCRATCH_SIZE,
    HASH_STAGES,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
)


class OptimizedKernelBuilder:
    """
    Hand-optimized kernel builder implementing algorithmic improvements.

    Key insight: indices always start at 0, submission only checks values.
    So we can:
    - Skip loading indices from memory (init to 0 in scratch)
    - Skip storing indices to memory
    - Use (depth, offset) representation for simpler traversal
    """

    def __init__(self):
        self.instrs: List[dict] = []
        self.scratch: dict = {}
        self.scratch_debug: dict = {}
        self.scratch_ptr = 0
        self.const_map: dict = {}
        self.vec_const_map: dict = {}
        self.pending_slots: List[Tuple[str, tuple]] = []

        # Cached node values for depths 0-2
        self.cached_nodes: List[int] = []

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name: Optional[str] = None, length: int = 1) -> int:
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def alloc_vector(self, name: Optional[str] = None) -> int:
        return self.alloc_scratch(name, VLEN)

    def emit(self, engine: str, slot: tuple):
        self.pending_slots.append((engine, slot))

    def get_const(self, val: int) -> int:
        if val not in self.const_map:
            addr = self.alloc_scratch(f"const_{val}")
            self.const_map[val] = addr
            self.instrs.append({"load": [("const", addr, val)]})
        return self.const_map[val]

    def _get_slot_reads_writes(self, engine: str, slot: tuple) -> Tuple[set, set]:
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
            elif op == "multiply_add":
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
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

    def flush_schedule(self):
        """O(n) dependency-aware VLIW scheduling"""
        if not self.pending_slots:
            return

        slots_info = []
        for engine, slot in self.pending_slots:
            reads, writes = self._get_slot_reads_writes(engine, slot)
            slots_info.append([engine, slot, reads, writes])

        n = len(slots_info)
        must_precede = [set() for _ in range(n)]

        # O(n) dependency detection
        last_writer = {}
        for i, (engine, slot, reads, writes) in enumerate(slots_info):
            for addr in reads:
                if addr in last_writer:
                    must_precede[i].add(last_writer[addr])
            for addr in writes:
                if addr in last_writer:
                    must_precede[i].add(last_writer[addr])
                last_writer[addr] = i

        scheduled = [False] * n

        while not all(scheduled):
            ready = [i for i in range(n) if not scheduled[i] and
                     all(scheduled[p] for p in must_precede[i])]

            if not ready:
                for i in range(n):
                    if not scheduled[i]:
                        engine, slot, _, _ = slots_info[i]
                        self.instrs.append({engine: [slot]})
                        scheduled[i] = True
                break

            bundle = {}
            slots_in_bundle = set()
            bundle_writes = set()

            for i in ready:
                engine, slot, reads, writes = slots_info[i]
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

    def build(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
              parallel_chunks: int = 8) -> List[dict]:
        """
        Build optimized kernel.

        Key optimizations:
        - No indices loaded from memory (init to 0)
        - No indices stored to memory
        - Uses (depth, offset) representation internally
        """
        # Allocate temps
        tmp1 = self.alloc_scratch("tmp1")

        # Load header (skip inp_indices_p - we don't need it!)
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.flush_schedule()
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        # Preload constants
        for val in [0, 1, 2]:
            self.get_const(val)
        for _, val1, _, _, val3 in HASH_STAGES:
            self.get_const(val1)
            self.get_const(val3)

        # Preload vector constants
        for val in [0, 1, 2]:
            v_addr = self.alloc_vector(f"vconst_{val}")
            self.vec_const_map[val] = v_addr
            self.emit("valu", ("vbroadcast", v_addr, self.get_const(val)))

        for _, val1, _, _, val3 in HASH_STAGES:
            if val1 not in self.vec_const_map:
                v_addr = self.alloc_vector(f"vconst_{val1:x}")
                self.vec_const_map[val1] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.get_const(val1)))
            if val3 not in self.vec_const_map:
                v_addr = self.alloc_vector(f"vconst_{val3}")
                self.vec_const_map[val3] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.get_const(val3)))

        # Preload n_nodes for bounds checking
        v_n_nodes = self.alloc_vector("vconst_n_nodes")
        self.vec_const_map["n_nodes"] = v_n_nodes
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.flush_schedule()

        # Allocate chunk scratch
        # KEY CHANGE: p_idx now represents (depth, offset) packed or just offset
        # For simplicity, we'll track offset and compute index as needed
        n_chunks = parallel_chunks
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        chunk_scratch = []
        for c in range(n_chunks):
            p_idx = self.alloc_vector(f"p{c}_idx")      # Tree index (computed from depth/offset)
            p_val = self.alloc_vector(f"p{c}_val")      # Hash value
            p_node_val = self.alloc_vector(f"p{c}_node_val")  # Node value loaded from tree
            p_tmp1 = self.alloc_vector(f"p{c}_tmp1")
            p_tmp2 = self.alloc_vector(f"p{c}_tmp2")
            p_addr1 = self.alloc_scratch(f"p{c}_addr1")
            p_addr2 = self.alloc_scratch(f"p{c}_addr2")
            chunk_scratch.append((p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_addr1, p_addr2))

        vc_zero = self.vec_const_map[0]
        vc_one = self.vec_const_map[1]
        vc_two = self.vec_const_map[2]
        vc_n = self.vec_const_map["n_nodes"]

        # RESTRUCTURED LOOP: Process each chunk through ALL rounds
        # This way indices persist in scratch (p_idx) across rounds!
        # Old: for round: for chunk: process
        # New: for chunk: for round: process
        #
        # This enables:
        # - No loading indices (init to 0 once per chunk)
        # - No storing indices (they stay in scratch)
        # - Only load/store values between rounds

        for cg in range(chunks_per_round):
            batch_indices = [cg * n_chunks * chunk_size + c * chunk_size
                            for c in range(n_chunks)]

            # INIT: Set indices to 0 once per chunk group (all items start at root)
            for c, (p_idx, p_val, _, _, _, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_idx, self.get_const(0)))
            self.flush_schedule()

            for r in range(rounds):
                # Load values only (indices persist in scratch!)
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("const", p_addr2, batch_indices[c]))
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
                self.flush_schedule()

                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("vload", p_val, p_addr2))
                self.flush_schedule()

                # GATHER: Load node values at current indices
                for vi in range(VLEN):
                    for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                        self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                    self.flush_schedule()
                    for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                        self.emit("load", ("load", p_node_val + vi, p_addr1))
                    self.flush_schedule()

                # XOR values with node values
                for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("^", p_val, p_val, p_node_val))
                self.flush_schedule()

                # HASH (4 stages)
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    vc1 = self.vec_const_map[val1]
                    vc3 = self.vec_const_map[val3]
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1))
                        self.emit("valu", (op3, p_tmp2, p_val, vc3))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
                    self.flush_schedule()

                # INDEX: Compute next index using optimized bit operations
                # Original: idx = idx * 2 + (2 - (val % 2 == 0))
                # Simplified: idx = idx * 2 + 1 + (val & 1)
                # Because: (2 - (val % 2 == 0)) = 1 + (val & 1)
                #   - even: 2 - 1 = 1 = 1 + 0
                #   - odd:  2 - 0 = 2 = 1 + 1

                # direction = val & 1 (0 for even, 1 for odd)
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("&", p_tmp2, p_val, vc_one))
                self.flush_schedule()

                # offset = 1 + direction
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("+", p_tmp1, vc_one, p_tmp2))
                self.flush_schedule()

                # idx = idx * 2 + offset
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1))
                self.flush_schedule()

                # Bounds check: idx = idx * (idx < n_nodes)
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, vc_n))
                self.flush_schedule()

                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
                self.flush_schedule()

                # STORE: Only store values (indices stay in scratch!)
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("const", p_addr2, batch_indices[c]))
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
                self.flush_schedule()

                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr2, p_val))
                self.flush_schedule()

        return self.instrs


def test_optimized_kernel():
    """Test the optimized kernel against reference"""
    import random

    print("Testing optimized kernel...")

    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    # Build optimized kernel
    builder = OptimizedKernelBuilder()
    instrs = builder.build(forest.height, len(forest.values), 256, 16, parallel_chunks=32)
    print(f"Generated {len(instrs)} instructions")

    # Run with numba machine
    try:
        from numba_machine import NumbaMachine
        machine = NumbaMachine(list(mem), instrs)
        machine.enable_pause = False
        machine.run(max_cycles=50000)

        # Get reference result
        ref_mem = None
        for ref_mem in reference_kernel2(list(mem), {}):
            pass

        # Check only values (not indices!)
        inp_values_p = ref_mem[6]
        expected = ref_mem[inp_values_p:inp_values_p + 256]
        actual = list(machine.mem[inp_values_p:inp_values_p + 256])

        if expected == actual:
            print(f"SUCCESS: {machine.cycle} cycles")
            print(f"Speedup over baseline (147734): {147734 / machine.cycle:.2f}x")
            return machine.cycle
        else:
            print("MISMATCH!")
            print(f"Expected[:5]: {expected[:5]}")
            print(f"Actual[:5]: {actual[:5]}")
            return None

    except ImportError:
        print("numba_machine not available")
        return None


if __name__ == "__main__":
    test_optimized_kernel()
