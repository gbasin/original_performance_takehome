"""
Numba JIT-compiled VLIW simulator.

Compiles the instruction execution to native code.
"""

import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList
import time

VLEN = 8
SCRATCH_SIZE = 1536
MASK = (1 << 32) - 1

# Encode operations as integers for numba
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_XOR = 3
OP_AND = 4
OP_OR = 5
OP_SHL = 6
OP_SHR = 7
OP_MOD = 8
OP_LT = 9
OP_EQ = 10
OP_DIV = 11

# Encode instruction types
INSTR_ALU = 0
INSTR_VALU = 1
INSTR_VALU_BROADCAST = 2
INSTR_VALU_MULADD = 3
INSTR_LOAD_CONST = 4
INSTR_LOAD = 5
INSTR_VLOAD = 6
INSTR_STORE = 7
INSTR_VSTORE = 8
INSTR_FLOW_PAUSE = 9
INSTR_FLOW_HALT = 10
INSTR_FLOW_SELECT = 11
INSTR_FLOW_VSELECT = 12
INSTR_NOP = 13
INSTR_BUNDLE_END = 99  # Marker for end of bundle

OP_MAP = {
    "+": OP_ADD, "-": OP_SUB, "*": OP_MUL, "^": OP_XOR,
    "&": OP_AND, "|": OP_OR, "<<": OP_SHL, ">>": OP_SHR,
    "%": OP_MOD, "<": OP_LT, "==": OP_EQ, "//": OP_DIV,
}


def compile_program(program):
    """
    Convert instruction list to flat numpy arrays for numba.
    Each instruction becomes: [type, op, dest, src1, src2, src3]
    Bundle ends marked with INSTR_BUNDLE_END.
    """
    compiled = []

    for bundle in program:
        bundle_has_ops = False

        for engine, slots in bundle.items():
            if engine == "debug":
                continue

            for slot in slots:
                op_name = slot[0]
                bundle_has_ops = True

                if engine == "alu":
                    op = OP_MAP.get(op_name, -1)
                    compiled.append([INSTR_ALU, op, slot[1], slot[2], slot[3], 0])

                elif engine == "valu":
                    if op_name == "vbroadcast":
                        compiled.append([INSTR_VALU_BROADCAST, 0, slot[1], slot[2], 0, 0])
                    elif op_name == "multiply_add":
                        compiled.append([INSTR_VALU_MULADD, 0, slot[1], slot[2], slot[3], slot[4]])
                    else:
                        op = OP_MAP.get(op_name, -1)
                        compiled.append([INSTR_VALU, op, slot[1], slot[2], slot[3], 0])

                elif engine == "load":
                    if op_name == "const":
                        compiled.append([INSTR_LOAD_CONST, 0, slot[1], slot[2], 0, 0])
                    elif op_name == "load":
                        compiled.append([INSTR_LOAD, 0, slot[1], slot[2], 0, 0])
                    elif op_name == "vload":
                        compiled.append([INSTR_VLOAD, 0, slot[1], slot[2], 0, 0])

                elif engine == "store":
                    if op_name == "store":
                        compiled.append([INSTR_STORE, 0, slot[1], slot[2], 0, 0])
                    elif op_name == "vstore":
                        compiled.append([INSTR_VSTORE, 0, slot[1], slot[2], 0, 0])

                elif engine == "flow":
                    if op_name == "pause":
                        compiled.append([INSTR_FLOW_PAUSE, 0, 0, 0, 0, 0])
                    elif op_name == "halt":
                        compiled.append([INSTR_FLOW_HALT, 0, 0, 0, 0, 0])
                    elif op_name == "select":
                        compiled.append([INSTR_FLOW_SELECT, 0, slot[1], slot[2], slot[3], slot[4]])
                    elif op_name == "vselect":
                        compiled.append([INSTR_FLOW_VSELECT, 0, slot[1], slot[2], slot[3], slot[4]])

        if bundle_has_ops:
            compiled.append([INSTR_BUNDLE_END, 0, 0, 0, 0, 0])

    return np.array(compiled, dtype=np.int64)


@njit(cache=True)
def alu_op(op, v1, v2):
    """Execute a single ALU operation"""
    MASK = 0xFFFFFFFF
    if op == OP_ADD:
        return (v1 + v2) & MASK
    elif op == OP_SUB:
        return (v1 - v2) & MASK
    elif op == OP_MUL:
        return (v1 * v2) & MASK
    elif op == OP_XOR:
        return v1 ^ v2
    elif op == OP_AND:
        return v1 & v2
    elif op == OP_OR:
        return v1 | v2
    elif op == OP_SHL:
        return (v1 << v2) & MASK
    elif op == OP_SHR:
        return v1 >> v2
    elif op == OP_MOD:
        return v1 % v2 if v2 != 0 else 0
    elif op == OP_LT:
        return 1 if v1 < v2 else 0
    elif op == OP_EQ:
        return 1 if v1 == v2 else 0
    elif op == OP_DIV:
        return v1 // v2 if v2 != 0 else 0
    return 0


@njit(cache=True)
def run_compiled(compiled, mem, scratch, max_cycles, start_cycle, start_pc, enable_pause):
    """
    JIT-compiled execution loop.

    Returns: (final_cycle, final_pc, paused)
    """
    MASK = 0xFFFFFFFF
    pc = start_pc
    cycle = start_cycle
    n_instrs = len(compiled)

    # Pre-allocate write buffers (fixed size, track count)
    MAX_WRITES = 256
    scratch_write_addrs = np.zeros(MAX_WRITES, dtype=np.int64)
    scratch_write_vals = np.zeros(MAX_WRITES, dtype=np.int64)
    mem_write_addrs = np.zeros(MAX_WRITES, dtype=np.int64)
    mem_write_vals = np.zeros(MAX_WRITES, dtype=np.int64)

    while pc < n_instrs:
        if max_cycles > 0 and cycle >= max_cycles:
            return cycle, pc, False

        # Process one bundle
        n_scratch_writes = 0
        n_mem_writes = 0
        paused = False

        while pc < n_instrs:
            instr = compiled[pc]
            itype = instr[0]
            pc += 1

            if itype == INSTR_BUNDLE_END:
                break

            op = instr[1]
            dest = instr[2]
            src1 = instr[3]
            src2 = instr[4]
            src3 = instr[5]

            if itype == INSTR_ALU:
                v1 = scratch[src1]
                v2 = scratch[src2]
                result = alu_op(op, v1, v2)
                scratch_write_addrs[n_scratch_writes] = dest
                scratch_write_vals[n_scratch_writes] = result
                n_scratch_writes += 1

            elif itype == INSTR_VALU:
                for i in range(8):  # VLEN=8
                    v1 = scratch[src1 + i]
                    v2 = scratch[src2 + i]
                    result = alu_op(op, v1, v2)
                    scratch_write_addrs[n_scratch_writes] = dest + i
                    scratch_write_vals[n_scratch_writes] = result
                    n_scratch_writes += 1

            elif itype == INSTR_VALU_BROADCAST:
                val = scratch[src1]
                for i in range(8):
                    scratch_write_addrs[n_scratch_writes] = dest + i
                    scratch_write_vals[n_scratch_writes] = val
                    n_scratch_writes += 1

            elif itype == INSTR_VALU_MULADD:
                for i in range(8):
                    a = scratch[src1 + i]
                    b = scratch[src2 + i]
                    c = scratch[src3 + i]
                    result = ((a * b) + c) & MASK
                    scratch_write_addrs[n_scratch_writes] = dest + i
                    scratch_write_vals[n_scratch_writes] = result
                    n_scratch_writes += 1

            elif itype == INSTR_LOAD_CONST:
                scratch_write_addrs[n_scratch_writes] = dest
                scratch_write_vals[n_scratch_writes] = src1 & MASK
                n_scratch_writes += 1

            elif itype == INSTR_LOAD:
                addr = scratch[src1]
                scratch_write_addrs[n_scratch_writes] = dest
                scratch_write_vals[n_scratch_writes] = mem[addr]
                n_scratch_writes += 1

            elif itype == INSTR_VLOAD:
                base = scratch[src1]
                for i in range(8):
                    scratch_write_addrs[n_scratch_writes] = dest + i
                    scratch_write_vals[n_scratch_writes] = mem[base + i]
                    n_scratch_writes += 1

            elif itype == INSTR_STORE:
                addr = scratch[dest]
                mem_write_addrs[n_mem_writes] = addr
                mem_write_vals[n_mem_writes] = scratch[src1]
                n_mem_writes += 1

            elif itype == INSTR_VSTORE:
                base = scratch[dest]
                for i in range(8):
                    mem_write_addrs[n_mem_writes] = base + i
                    mem_write_vals[n_mem_writes] = scratch[src1 + i]
                    n_mem_writes += 1

            elif itype == INSTR_FLOW_PAUSE:
                if enable_pause:
                    paused = True

            elif itype == INSTR_FLOW_HALT:
                # Commit writes and return
                for i in range(n_scratch_writes):
                    scratch[scratch_write_addrs[i]] = scratch_write_vals[i]
                for i in range(n_mem_writes):
                    mem[mem_write_addrs[i]] = mem_write_vals[i]
                return cycle, pc, False

            elif itype == INSTR_FLOW_SELECT:
                cond = scratch[src1]
                a = scratch[src2]
                b = scratch[src3]
                scratch_write_addrs[n_scratch_writes] = dest
                scratch_write_vals[n_scratch_writes] = a if cond != 0 else b
                n_scratch_writes += 1

            elif itype == INSTR_FLOW_VSELECT:
                for i in range(8):
                    cond = scratch[src1 + i]
                    a = scratch[src2 + i]
                    b = scratch[src3 + i]
                    scratch_write_addrs[n_scratch_writes] = dest + i
                    scratch_write_vals[n_scratch_writes] = a if cond != 0 else b
                    n_scratch_writes += 1

        # Commit writes
        for i in range(n_scratch_writes):
            scratch[scratch_write_addrs[i]] = scratch_write_vals[i]
        for i in range(n_mem_writes):
            mem[mem_write_addrs[i]] = mem_write_vals[i]

        cycle += 1

        if paused:
            return cycle, pc, True

    return cycle, pc, False


class NumbaMachine:
    """Wrapper around numba-compiled simulator"""

    def __init__(self, mem_dump, program, debug_info=None, n_cores=1):
        self.mem = np.array(mem_dump, dtype=np.int64)
        self.scratch = np.zeros(SCRATCH_SIZE, dtype=np.int64)
        self.compiled = compile_program(program)
        self.cycle = 0
        self.enable_pause = True
        self._paused = False
        self._pc = 0

    def run(self, max_cycles=0):
        if self._paused:
            self._paused = False

        # Run the JIT-compiled loop
        self.cycle, self._pc, self._paused = run_compiled(
            self.compiled,
            self.mem,
            self.scratch,
            max_cycles,
            self.cycle,
            self._pc,
            self.enable_pause
        )


def test_numba_machine():
    """Compare numba vs original"""
    from problem import Tree, Input, build_mem_image, reference_kernel2, Machine, DebugInfo
    import random

    random.seed(42)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    from gp_optimizer import seeded_program, GPKernelBuilder

    prog = seeded_program('minimal_flush')
    builder = GPKernelBuilder(prog)
    instrs = builder.build(forest.height, len(forest.values), 256, 16)

    print(f"Testing with {len(instrs)} instructions...")

    # Compile once (warmup)
    print("Compiling (first run includes JIT compilation)...")
    compiled = compile_program(instrs)
    print(f"Compiled to {len(compiled)} micro-ops")

    # Warmup JIT
    test_mem = np.array(mem, dtype=np.int64)
    test_scratch = np.zeros(SCRATCH_SIZE, dtype=np.int64)
    run_compiled(compiled, test_mem, test_scratch, 100, 0, 0, True)
    print("JIT warmup complete")

    # Test original
    mem1 = list(mem)
    orig_debug = DebugInfo(scratch_map=builder.scratch_debug)
    orig = Machine(mem1, instrs, orig_debug, n_cores=1)
    orig.enable_pause = True

    start = time.time()
    ref_gen = reference_kernel2(mem1, {})
    for _ in ref_gen:
        orig.run(max_cycles=50000)
    orig_time = time.time() - start
    print(f"Original: {orig.cycle} cycles in {orig_time:.3f}s")

    # Test numba
    mem2 = list(mem)
    numba_m = NumbaMachine(mem2, instrs)
    numba_m.enable_pause = True

    start = time.time()
    ref_gen = reference_kernel2(mem2, {})
    for _ in ref_gen:
        numba_m.run(max_cycles=50000)
    numba_time = time.time() - start
    print(f"Numba:    {numba_m.cycle} cycles in {numba_time:.3f}s")

    print(f"Speedup: {orig_time/numba_time:.2f}x")

    # Verify
    inp_values_p = mem1[6]
    orig_result = orig.mem[inp_values_p:inp_values_p+256]
    numba_result = list(numba_m.mem[inp_values_p:inp_values_p+256])
    assert orig_result == numba_result, "Result mismatch!"
    print("Results match!")


if __name__ == "__main__":
    test_numba_machine()
