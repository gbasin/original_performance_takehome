"""
NumPy-accelerated VLIW simulator.

Replaces Python loops with vectorized numpy operations.
"""

import numpy as np
from typing import Literal
from enum import Enum
from dataclasses import dataclass

Engine = Literal["alu", "load", "store", "flow", "valu", "debug"]
Instruction = dict[Engine, list[tuple]]

VLEN = 8
SCRATCH_SIZE = 1536

SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}


class CoreState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@dataclass
class DebugInfo:
    scratch_map: dict[int, tuple[str, int]]


class FastMachine:
    """
    NumPy-accelerated VLIW simulator.

    Key optimizations:
    - Memory as numpy arrays (not Python lists)
    - Vector ops use numpy slicing (no Python loops)
    - Precompute instruction dispatch
    """

    def __init__(
        self,
        mem_dump: list[int],
        program: list[Instruction],
        debug_info: DebugInfo,
        n_cores: int = 1,
    ):
        # Use numpy arrays for memory
        self.mem = np.array(mem_dump, dtype=np.int64)
        self.scratch = np.zeros(SCRATCH_SIZE, dtype=np.int64)

        self.program = program
        self.debug_info = debug_info
        self.pc = 0
        self.state = CoreState.RUNNING
        self.cycle = 0
        self.enable_pause = True

        # Mask for 32-bit wraparound
        self.MASK = (1 << 32) - 1

    def run(self, max_cycles: int = 0):
        if self.state == CoreState.PAUSED:
            self.state = CoreState.RUNNING

        scratch = self.scratch
        mem = self.mem
        program = self.program
        MASK = self.MASK

        while self.state == CoreState.RUNNING:
            if max_cycles > 0 and self.cycle >= max_cycles:
                self.state = CoreState.STOPPED
                break

            if self.pc >= len(program):
                self.state = CoreState.STOPPED
                break

            instr = program[self.pc]
            self.pc += 1

            # Temporary write buffers
            scratch_writes = {}
            mem_writes = {}
            has_non_debug = False

            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                has_non_debug = True

                for slot in slots:
                    op = slot[0]

                    if engine == "alu":
                        dest, a1, a2 = slot[1], slot[2], slot[3]
                        v1, v2 = scratch[a1], scratch[a2]

                        if op == "+":
                            res = (v1 + v2) & MASK
                        elif op == "-":
                            res = (v1 - v2) & MASK
                        elif op == "*":
                            res = (v1 * v2) & MASK
                        elif op == "^":
                            res = v1 ^ v2
                        elif op == "&":
                            res = v1 & v2
                        elif op == "|":
                            res = v1 | v2
                        elif op == "<<":
                            res = (v1 << v2) & MASK
                        elif op == ">>":
                            res = v1 >> v2
                        elif op == "%":
                            res = v1 % v2 if v2 != 0 else 0
                        elif op == "<":
                            res = 1 if v1 < v2 else 0
                        elif op == "==":
                            res = 1 if v1 == v2 else 0
                        elif op == "//":
                            res = v1 // v2 if v2 != 0 else 0
                        else:
                            raise ValueError(f"Unknown ALU op: {op}")

                        scratch_writes[dest] = res

                    elif engine == "valu":
                        if op == "vbroadcast":
                            dest, src = slot[1], slot[2]
                            val = scratch[src]
                            # Vectorized write
                            for i in range(VLEN):
                                scratch_writes[dest + i] = val

                        elif op == "multiply_add":
                            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
                            # Vectorized multiply-add
                            va = scratch[a:a+VLEN]
                            vb = scratch[b:b+VLEN]
                            vc = scratch[c:c+VLEN]
                            result = ((va * vb) + vc) & MASK
                            for i in range(VLEN):
                                scratch_writes[dest + i] = result[i]

                        else:
                            # Binary vector op
                            dest, a1, a2 = slot[1], slot[2], slot[3]
                            v1 = scratch[a1:a1+VLEN]
                            v2 = scratch[a2:a2+VLEN]

                            if op == "+":
                                result = (v1 + v2) & MASK
                            elif op == "-":
                                result = (v1 - v2) & MASK
                            elif op == "*":
                                result = (v1 * v2) & MASK
                            elif op == "^":
                                result = v1 ^ v2
                            elif op == "&":
                                result = v1 & v2
                            elif op == "|":
                                result = v1 | v2
                            elif op == "<<":
                                result = (v1 << v2) & MASK
                            elif op == ">>":
                                result = v1 >> v2
                            elif op == "%":
                                result = np.where(v2 != 0, v1 % v2, 0)
                            elif op == "<":
                                result = (v1 < v2).astype(np.int64)
                            elif op == "==":
                                result = (v1 == v2).astype(np.int64)
                            else:
                                raise ValueError(f"Unknown VALU op: {op}")

                            for i in range(VLEN):
                                scratch_writes[dest + i] = result[i]

                    elif engine == "load":
                        if op == "const":
                            dest, val = slot[1], slot[2]
                            scratch_writes[dest] = val & MASK

                        elif op == "load":
                            dest, addr = slot[1], slot[2]
                            scratch_writes[dest] = mem[scratch[addr]]

                        elif op == "vload":
                            dest, addr = slot[1], slot[2]
                            base = scratch[addr]
                            vals = mem[base:base+VLEN]
                            for i in range(VLEN):
                                scratch_writes[dest + i] = vals[i]

                    elif engine == "store":
                        if op == "store":
                            addr, src = slot[1], slot[2]
                            mem_writes[scratch[addr]] = scratch[src]

                        elif op == "vstore":
                            addr, src = slot[1], slot[2]
                            base = scratch[addr]
                            for i in range(VLEN):
                                mem_writes[base + i] = scratch[src + i]

                    elif engine == "flow":
                        if op == "pause":
                            if self.enable_pause:
                                self.state = CoreState.PAUSED

                        elif op == "halt":
                            self.state = CoreState.STOPPED

                        elif op == "select":
                            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                            scratch_writes[dest] = scratch[a] if scratch[cond] != 0 else scratch[b]

                        elif op == "vselect":
                            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                            vc = scratch[cond:cond+VLEN]
                            va = scratch[a:a+VLEN]
                            vb = scratch[b:b+VLEN]
                            result = np.where(vc != 0, va, vb)
                            for i in range(VLEN):
                                scratch_writes[dest + i] = result[i]

                        elif op == "add_imm":
                            dest, a, imm = slot[1], slot[2], slot[3]
                            scratch_writes[dest] = (scratch[a] + imm) & MASK

                        elif op == "jump":
                            self.pc = slot[1]

                        elif op == "cond_jump":
                            cond, addr = slot[1], slot[2]
                            if scratch[cond] != 0:
                                self.pc = addr

            # Commit writes
            for addr, val in scratch_writes.items():
                scratch[addr] = val
            for addr, val in mem_writes.items():
                mem[addr] = val

            if has_non_debug:
                self.cycle += 1


def test_fast_machine():
    """Quick test to verify FastMachine matches original"""
    from problem import Tree, Input, build_mem_image, reference_kernel2, Machine, DebugInfo as OrigDebugInfo
    import random
    import time

    random.seed(42)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    # Use a simple seeded program from GP
    from gp_optimizer import seeded_program, GPKernelBuilder

    prog = seeded_program('minimal_flush')
    builder = GPKernelBuilder(prog)
    instrs = builder.build(forest.height, len(forest.values), 256, 16)

    print(f"Testing with {len(instrs)} instructions...")

    # Test original
    mem1 = list(mem)
    orig_debug = OrigDebugInfo(scratch_map=builder.scratch_debug)
    orig = Machine(mem1, instrs, orig_debug, n_cores=1)
    orig.enable_pause = True

    start = time.time()
    ref_gen = reference_kernel2(mem1, {})
    for _ in ref_gen:
        orig.run(max_cycles=50000)
    orig_time = time.time() - start
    print(f"Original: {orig.cycle} cycles in {orig_time:.3f}s")

    # Test fast
    mem2 = list(mem)
    fast_debug = DebugInfo(scratch_map=builder.scratch_debug)
    fast = FastMachine(mem2, instrs, fast_debug)
    fast.enable_pause = True

    start = time.time()
    ref_gen = reference_kernel2(mem2, {})
    for _ in ref_gen:
        fast.run(max_cycles=50000)
    fast_time = time.time() - start
    print(f"Fast:     {fast.cycle} cycles in {fast_time:.3f}s")

    print(f"Speedup: {orig_time/fast_time:.2f}x")

    # Verify results match
    assert orig.cycle == fast.cycle, f"Cycle mismatch: {orig.cycle} vs {fast.cycle}"

    inp_values_p = mem1[6]
    orig_result = orig.mem[inp_values_p:inp_values_p+256]
    fast_result = list(fast.mem[inp_values_p:inp_values_p+256])
    assert orig_result == fast_result, "Result mismatch!"

    print("Results match!")


if __name__ == "__main__":
    test_fast_machine()
