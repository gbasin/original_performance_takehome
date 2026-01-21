"""
Genetic Algorithm optimizer for VLIW kernel performance.

This module uses a parameterized approach where the genome controls
high-level transformation parameters, and a code generator always
produces valid VLIW programs.

Usage:
    python ga_optimizer.py [--generations N] [--population N] [--workers N]
"""

from dataclasses import dataclass, field, fields, asdict
from typing import List, Tuple, Optional, Any, Callable
from enum import Enum
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import random
import json
import time
import argparse

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
    build_mem_image,
    reference_kernel2,
)


# =============================================================================
# Genome Definition
# =============================================================================

class ScheduleStrategy(Enum):
    GREEDY = "greedy"
    CRITICAL_PATH = "critical_path"
    BALANCED = "balanced"
    LOAD_FIRST = "load_first"


class LoopOrder(Enum):
    ROUND_BATCH = "round_batch"  # for r in rounds: for b in batch
    BATCH_ROUND = "batch_round"  # for b in batch: for r in rounds


class ScratchAllocation(Enum):
    COMPACT = "compact"
    ALIGNED = "aligned"       # Align vectors to VLEN boundaries
    SEPARATED = "separated"   # Separate scalar/vector regions


@dataclass
class Genome:
    """
    Parameterized representation of kernel optimizations.
    All parameters are bounded to ensure valid code generation.
    """
    # Loop transformations
    batch_unroll_factor: int = 1       # 1, 2, 4, 8 (must divide batch_size)
    round_unroll_factor: int = 1       # 1, 2, 4, 8, 16
    loop_order: LoopOrder = LoopOrder.ROUND_BATCH

    # Vectorization (VLEN = 8)
    vectorize_batch: bool = False      # Process VLEN batch elements together
    vectorize_hash: bool = False       # Vectorize hash computation

    # Scheduling hints
    schedule_strategy: ScheduleStrategy = ScheduleStrategy.GREEDY
    max_bundle_fill: float = 1.0       # 0.5-1.0, how full to pack bundles
    load_ahead_distance: int = 0       # Cycles to issue loads early

    # Memory/scratch layout
    scratch_allocation: ScratchAllocation = ScratchAllocation.COMPACT
    preload_constants: bool = True     # Load all constants upfront

    # Instruction-level tweaks
    use_add_imm: bool = False          # Use add_imm instead of const+add where possible
    fuse_hash_stages: bool = False     # Try to pack hash stages together

    def __post_init__(self):
        """Clamp values to valid ranges"""
        self.batch_unroll_factor = max(1, min(32, self.batch_unroll_factor))
        self.round_unroll_factor = max(1, min(16, self.round_unroll_factor))
        self.max_bundle_fill = max(0.05, min(1.0, self.max_bundle_fill))  # Allow very low for safe mode
        self.load_ahead_distance = max(0, min(10, self.load_ahead_distance))


def random_genome() -> Genome:
    """Generate a random valid genome"""
    return Genome(
        batch_unroll_factor=random.choice([1, 2, 4, 8]),
        round_unroll_factor=random.choice([1, 2, 4, 8, 16]),
        loop_order=random.choice(list(LoopOrder)),
        vectorize_batch=False,  # Disable for now - needs gather support
        vectorize_hash=random.choice([True, False]),
        schedule_strategy=random.choice(list(ScheduleStrategy)),
        max_bundle_fill=random.uniform(0.5, 1.0),  # Dependency-aware scheduling enabled
        load_ahead_distance=random.randint(0, 5),
        scratch_allocation=random.choice(list(ScratchAllocation)),
        preload_constants=random.choice([True, False]),
        use_add_imm=random.choice([True, False]),
        fuse_hash_stages=random.choice([True, False]),
    )


def baseline_genome() -> Genome:
    """Genome that approximates the baseline scalar implementation"""
    return Genome(
        batch_unroll_factor=1,
        round_unroll_factor=1,
        loop_order=LoopOrder.ROUND_BATCH,
        vectorize_batch=False,
        vectorize_hash=False,
        schedule_strategy=ScheduleStrategy.GREEDY,
        max_bundle_fill=1.0,  # Use dependency-aware scheduling
        load_ahead_distance=0,
        scratch_allocation=ScratchAllocation.COMPACT,
        preload_constants=True,
        use_add_imm=False,
        fuse_hash_stages=False,
    )


# =============================================================================
# Parameterized Kernel Builder
# =============================================================================

class ParameterizedKernelBuilder:
    """
    Builds VLIW kernels based on genome parameters.
    Always produces valid code by construction.
    """

    def __init__(self, genome: Genome):
        self.genome = genome
        self.instrs: List[dict] = []
        self.scratch: dict = {}
        self.scratch_debug: dict = {}
        self.scratch_ptr = 0
        self.const_map: dict = {}

        # For scheduling
        self.pending_slots: List[Tuple[str, tuple]] = []

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name: Optional[str] = None, length: int = 1) -> int:
        """Allocate scratch space, respecting alignment if configured"""
        if self.genome.scratch_allocation == ScratchAllocation.ALIGNED and length >= VLEN:
            # Align to VLEN boundary
            if self.scratch_ptr % VLEN != 0:
                self.scratch_ptr += VLEN - (self.scratch_ptr % VLEN)

        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vector_scratch(self, name: Optional[str] = None) -> int:
        """Allocate VLEN-sized scratch space"""
        return self.alloc_scratch(name, VLEN)

    def emit(self, engine: str, slot: tuple):
        """Add a slot to pending operations"""
        self.pending_slots.append((engine, slot))

    def emit_const(self, dest: int, val: int):
        """Emit a constant load, with caching"""
        if val not in self.const_map:
            self.emit("load", ("const", dest, val))
            self.const_map[val] = dest
        elif self.const_map[val] != dest:
            # Copy from cached location
            self.emit("alu", ("+", dest, self.const_map[val], self.get_const(0)))

    def get_const(self, val: int) -> int:
        """Get address of a constant, allocating and loading if needed"""
        if val not in self.const_map:
            addr = self.alloc_scratch(f"const_{val}")
            self.const_map[val] = addr
            # Must emit the const load instruction!
            self.instrs.append({"load": [("const", addr, val)]})
        return self.const_map[val]

    def _get_slot_reads_writes(self, engine: str, slot: tuple) -> Tuple[set, set]:
        """
        Extract scratch addresses read and written by a slot.
        Returns (reads, writes) as sets of addresses.
        """
        reads = set()
        writes = set()
        op = slot[0]

        if engine == "alu":
            # Format: (op, dest, src1, src2)
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])

        elif engine == "valu":
            if op == "vbroadcast":
                # (vbroadcast, dest, src) - dest is VLEN addresses
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
            elif op == "multiply_add":
                # (multiply_add, dest, a, b, c)
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
            else:
                # (op, dest, src1, src2) - all VLEN
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)

        elif engine == "load":
            if op == "const":
                # (const, dest, value)
                writes.add(slot[1])
            elif op == "load":
                # (load, dest, addr_scratch)
                writes.add(slot[1])
                reads.add(slot[2])  # Address comes from scratch
            elif op == "load_offset":
                # (load_offset, dest, addr, offset)
                writes.add(slot[1] + slot[3])
                reads.add(slot[2] + slot[3])
            elif op == "vload":
                # (vload, dest, addr_scratch) - dest is VLEN addresses
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])

        elif engine == "store":
            if op == "store":
                # (store, addr_scratch, src_scratch)
                reads.add(slot[1])  # Address
                reads.add(slot[2])  # Value
            elif op == "vstore":
                # (vstore, addr_scratch, src_scratch)
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)

        elif engine == "flow":
            if op == "select":
                # (select, dest, cond, a, b)
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
            elif op == "vselect":
                # (vselect, dest, cond, a, b) - all VLEN
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
            elif op == "add_imm":
                # (add_imm, dest, src, imm)
                writes.add(slot[1])
                reads.add(slot[2])
            elif op in ("cond_jump", "cond_jump_rel"):
                reads.add(slot[1])
            elif op == "jump_indirect":
                reads.add(slot[1])
            elif op == "trace_write":
                reads.add(slot[1])
            elif op == "coreid":
                writes.add(slot[1])
            # halt, pause, jump don't read/write scratch

        return reads, writes

    def _has_dependency(self, slot1_info, slot2_info) -> bool:
        """
        Check if slot2 depends on slot1 (RAW, WAR, or WAW hazard).
        slot_info = [engine, slot, reads, writes, scheduled]
        """
        _, _, _, writes1, _ = slot1_info
        _, _, reads2, writes2, _ = slot2_info

        # RAW: slot2 reads what slot1 writes
        if writes1 & reads2:
            return True
        # WAW: both write same location
        if writes1 & writes2:
            return True
        # WAR: slot2 writes what slot1 reads (but in VLIW this is OK within same cycle)
        # Actually in this VLIW, reads happen before writes, so WAR within cycle is fine
        # But across cycles, if slot2 comes after slot1 and writes what slot1 reads,
        # that's fine because slot1 already read the old value.
        # So we only need RAW and WAW checks.

        return False

    def _schedule_with_dependencies(self, max_fill: float):
        """
        List scheduling that respects data dependencies.
        """
        if not self.pending_slots:
            return

        # Build slot info: (engine, slot, reads, writes, scheduled)
        slots_info = []
        for engine, slot in self.pending_slots:
            reads, writes = self._get_slot_reads_writes(engine, slot)
            slots_info.append([engine, slot, reads, writes, False])

        # Build dependency graph (who must come before whom)
        n = len(slots_info)
        must_precede = [set() for _ in range(n)]  # must_precede[i] = slots that must come before i

        for i in range(n):
            for j in range(i + 1, n):
                if self._has_dependency(slots_info[i], slots_info[j]):
                    must_precede[j].add(i)

        # List scheduling
        scheduled = [False] * n
        while not all(scheduled):
            # Find all ready slots (predecessors all scheduled)
            ready = []
            for i in range(n):
                if not scheduled[i]:
                    if all(scheduled[p] for p in must_precede[i]):
                        ready.append(i)

            if not ready:
                # Cycle detected or bug - fall back to safe mode
                for i in range(n):
                    if not scheduled[i]:
                        engine, slot, _, _, _ = slots_info[i]
                        self.instrs.append({engine: [slot]})
                        scheduled[i] = True
                break

            # Pack ready slots into a bundle respecting slot limits
            bundle = {}
            slots_in_bundle = set()
            bundle_writes = set()  # Track writes in this bundle for WAW check

            for i in ready:
                engine, slot, reads, writes, _ = slots_info[i]

                # Check if this slot can go in the current bundle
                limit = int(SLOT_LIMITS.get(engine, 0) * max_fill)
                current_count = len(bundle.get(engine, []))

                # Check WAW within bundle
                if writes & bundle_writes:
                    continue  # Can't add - would cause WAW within bundle

                if current_count < max(1, limit):
                    if engine not in bundle:
                        bundle[engine] = []
                    bundle[engine].append(slot)
                    slots_in_bundle.add(i)
                    bundle_writes |= writes

            # Emit bundle if non-empty
            if bundle:
                self.instrs.append(bundle)
                for i in slots_in_bundle:
                    scheduled[i] = True

    def flush_schedule(self):
        """
        Convert pending slots into scheduled instruction bundles.
        This is where VLIW packing happens.
        """
        if not self.pending_slots:
            return

        strategy = self.genome.schedule_strategy
        max_fill = self.genome.max_bundle_fill

        # Use dependency-aware scheduling for higher fill rates
        if max_fill >= 0.2:
            self._schedule_with_dependencies(max_fill)
        elif strategy == ScheduleStrategy.GREEDY:
            self._schedule_greedy(max_fill)
        elif strategy == ScheduleStrategy.LOAD_FIRST:
            self._schedule_load_first(max_fill)
        elif strategy == ScheduleStrategy.CRITICAL_PATH:
            self._schedule_critical_path(max_fill)
        else:  # BALANCED
            self._schedule_balanced(max_fill)

        self.pending_slots = []

    def _schedule_greedy(self, max_fill: float):
        """Pack slots greedily into bundles"""
        slots = list(self.pending_slots)

        # If max_fill is very low, just emit one slot per instruction (safe mode)
        if max_fill < 0.2:
            for engine, slot in slots:
                self.instrs.append({engine: [slot]})
            return

        while slots:
            bundle = {engine: [] for engine in SLOT_LIMITS.keys()}
            remaining = []

            for engine, slot in slots:
                limit = int(SLOT_LIMITS.get(engine, 0) * max_fill)
                if engine in bundle and len(bundle[engine]) < max(1, limit):
                    bundle[engine].append(slot)
                else:
                    remaining.append((engine, slot))

            # Remove empty engines
            bundle = {k: v for k, v in bundle.items() if v}
            if bundle:
                self.instrs.append(bundle)

            slots = remaining

    def _schedule_load_first(self, max_fill: float):
        """Prioritize load operations"""
        loads = [(e, s) for e, s in self.pending_slots if e == "load"]
        others = [(e, s) for e, s in self.pending_slots if e != "load"]
        self.pending_slots = loads + others
        self._schedule_greedy(max_fill)

    def _schedule_critical_path(self, max_fill: float):
        """Simple critical path heuristic - prioritize ALU chains"""
        # For now, just reverse priority (ALU last means more parallelism)
        alu = [(e, s) for e, s in self.pending_slots if e in ("alu", "valu")]
        others = [(e, s) for e, s in self.pending_slots if e not in ("alu", "valu")]
        self.pending_slots = others + alu
        self._schedule_greedy(max_fill)

    def _schedule_balanced(self, max_fill: float):
        """Try to balance across engines"""
        # Group by engine type
        by_engine = {}
        for e, s in self.pending_slots:
            by_engine.setdefault(e, []).append((e, s))

        # Interleave
        result = []
        while any(by_engine.values()):
            for engine in list(by_engine.keys()):
                if by_engine[engine]:
                    result.append(by_engine[engine].pop(0))
                if not by_engine[engine]:
                    del by_engine[engine]

        self.pending_slots = result
        self._schedule_greedy(max_fill)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int
    ) -> List[dict]:
        """
        Build the kernel based on genome parameters.
        """
        genome = self.genome

        # Allocate temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Header variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p"
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Load header from memory
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.flush_schedule()
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        # Preload constants if enabled
        if genome.preload_constants:
            for val in [0, 1, 2]:
                addr = self.alloc_scratch(f"const_{val}")
                self.const_map[val] = addr
                self.emit("load", ("const", addr, val))
            # Hash constants
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if val1 not in self.const_map:
                    addr = self.alloc_scratch()
                    self.const_map[val1] = addr
                    self.emit("load", ("const", addr, val1))
                if val3 not in self.const_map:
                    addr = self.alloc_scratch()
                    self.const_map[val3] = addr
                    self.emit("load", ("const", addr, val3))
            self.flush_schedule()

        zero_const = self.get_const(0)
        one_const = self.get_const(1)
        two_const = self.get_const(2)

        # Pause for debugging sync
        self.instrs.append({"flow": [("pause",)]})

        # Allocate per-iteration scratch
        if genome.vectorize_batch:
            # Vector registers for batch processing
            v_idx = self.alloc_vector_scratch("v_idx")
            v_val = self.alloc_vector_scratch("v_val")
            v_node_val = self.alloc_vector_scratch("v_node_val")
            v_addr = self.alloc_vector_scratch("v_addr")
            v_tmp1 = self.alloc_vector_scratch("v_tmp1")
            v_tmp2 = self.alloc_vector_scratch("v_tmp2")
            v_tmp3 = self.alloc_vector_scratch("v_tmp3")

        # Scalar registers (also needed for vector case)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Determine unroll factors (must divide evenly)
        batch_unroll = genome.batch_unroll_factor
        while batch_size % batch_unroll != 0 and batch_unroll > 1:
            batch_unroll //= 2

        round_unroll = genome.round_unroll_factor
        while rounds % round_unroll != 0 and round_unroll > 1:
            round_unroll //= 2

        # Effective iterations
        if genome.vectorize_batch:
            batch_iters = batch_size // VLEN // batch_unroll
            batch_step = VLEN
        else:
            batch_iters = batch_size // batch_unroll
            batch_step = 1

        round_iters = rounds // round_unroll

        # Generate loop body based on loop order
        if genome.loop_order == LoopOrder.ROUND_BATCH:
            self._emit_round_batch_loop(
                rounds, round_unroll, round_iters,
                batch_size, batch_unroll, batch_iters, batch_step,
                tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                tmp1, tmp2, tmp3,
                zero_const, one_const, two_const,
                genome
            )
        else:
            self._emit_batch_round_loop(
                rounds, round_unroll, round_iters,
                batch_size, batch_unroll, batch_iters, batch_step,
                tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                tmp1, tmp2, tmp3,
                zero_const, one_const, two_const,
                genome
            )

        # Final pause
        self.instrs.append({"flow": [("pause",)]})

        return self.instrs

    def _emit_round_batch_loop(
        self, rounds, round_unroll, round_iters,
        batch_size, batch_unroll, batch_iters, batch_step,
        tmp_idx, tmp_val, tmp_node_val, tmp_addr,
        tmp1, tmp2, tmp3,
        zero_const, one_const, two_const,
        genome
    ):
        """Emit loop with rounds as outer loop"""
        for r_base in range(0, rounds, round_unroll):
            for r_offset in range(round_unroll):
                r = r_base + r_offset
                if r >= rounds:
                    break

                for b_base in range(0, batch_size, batch_unroll * batch_step):
                    for b_offset in range(batch_unroll):
                        b = b_base + b_offset * batch_step
                        if b >= batch_size:
                            break

                        if genome.vectorize_batch and batch_step == VLEN:
                            self._emit_vectorized_iteration(
                                r, b, tmp1, tmp2, tmp3,
                                zero_const, one_const, two_const
                            )
                        else:
                            self._emit_scalar_iteration(
                                r, b, tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                                tmp1, tmp2, tmp3,
                                zero_const, one_const, two_const
                            )

    def _emit_batch_round_loop(
        self, rounds, round_unroll, round_iters,
        batch_size, batch_unroll, batch_iters, batch_step,
        tmp_idx, tmp_val, tmp_node_val, tmp_addr,
        tmp1, tmp2, tmp3,
        zero_const, one_const, two_const,
        genome
    ):
        """Emit loop with batch as outer loop"""
        for b_base in range(0, batch_size, batch_unroll * batch_step):
            for b_offset in range(batch_unroll):
                b = b_base + b_offset * batch_step
                if b >= batch_size:
                    break

                for r_base in range(0, rounds, round_unroll):
                    for r_offset in range(round_unroll):
                        r = r_base + r_offset
                        if r >= rounds:
                            break

                        if genome.vectorize_batch and batch_step == VLEN:
                            self._emit_vectorized_iteration(
                                r, b, tmp1, tmp2, tmp3,
                                zero_const, one_const, two_const
                            )
                        else:
                            self._emit_scalar_iteration(
                                r, b, tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                                tmp1, tmp2, tmp3,
                                zero_const, one_const, two_const
                            )

    def _emit_scalar_iteration(
        self, round_idx: int, batch_idx: int,
        tmp_idx, tmp_val, tmp_node_val, tmp_addr,
        tmp1, tmp2, tmp3,
        zero_const, one_const, two_const
    ):
        """Emit one scalar iteration of the kernel"""
        genome = self.genome
        i_const_addr = self.get_const(batch_idx)

        # idx = mem[inp_indices_p + i]
        if genome.use_add_imm and batch_idx < 256:
            self.emit("flow", ("add_imm", tmp_addr, self.scratch["inp_indices_p"], batch_idx))
        else:
            self.emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const_addr))
        self.flush_schedule()
        self.emit("load", ("load", tmp_idx, tmp_addr))

        # val = mem[inp_values_p + i]
        if genome.use_add_imm and batch_idx < 256:
            self.emit("flow", ("add_imm", tmp_addr, self.scratch["inp_values_p"], batch_idx))
        else:
            self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const_addr))
        self.flush_schedule()
        self.emit("load", ("load", tmp_val, tmp_addr))
        self.flush_schedule()

        # node_val = mem[forest_values_p + idx]
        self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
        self.flush_schedule()
        self.emit("load", ("load", tmp_node_val, tmp_addr))
        self.flush_schedule()

        # val = myhash(val ^ node_val)
        self.emit("alu", ("^", tmp_val, tmp_val, tmp_node_val))
        self.flush_schedule()

        # Hash stages
        self._emit_hash(tmp_val, tmp1, tmp2, genome.fuse_hash_stages)

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        self.emit("alu", ("%", tmp1, tmp_val, two_const))
        self.emit("alu", ("==", tmp1, tmp1, zero_const))
        self.flush_schedule()
        self.emit("flow", ("select", tmp3, tmp1, one_const, two_const))
        self.emit("alu", ("*", tmp_idx, tmp_idx, two_const))
        self.flush_schedule()
        self.emit("alu", ("+", tmp_idx, tmp_idx, tmp3))
        self.flush_schedule()

        # idx = 0 if idx >= n_nodes else idx
        self.emit("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
        self.flush_schedule()
        self.emit("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))
        self.flush_schedule()

        # mem[inp_indices_p + i] = idx
        if genome.use_add_imm and batch_idx < 256:
            self.emit("flow", ("add_imm", tmp_addr, self.scratch["inp_indices_p"], batch_idx))
        else:
            self.emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const_addr))
        self.flush_schedule()
        self.emit("store", ("store", tmp_addr, tmp_idx))

        # mem[inp_values_p + i] = val
        if genome.use_add_imm and batch_idx < 256:
            self.emit("flow", ("add_imm", tmp_addr, self.scratch["inp_values_p"], batch_idx))
        else:
            self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const_addr))
        self.flush_schedule()
        self.emit("store", ("store", tmp_addr, tmp_val))
        self.flush_schedule()

    def _emit_vectorized_iteration(
        self, round_idx: int, batch_idx: int,
        tmp1, tmp2, tmp3,
        zero_const, one_const, two_const
    ):
        """Emit one vectorized iteration processing VLEN elements"""
        # Get vector scratch addresses
        v_idx = self.scratch["v_idx"]
        v_val = self.scratch["v_val"]
        v_node_val = self.scratch["v_node_val"]
        v_addr = self.scratch["v_addr"]
        v_tmp1 = self.scratch["v_tmp1"]
        v_tmp2 = self.scratch["v_tmp2"]

        base_addr = self.alloc_scratch()

        # Load base address for this batch chunk
        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_indices_p"], base_addr))
        self.flush_schedule()

        # vload indices
        self.emit("load", ("vload", v_idx, base_addr))
        self.flush_schedule()

        # Load values
        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_values_p"], base_addr))
        self.flush_schedule()
        self.emit("load", ("vload", v_val, base_addr))
        self.flush_schedule()

        # For node_val, we need gather (not directly supported, so fall back to scalar loads)
        # This is a limitation - true vectorization needs gather support
        # For now, emit scalar loads for node values
        tmp_addr = self.alloc_scratch()
        for vi in range(VLEN):
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], v_idx + vi))
            self.flush_schedule()
            self.emit("load", ("load", v_node_val + vi, tmp_addr))
            self.flush_schedule()

        # val = val ^ node_val (vectorized)
        self.emit("valu", ("^", v_val, v_val, v_node_val))
        self.flush_schedule()

        # Vectorized hash
        self._emit_vector_hash(v_val, v_tmp1, v_tmp2)

        # Compute next indices (vectorized)
        self.emit("valu", ("vbroadcast", v_tmp1, two_const))
        self.flush_schedule()
        self.emit("valu", ("%", v_tmp2, v_val, v_tmp1))  # val % 2
        self.emit("valu", ("vbroadcast", v_tmp1, zero_const))
        self.flush_schedule()
        self.emit("valu", ("==", v_tmp2, v_tmp2, v_tmp1))  # == 0
        self.emit("valu", ("vbroadcast", v_tmp1, one_const))
        self.flush_schedule()

        # Can't easily do vselect with scalars, need broadcasts
        v_one = self.alloc_vector_scratch()
        v_two = self.alloc_vector_scratch()
        self.emit("valu", ("vbroadcast", v_one, one_const))
        self.emit("valu", ("vbroadcast", v_two, two_const))
        self.flush_schedule()

        self.emit("flow", ("vselect", v_tmp1, v_tmp2, v_one, v_two))  # 1 or 2
        self.emit("valu", ("*", v_idx, v_idx, v_two))  # 2*idx
        self.flush_schedule()
        self.emit("valu", ("+", v_idx, v_idx, v_tmp1))  # + offset
        self.flush_schedule()

        # Wrap check (vectorized)
        self.emit("valu", ("vbroadcast", v_tmp1, self.scratch["n_nodes"]))
        self.flush_schedule()
        self.emit("valu", ("<", v_tmp2, v_idx, v_tmp1))
        self.emit("valu", ("vbroadcast", v_tmp1, zero_const))
        self.flush_schedule()
        self.emit("flow", ("vselect", v_idx, v_tmp2, v_idx, v_tmp1))
        self.flush_schedule()

        # Store results
        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_indices_p"], base_addr))
        self.flush_schedule()
        self.emit("store", ("vstore", base_addr, v_idx))

        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_values_p"], base_addr))
        self.flush_schedule()
        self.emit("store", ("vstore", base_addr, v_val))
        self.flush_schedule()

    def _emit_hash(self, val_addr: int, tmp1: int, tmp2: int, fuse: bool = False):
        """Emit scalar hash computation"""
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.emit("alu", (op1, tmp1, val_addr, self.get_const(val1)))
            self.emit("alu", (op3, tmp2, val_addr, self.get_const(val3)))
            if fuse:
                pass  # Don't flush, let scheduler pack them
            else:
                self.flush_schedule()
            self.emit("alu", (op2, val_addr, tmp1, tmp2))
            self.flush_schedule()

    def _emit_vector_hash(self, v_val: int, v_tmp1: int, v_tmp2: int):
        """Emit vectorized hash computation"""
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # Need to broadcast constants to vectors
            v_const1 = self.alloc_vector_scratch()
            v_const3 = self.alloc_vector_scratch()
            self.emit("valu", ("vbroadcast", v_const1, self.get_const(val1)))
            self.emit("valu", ("vbroadcast", v_const3, self.get_const(val3)))
            self.flush_schedule()

            self.emit("valu", (op1, v_tmp1, v_val, v_const1))
            self.emit("valu", (op3, v_tmp2, v_val, v_const3))
            self.flush_schedule()
            self.emit("valu", (op2, v_val, v_tmp1, v_tmp2))
            self.flush_schedule()


# =============================================================================
# Fitness Evaluation
# =============================================================================

# Test parameters (matching submission_tests.py)
TEST_FOREST_HEIGHT = 10
TEST_ROUNDS = 16
TEST_BATCH_SIZE = 256
TEST_SEED = 123


def evaluate_genome(genome: Genome, timeout_cycles: int = 200000) -> float:
    """
    Evaluate a genome by building and running the kernel.
    Returns cycle count (lower is better), or inf if invalid/timeout.
    """
    try:
        random.seed(TEST_SEED)
        forest = Tree.generate(TEST_FOREST_HEIGHT)
        inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
        mem = build_mem_image(forest, inp)

        # Build kernel from genome
        builder = ParameterizedKernelBuilder(genome)
        instrs = builder.build_kernel(
            forest.height,
            len(forest.values),
            len(inp.indices),
            TEST_ROUNDS
        )

        # Run simulation
        machine = Machine(
            mem,
            instrs,
            builder.debug_info(),
            n_cores=N_CORES,
        )
        machine.enable_pause = True
        machine.enable_debug = False

        # Get reference for correctness check
        value_trace = {}
        ref_gen = reference_kernel2(list(mem), value_trace)
        ref_mem = None
        for ref_mem in ref_gen:
            machine.run()
            if machine.cycle > timeout_cycles:
                return float('inf')

        # Check correctness
        if ref_mem is not None:
            inp_values_p = ref_mem[6]
            inp_indices_p = ref_mem[5]

            if machine.mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE] != \
               ref_mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE]:
                return float('inf')

        return float(machine.cycle)

    except Exception as e:
        # Any error means invalid genome
        return float('inf')


def evaluate_genome_wrapper(args):
    """Wrapper for multiprocessing"""
    genome, timeout = args
    return evaluate_genome(genome, timeout)


# =============================================================================
# Genetic Algorithm Operators
# =============================================================================

def tournament_select(population: List[Genome], fitnesses: List[float], k: int = 3) -> Genome:
    """Select individual via tournament selection"""
    indices = random.sample(range(len(population)), k)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best_idx])


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """Uniform crossover between two parents"""
    child = Genome()

    for f in fields(Genome):
        if random.random() < 0.5:
            setattr(child, f.name, getattr(parent1, f.name))
        else:
            setattr(child, f.name, getattr(parent2, f.name))

    return child


def mutate(genome: Genome, rate: float = 0.1) -> Genome:
    """Mutate genome with given probability per gene"""
    genome = deepcopy(genome)

    for f in fields(Genome):
        if random.random() > rate:
            continue

        val = getattr(genome, f.name)

        if isinstance(val, bool):
            setattr(genome, f.name, not val)

        elif isinstance(val, int):
            # Multiply/divide by 2, or small adjustment
            if random.random() < 0.5:
                new_val = val * random.choice([2, 2, 1, 1]) // random.choice([1, 1, 2, 2])
            else:
                new_val = val + random.randint(-2, 2)
            setattr(genome, f.name, max(1, new_val))

        elif isinstance(val, float):
            new_val = val + random.gauss(0, 0.1)
            setattr(genome, f.name, new_val)

        elif isinstance(val, Enum):
            options = list(type(val))
            setattr(genome, f.name, random.choice(options))

    # Re-clamp values
    genome.__post_init__()
    return genome


def local_search(genome: Genome, iterations: int = 5) -> Tuple[Genome, float]:
    """
    Greedy local search: try small perturbations to scheduling params.
    """
    best_genome = genome
    best_fitness = evaluate_genome(genome)

    if best_fitness == float('inf'):
        return genome, best_fitness

    for _ in range(iterations):
        # Try perturbations to scheduling-related params
        candidate = deepcopy(best_genome)

        # Randomly perturb one scheduling parameter
        param = random.choice([
            'schedule_strategy', 'max_bundle_fill', 'load_ahead_distance',
            'fuse_hash_stages', 'preload_constants'
        ])

        val = getattr(candidate, param)
        if isinstance(val, Enum):
            setattr(candidate, param, random.choice(list(type(val))))
        elif isinstance(val, bool):
            setattr(candidate, param, not val)
        elif isinstance(val, float):
            setattr(candidate, param, val + random.gauss(0, 0.05))
        elif isinstance(val, int):
            setattr(candidate, param, val + random.randint(-1, 1))

        candidate.__post_init__()

        fitness = evaluate_genome(candidate)
        if fitness < best_fitness:
            best_genome = candidate
            best_fitness = fitness

    return best_genome, best_fitness


# =============================================================================
# Main GA Loop
# =============================================================================

class GeneticOptimizer:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        tournament_size: int = 3,
        local_search_prob: float = 0.1,
        workers: int = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.local_search_prob = local_search_prob
        self.workers = workers or cpu_count()

        self.population: List[Genome] = []
        self.fitnesses: List[float] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float('inf')
        self.history: List[dict] = []

    def initialize_population(self):
        """Create initial population with mixed seeding"""
        self.population = []

        # 20% seeded from baseline with mutations
        n_seeded = self.population_size // 5
        base = baseline_genome()
        for _ in range(n_seeded):
            self.population.append(mutate(base, rate=0.3))

        # 80% random
        while len(self.population) < self.population_size:
            self.population.append(random_genome())

    def evaluate_population(self):
        """Evaluate all individuals in parallel"""
        timeout = int(self.best_fitness * 2) if self.best_fitness < float('inf') else 200000

        with Pool(processes=self.workers) as pool:
            args = [(g, timeout) for g in self.population]
            self.fitnesses = pool.map(evaluate_genome_wrapper, args)

        # Update best
        for i, fitness in enumerate(self.fitnesses):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = deepcopy(self.population[i])

    def evolve_generation(self):
        """Create next generation"""
        new_population = []

        # Elitism: keep best individuals
        sorted_indices = sorted(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        for i in range(self.elitism):
            if sorted_indices[i] < len(self.population):
                new_population.append(deepcopy(self.population[sorted_indices[i]]))

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = tournament_select(self.population, self.fitnesses, self.tournament_size)
            parent2 = tournament_select(self.population, self.fitnesses, self.tournament_size)

            # Crossover
            if random.random() < self.crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)

            # Mutation
            child = mutate(child, self.mutation_rate)

            # Local search (occasionally)
            if random.random() < self.local_search_prob:
                child, _ = local_search(child, iterations=3)

            new_population.append(child)

        self.population = new_population

    def run(self, verbose: bool = True):
        """Run the genetic algorithm"""
        print(f"Starting GA with {self.population_size} individuals, {self.generations} generations")
        print(f"Using {self.workers} parallel workers")
        print()

        self.initialize_population()

        for gen in range(self.generations):
            start_time = time.time()

            self.evaluate_population()

            # Statistics
            valid_fitnesses = [f for f in self.fitnesses if f < float('inf')]
            if valid_fitnesses:
                avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
                min_fitness = min(valid_fitnesses)
            else:
                avg_fitness = float('inf')
                min_fitness = float('inf')

            gen_time = time.time() - start_time

            stats = {
                'generation': gen,
                'best_fitness': self.best_fitness,
                'gen_best': min_fitness,
                'gen_avg': avg_fitness,
                'valid_count': len(valid_fitnesses),
                'time': gen_time,
            }
            self.history.append(stats)

            if verbose:
                print(f"Gen {gen:3d}: best={self.best_fitness:,.0f} cycles, "
                      f"gen_best={min_fitness:,.0f}, gen_avg={avg_fitness:,.0f}, "
                      f"valid={len(valid_fitnesses)}/{self.population_size}, "
                      f"time={gen_time:.1f}s")

            # Early stopping if we hit a great solution
            if self.best_fitness < 1500:
                print(f"\nExcellent solution found! {self.best_fitness} cycles")
                break

            self.evolve_generation()

        print(f"\nBest solution: {self.best_fitness:,.0f} cycles")
        if self.best_genome is not None:
            print(f"Genome: {asdict(self.best_genome)}")

        return self.best_genome, self.best_fitness

    def save_results(self, filename: str = "ga_results.json"):
        """Save results to file"""
        results = {
            'best_fitness': self.best_fitness,
            'best_genome': asdict(self.best_genome) if self.best_genome else None,
            'history': self.history,
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
            }
        }

        # Convert enums to strings
        if results['best_genome']:
            for k, v in results['best_genome'].items():
                if isinstance(v, Enum):
                    results['best_genome'][k] = v.value

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filename}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GA optimizer for VLIW kernel')
    parser.add_argument('--generations', '-g', type=int, default=50, help='Number of generations')
    parser.add_argument('--population', '-p', type=int, default=30, help='Population size')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Parallel workers')
    parser.add_argument('--mutation', '-m', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--output', '-o', type=str, default='ga_results.json', help='Output file')

    args = parser.parse_args()

    optimizer = GeneticOptimizer(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation,
        workers=args.workers,
    )

    best_genome, best_fitness = optimizer.run()
    optimizer.save_results(args.output)

    # Print speedup
    BASELINE = 147734
    print(f"\nSpeedup over baseline: {BASELINE / best_fitness:.2f}x")


if __name__ == "__main__":
    main()
