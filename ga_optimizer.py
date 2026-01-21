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
    pipeline_depth: int = 1            # Number of vector iterations to overlap (1-4)

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
    use_loops: bool = False            # Use actual loop instructions instead of full unrolling

    # Advanced optimizations
    preload_vector_constants: bool = False  # Precompute constant vectors to avoid vbroadcast
    preload_n_nodes: bool = False           # Preload n_nodes as vector constant
    software_pipeline: bool = False         # Overlap loads with computation
    parallel_chunks: int = 1                # Process multiple vector chunks in parallel (1-32)
    use_multiply_add: bool = False          # Use multiply_add for index computation
    minimize_flushes: bool = False          # Reduce flush_schedule calls for better packing

    def __post_init__(self):
        """Clamp values to valid ranges"""
        self.batch_unroll_factor = max(1, min(32, self.batch_unroll_factor))
        self.round_unroll_factor = max(1, min(16, self.round_unroll_factor))
        self.max_bundle_fill = max(0.05, min(1.0, self.max_bundle_fill))  # Allow very low for safe mode
        self.load_ahead_distance = max(0, min(10, self.load_ahead_distance))
        self.pipeline_depth = max(1, min(4, self.pipeline_depth))  # 1-4 iterations overlapped
        # parallel_chunks must divide batch_size evenly: 256/(8*n) must be integer
        # Valid values: 1, 2, 4, 8, 16, 32
        valid_chunks = [1, 2, 4, 8, 16, 32]
        if self.parallel_chunks not in valid_chunks:
            self.parallel_chunks = min(valid_chunks, key=lambda x: abs(x - self.parallel_chunks))


def random_genome() -> Genome:
    """Generate a random valid genome"""
    return Genome(
        batch_unroll_factor=random.choice([1, 2, 4, 8]),
        round_unroll_factor=random.choice([1, 2, 4, 8, 16]),
        loop_order=random.choice(list(LoopOrder)),
        vectorize_batch=True,  # Enable vectorization (8x speedup)
        vectorize_hash=random.choice([True, False]),
        pipeline_depth=random.choice([1, 2]),  # Overlap multiple iterations
        schedule_strategy=random.choice(list(ScheduleStrategy)),
        max_bundle_fill=random.uniform(0.8, 1.0),  # High fill for dependency-aware scheduling
        load_ahead_distance=random.randint(0, 5),
        scratch_allocation=random.choice(list(ScratchAllocation)),
        preload_constants=True,  # Always preload - proven beneficial
        use_add_imm=random.choice([True, False]),
        fuse_hash_stages=random.choice([True, False]),
        use_loops=random.choice([True, False]),  # Try both loop-based and unrolled
        preload_vector_constants=True,  # Always preload - proven beneficial
        preload_n_nodes=random.choice([True, False]),  # Preload n_nodes vector
        software_pipeline=random.choice([True, False]),
        parallel_chunks=random.choice([1, 2, 4, 8, 16, 32]),  # Full range of chunk counts
        use_multiply_add=random.choice([True, False]),  # Try multiply_add optimization
        minimize_flushes=random.choice([True, False]),  # Try reducing flush barriers
    )


def baseline_genome() -> Genome:
    """Genome with our best known optimizations"""
    return Genome(
        batch_unroll_factor=1,
        round_unroll_factor=1,
        loop_order=LoopOrder.ROUND_BATCH,
        vectorize_batch=True,  # Enable vectorization
        vectorize_hash=True,
        schedule_strategy=ScheduleStrategy.GREEDY,
        max_bundle_fill=1.0,  # Use dependency-aware scheduling
        load_ahead_distance=0,
        scratch_allocation=ScratchAllocation.COMPACT,
        preload_constants=True,
        use_add_imm=False,
        fuse_hash_stages=True,
        preload_vector_constants=True,  # Avoid runtime broadcasts
        preload_n_nodes=True,  # Preload n_nodes as vector
        software_pipeline=False,
        parallel_chunks=8,  # Good balance of parallelism
        use_multiply_add=True,  # Use multiply_add for idx computation
        minimize_flushes=True,  # Reduce barriers for better packing
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

    def get_vec_const(self, val: int, temp_vec: int) -> int:
        """
        Get address of a vector constant.
        If preloaded, returns the preloaded address.
        Otherwise broadcasts to temp_vec and returns that.
        """
        if hasattr(self, 'vec_const_map') and val in self.vec_const_map:
            return self.vec_const_map[val]
        else:
            # Must broadcast at runtime
            self.emit("valu", ("vbroadcast", temp_vec, self.get_const(val)))
            return temp_vec

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
        v_idx = v_val = v_node_val = v_tmp1 = v_tmp2 = v_const_a = v_const_b = 0
        vec_base_addr = vec_tmp_addr = 0

        # Dictionary to hold preloaded vector constants
        self.vec_const_map = {}

        if genome.vectorize_batch:
            # Vector registers for batch processing
            v_idx = self.alloc_vector_scratch("v_idx")
            v_val = self.alloc_vector_scratch("v_val")
            v_node_val = self.alloc_vector_scratch("v_node_val")
            v_tmp1 = self.alloc_vector_scratch("v_tmp1")
            v_tmp2 = self.alloc_vector_scratch("v_tmp2")
            v_const_a = self.alloc_vector_scratch("v_const_a")  # Reusable for broadcasts
            v_const_b = self.alloc_vector_scratch("v_const_b")
            vec_base_addr = self.alloc_scratch("vec_base_addr")
            vec_tmp_addr = self.alloc_scratch("vec_tmp_addr")

            # Preload vector constants if enabled (saves vbroadcast in every iteration)
            if genome.preload_vector_constants:
                # Constants used in index computation
                for val in [0, 1, 2]:
                    v_addr = self.alloc_vector_scratch(f"vconst_{val}")
                    self.vec_const_map[val] = v_addr
                    self.emit("valu", ("vbroadcast", v_addr, self.get_const(val)))

                # Hash stage constants
                for _, val1, _, _, val3 in HASH_STAGES:
                    if val1 not in self.vec_const_map:
                        v_addr = self.alloc_vector_scratch(f"vconst_{val1:x}")
                        self.vec_const_map[val1] = v_addr
                        self.emit("valu", ("vbroadcast", v_addr, self.get_const(val1)))
                    if val3 not in self.vec_const_map:
                        v_addr = self.alloc_vector_scratch(f"vconst_{val3}")
                        self.vec_const_map[val3] = v_addr
                        self.emit("valu", ("vbroadcast", v_addr, self.get_const(val3)))

                # Preload n_nodes as vector constant if enabled
                if genome.preload_n_nodes:
                    v_addr = self.alloc_vector_scratch("vconst_n_nodes")
                    self.vec_const_map["n_nodes"] = v_addr
                    self.emit("valu", ("vbroadcast", v_addr, self.scratch["n_nodes"]))

                self.flush_schedule()

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

        # Allocate scratch for parallel chunks if needed
        parallel_vec_scratch = []
        if genome.parallel_chunks > 1 and genome.vectorize_batch:
            for c in range(genome.parallel_chunks):
                p_idx = self.alloc_vector_scratch(f"p{c}_idx")
                p_val = self.alloc_vector_scratch(f"p{c}_val")
                p_node_val = self.alloc_vector_scratch(f"p{c}_node_val")
                p_tmp1 = self.alloc_vector_scratch(f"p{c}_tmp1")
                p_tmp2 = self.alloc_vector_scratch(f"p{c}_tmp2")
                p_tmp_addr = self.alloc_scratch(f"p{c}_tmp_addr")
                p_base_addr = self.alloc_scratch(f"p{c}_base_addr")
                parallel_vec_scratch.append((p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr))

        # Vector scratch tuple for passing to loop functions
        vec_scratch = (v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_const_a, v_const_b,
                       vec_tmp_addr, vec_base_addr)

        # Generate loop body based on loop order and whether to use actual loops
        if genome.parallel_chunks > 1 and genome.vectorize_batch and batch_size >= VLEN * genome.parallel_chunks:
            self._emit_parallel_chunks_loop(
                rounds, batch_size,
                zero_const, one_const, two_const,
                v_const_a, v_const_b,
                parallel_vec_scratch,
                genome.parallel_chunks
            )
        elif genome.use_loops and genome.vectorize_batch:
            self._emit_vectorized_loop_based(
                rounds, batch_size, batch_step,
                zero_const, one_const, two_const,
                vec_scratch, tmp1, tmp2, tmp3
            )
        elif genome.loop_order == LoopOrder.ROUND_BATCH:
            self._emit_round_batch_loop(
                rounds, round_unroll, round_iters,
                batch_size, batch_unroll, batch_iters, batch_step,
                tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                tmp1, tmp2, tmp3,
                zero_const, one_const, two_const,
                genome, vec_scratch
            )
        else:
            self._emit_batch_round_loop(
                rounds, round_unroll, round_iters,
                batch_size, batch_unroll, batch_iters, batch_step,
                tmp_idx, tmp_val, tmp_node_val, tmp_addr,
                tmp1, tmp2, tmp3,
                zero_const, one_const, two_const,
                genome, vec_scratch
            )

        # Final pause
        self.instrs.append({"flow": [("pause",)]})

        return self.instrs

    def _emit_vectorized_loop_based(
        self, rounds, batch_size, batch_step,
        zero_const, one_const, two_const,
        vec_scratch, tmp1, tmp2, tmp3
    ):
        """
        Emit vectorized kernel using actual loop instructions instead of full unrolling.
        This dramatically reduces code size and may improve cache performance.
        """
        v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_const_a, v_const_b, vec_tmp_addr, vec_base_addr = vec_scratch

        # Allocate loop control variables
        round_counter = self.alloc_scratch("round_counter")
        batch_counter = self.alloc_scratch("batch_counter")
        batch_offset = self.alloc_scratch("batch_offset")  # Current batch position (0, 8, 16, ...)
        loop_cond = self.alloc_scratch("loop_cond")

        # Constants for loop
        batch_count = batch_size // batch_step  # Number of vector iterations per round
        vlen_const = self.get_const(batch_step)

        # Initialize round counter
        self.emit("load", ("const", round_counter, rounds))
        self.flush_schedule()

        # Outer loop: rounds
        round_loop_start = len(self.instrs)

        # Initialize batch counter and offset for this round
        self.emit("load", ("const", batch_counter, batch_count))
        self.emit("load", ("const", batch_offset, 0))
        self.flush_schedule()

        # Inner loop: batch chunks
        batch_loop_start = len(self.instrs)

        # ===== LOOP BODY: Process one vector chunk =====
        # Phase 1: Load indices and values
        self.emit("alu", ("+", vec_base_addr, self.scratch["inp_indices_p"], batch_offset))
        self.emit("alu", ("+", vec_tmp_addr, self.scratch["inp_values_p"], batch_offset))
        self.flush_schedule()

        self.emit("load", ("vload", v_idx, vec_base_addr))
        self.emit("load", ("vload", v_val, vec_tmp_addr))
        self.flush_schedule()

        # Phase 2: Gather node_val (dual-load optimization)
        tmp_addr2 = vec_base_addr  # Reuse for second address
        for vi in range(0, VLEN, 2):
            self.emit("alu", ("+", vec_tmp_addr, self.scratch["forest_values_p"], v_idx + vi))
            self.emit("alu", ("+", tmp_addr2, self.scratch["forest_values_p"], v_idx + vi + 1))
            self.flush_schedule()
            self.emit("load", ("load", v_node_val + vi, vec_tmp_addr))
            self.emit("load", ("load", v_node_val + vi + 1, tmp_addr2))
        self.flush_schedule()

        # Phase 3: XOR and hash
        self.emit("valu", ("^", v_val, v_val, v_node_val))

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
            self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
            self.emit("valu", (op1, v_tmp1, v_val, v_const_a))
            self.emit("valu", (op3, v_tmp2, v_val, v_const_b))
            self.flush_schedule()
            self.emit("valu", (op2, v_val, v_tmp1, v_tmp2))
        self.flush_schedule()

        # Phase 4: Compute next indices
        self.emit("valu", ("vbroadcast", v_tmp1, two_const))
        self.emit("valu", ("vbroadcast", v_const_a, zero_const))
        self.flush_schedule()

        self.emit("valu", ("%", v_tmp2, v_val, v_tmp1))
        self.emit("valu", ("*", v_idx, v_idx, v_tmp1))
        self.flush_schedule()

        self.emit("valu", ("==", v_tmp2, v_tmp2, v_const_a))
        self.emit("valu", ("vbroadcast", v_const_a, one_const))
        self.emit("valu", ("vbroadcast", v_const_b, two_const))
        self.flush_schedule()

        self.emit("flow", ("vselect", v_tmp1, v_tmp2, v_const_a, v_const_b))
        self.flush_schedule()

        self.emit("valu", ("+", v_idx, v_idx, v_tmp1))
        self.emit("valu", ("vbroadcast", v_tmp1, self.scratch["n_nodes"]))
        self.flush_schedule()

        # Phase 5: Wrap check
        self.emit("valu", ("<", v_tmp2, v_idx, v_tmp1))
        self.emit("valu", ("vbroadcast", v_tmp1, zero_const))
        self.flush_schedule()

        self.emit("flow", ("vselect", v_idx, v_tmp2, v_idx, v_tmp1))
        self.flush_schedule()

        # Phase 6: Store results
        self.emit("alu", ("+", vec_base_addr, self.scratch["inp_indices_p"], batch_offset))
        self.emit("alu", ("+", vec_tmp_addr, self.scratch["inp_values_p"], batch_offset))
        self.flush_schedule()

        self.emit("store", ("vstore", vec_base_addr, v_idx))
        self.emit("store", ("vstore", vec_tmp_addr, v_val))
        self.flush_schedule()

        # ===== END LOOP BODY =====

        # Update batch offset and counter
        self.emit("alu", ("+", batch_offset, batch_offset, vlen_const))
        self.emit("alu", ("-", batch_counter, batch_counter, one_const))
        self.flush_schedule()

        # Check if more batch chunks remain
        self.emit("alu", ("<", loop_cond, zero_const, batch_counter))
        self.flush_schedule()
        self.instrs.append({"flow": [("cond_jump", loop_cond, batch_loop_start)]})

        # Decrement round counter
        self.emit("alu", ("-", round_counter, round_counter, one_const))
        self.flush_schedule()

        # Check if more rounds remain
        self.emit("alu", ("<", loop_cond, zero_const, round_counter))
        self.flush_schedule()
        self.instrs.append({"flow": [("cond_jump", loop_cond, round_loop_start)]})

    def _emit_round_batch_loop(
        self, rounds, round_unroll, round_iters,
        batch_size, batch_unroll, batch_iters, batch_step,
        tmp_idx, tmp_val, tmp_node_val, tmp_addr,
        tmp1, tmp2, tmp3,
        zero_const, one_const, two_const,
        genome, vec_scratch
    ):
        """Emit loop with rounds as outer loop"""
        v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_const_a, v_const_b, vec_tmp_addr, vec_base_addr = vec_scratch

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
                                r, b,
                                zero_const, one_const, two_const,
                                v_idx, v_val, v_node_val, v_tmp1, v_tmp2,
                                v_const_a, v_const_b, vec_tmp_addr, vec_base_addr
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
        genome, vec_scratch
    ):
        """Emit loop with batch as outer loop"""
        v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_const_a, v_const_b, vec_tmp_addr, vec_base_addr = vec_scratch

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
                                r, b,
                                zero_const, one_const, two_const,
                                v_idx, v_val, v_node_val, v_tmp1, v_tmp2,
                                v_const_a, v_const_b, vec_tmp_addr, vec_base_addr
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
        zero_const, one_const, two_const,
        v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_const_a, v_const_b,
        tmp_addr, base_addr
    ):
        """Emit one vectorized iteration processing VLEN elements.
        Uses pre-allocated scratch to avoid running out of space.
        Minimizes flush calls to maximize VLIW packing."""

        # Use aggressive pipelining mode - minimal flushes
        aggressive = self.genome.software_pipeline

        # Phase 1: Load indices and values
        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("load", ("const", tmp_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_indices_p"], base_addr))
        self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr))
        self.flush_schedule()  # Must compute addresses before loads

        self.emit("load", ("vload", v_idx, base_addr))
        self.emit("load", ("vload", v_val, tmp_addr))
        self.flush_schedule()  # Must load idx before computing gather addresses

        # Phase 2: Gather node_val - use both load slots
        tmp_addr2 = base_addr
        for vi in range(0, VLEN, 2):
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], v_idx + vi))
            self.emit("alu", ("+", tmp_addr2, self.scratch["forest_values_p"], v_idx + vi + 1))
            if not aggressive:
                self.flush_schedule()
            self.emit("load", ("load", v_node_val + vi, tmp_addr))
            self.emit("load", ("load", v_node_val + vi + 1, tmp_addr2))
        self.flush_schedule()  # Must complete gather before XOR

        # Phase 3: XOR and hash (mostly independent VALU ops)
        self.emit("valu", ("^", v_val, v_val, v_node_val))

        # Vectorized hash - emit all operations and let scheduler handle dependencies
        has_preloaded = hasattr(self, 'vec_const_map') and len(self.vec_const_map) > 3
        aggressive = self.genome.software_pipeline

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if has_preloaded and val1 in self.vec_const_map and val3 in self.vec_const_map:
                # Use preloaded constants
                vc1 = self.vec_const_map[val1]
                vc3 = self.vec_const_map[val3]
                self.emit("valu", (op1, v_tmp1, v_val, vc1))
                self.emit("valu", (op3, v_tmp2, v_val, vc3))
            else:
                # Broadcast at runtime
                self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                self.emit("valu", (op1, v_tmp1, v_val, v_const_a))
                self.emit("valu", (op3, v_tmp2, v_val, v_const_b))
            # Only flush between stages if not aggressive, scheduler handles deps
            if not aggressive:
                self.flush_schedule()
            self.emit("valu", (op2, v_val, v_tmp1, v_tmp2))
        self.flush_schedule()

        # Phase 4: Compute next indices
        # Check if we have preloaded constants
        has_preloaded = hasattr(self, 'vec_const_map') and 2 in self.vec_const_map

        if has_preloaded:
            # Use preloaded constants directly
            vc_zero = self.vec_const_map[0]
            vc_two = self.vec_const_map[2]

            self.emit("valu", ("%", v_tmp2, v_val, vc_two))  # val % 2
            self.emit("valu", ("*", v_idx, v_idx, vc_two))   # 2*idx (can run in parallel!)
            self.flush_schedule()

            # tmp2 = (val % 2 == 0), so tmp2 is 0 or 1
            self.emit("valu", ("==", v_tmp2, v_tmp2, vc_zero))  # == 0
            self.flush_schedule()

            # Replace vselect with arithmetic: tmp1 = 2 - tmp2 (gives 1 if tmp2=1, 2 if tmp2=0)
            self.emit("valu", ("-", v_tmp1, vc_two, v_tmp2))
            self.flush_schedule()

            self.emit("valu", ("+", v_idx, v_idx, v_tmp1))  # 2*idx + offset
            self.emit("valu", ("vbroadcast", v_tmp1, self.scratch["n_nodes"]))
            self.flush_schedule()

            # Phase 5: Wrap check
            # tmp2 = (idx < n_nodes), 0 or 1
            self.emit("valu", ("<", v_tmp2, v_idx, v_tmp1))
            self.flush_schedule()

            # Replace vselect with arithmetic: idx = idx * tmp2 (gives idx if tmp2=1, 0 if tmp2=0)
            self.emit("valu", ("*", v_idx, v_idx, v_tmp2))
            self.flush_schedule()
        else:
            # Use explicit broadcasts (original approach)
            self.emit("valu", ("vbroadcast", v_tmp1, two_const))
            self.emit("valu", ("vbroadcast", v_const_a, zero_const))
            self.flush_schedule()

            self.emit("valu", ("%", v_tmp2, v_val, v_tmp1))  # val % 2
            self.emit("valu", ("*", v_idx, v_idx, v_tmp1))   # 2*idx
            self.flush_schedule()

            self.emit("valu", ("==", v_tmp2, v_tmp2, v_const_a))  # == 0
            self.emit("valu", ("vbroadcast", v_const_a, one_const))
            self.emit("valu", ("vbroadcast", v_const_b, two_const))
            self.flush_schedule()

            self.emit("flow", ("vselect", v_tmp1, v_tmp2, v_const_a, v_const_b))  # 1 or 2
            self.flush_schedule()

            self.emit("valu", ("+", v_idx, v_idx, v_tmp1))  # 2*idx + offset
            self.emit("valu", ("vbroadcast", v_tmp1, self.scratch["n_nodes"]))
            self.flush_schedule()

            # Phase 5: Wrap check
            self.emit("valu", ("<", v_tmp2, v_idx, v_tmp1))
            self.emit("valu", ("vbroadcast", v_tmp1, zero_const))
            self.flush_schedule()

            self.emit("flow", ("vselect", v_idx, v_tmp2, v_idx, v_tmp1))
            self.flush_schedule()

        # Phase 6: Store results (use both store slots in parallel)
        self.emit("load", ("const", base_addr, batch_idx))
        self.emit("load", ("const", tmp_addr, batch_idx))
        self.emit("alu", ("+", base_addr, self.scratch["inp_indices_p"], base_addr))
        self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr))
        self.flush_schedule()

        # Both vstores can happen in same cycle since we have 2 store slots
        self.emit("store", ("vstore", base_addr, v_idx))
        self.emit("store", ("vstore", tmp_addr, v_val))
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

    def _emit_vector_hash(self, v_val: int, v_tmp1: int, v_tmp2: int, v_const_a: int, v_const_b: int):
        """Emit vectorized hash computation using pre-allocated vector scratch"""
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # Broadcast constants to reusable vector scratch
            self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
            self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))

            self.emit("valu", (op1, v_tmp1, v_val, v_const_a))
            self.emit("valu", (op3, v_tmp2, v_val, v_const_b))
            self.flush_schedule()
            self.emit("valu", (op2, v_val, v_tmp1, v_tmp2))
            self.flush_schedule()

    def _emit_parallel_chunks_loop(
        self, rounds, batch_size,
        zero_const, one_const, two_const,
        v_const_a, v_const_b,
        parallel_vec_scratch,
        n_chunks
    ):
        """
        Process multiple vector chunks in parallel to maximize VLIW utilization.
        Interleaves operations from different chunks so the scheduler can pack them.
        Uses genome parameters for optimizations.
        """
        genome = self.genome
        has_preloaded = hasattr(self, 'vec_const_map') and len(self.vec_const_map) > 3
        has_n_nodes_preloaded = hasattr(self, 'vec_const_map') and "n_nodes" in self.vec_const_map
        use_multiply_add = genome.use_multiply_add
        minimize_flushes = genome.minimize_flushes
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        for _ in range(rounds):
            for chunk_group in range(chunks_per_round):
                batch_indices = [chunk_group * n_chunks * chunk_size + c * chunk_size for c in range(n_chunks)]

                # Phase 1: Load indices and values for ALL chunks
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("load", ("const", p_base_addr, batch_indices[c]))
                    self.emit("load", ("const", p_tmp_addr, batch_indices[c]))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("alu", ("+", p_base_addr, self.scratch["inp_indices_p"], p_base_addr))
                    self.emit("alu", ("+", p_tmp_addr, self.scratch["inp_values_p"], p_tmp_addr))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("load", ("vload", p_idx, p_base_addr))
                    self.emit("load", ("vload", p_val, p_tmp_addr))
                self.flush_schedule()

                # Phase 2: Gather node_val (bottleneck - 2 loads/cycle)
                for vi in range(VLEN):
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("alu", ("+", p_tmp_addr, self.scratch["forest_values_p"], p_idx + vi))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("load", ("load", p_node_val + vi, p_tmp_addr))
                        if (c + 1) % 2 == 0:
                            self.flush_schedule()
                    if n_chunks % 2 != 0:
                        self.flush_schedule()

                # Phase 3: XOR
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("valu", ("^", p_val, p_val, p_node_val))
                if not minimize_flushes:
                    self.flush_schedule()

                # Phase 4: Hash
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    if has_preloaded and val1 in self.vec_const_map and val3 in self.vec_const_map:
                        vc1 = self.vec_const_map[val1]
                        vc3 = self.vec_const_map[val3]
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", (op1, p_tmp1, p_val, vc1))
                            self.emit("valu", (op3, p_tmp2, p_val, vc3))
                    else:
                        self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                        self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", (op1, p_tmp1, p_val, v_const_a))
                            self.emit("valu", (op3, p_tmp2, p_val, v_const_b))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
                self.flush_schedule()

                # Phase 5: Index computation
                if has_preloaded and 2 in self.vec_const_map:
                    vc_zero = self.vec_const_map[0]
                    vc_two = self.vec_const_map[2]
                    vc_n_nodes = self.vec_const_map.get("n_nodes")

                    # Compute val % 2
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("%", p_tmp2, p_val, vc_two))
                    self.flush_schedule()

                    # tmp2 = (val % 2 == 0)
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
                    self.flush_schedule()

                    # tmp1 = 2 - tmp2 (offset)
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))
                    self.flush_schedule()

                    # idx = 2 * idx + offset (use multiply_add if enabled)
                    if use_multiply_add:
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1))
                    else:
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("*", p_idx, p_idx, vc_two))
                        self.flush_schedule()
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
                    self.flush_schedule()

                    # Bounds check: idx = idx < n_nodes ? idx : 0
                    if vc_n_nodes is not None and has_n_nodes_preloaded:
                        # Use preloaded n_nodes vector
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("<", p_tmp2, p_idx, vc_n_nodes))
                    else:
                        # Broadcast n_nodes at runtime
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
                        self.flush_schedule()
                        for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                            self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
                    self.flush_schedule()

                    # idx = idx * tmp2 (gives idx if in bounds, 0 otherwise)
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
                    self.flush_schedule()
                else:
                    # Non-preloaded fallback
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("vbroadcast", p_tmp1, two_const))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("%", p_tmp2, p_val, p_tmp1))
                        self.emit("valu", ("*", p_idx, p_idx, p_tmp1))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("vbroadcast", p_tmp1, zero_const))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("==", p_tmp2, p_tmp2, p_tmp1))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("-", p_tmp1, p_tmp1, p_tmp2))  # Fix: use arithmetic
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
                    self.flush_schedule()
                    # Bounds check
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
                    self.flush_schedule()
                    for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                        self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
                    self.flush_schedule()

                # Phase 6: Store results
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("load", ("const", p_base_addr, batch_indices[c]))
                    self.emit("load", ("const", p_tmp_addr, batch_indices[c]))
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("alu", ("+", p_base_addr, self.scratch["inp_indices_p"], p_base_addr))
                    self.emit("alu", ("+", p_tmp_addr, self.scratch["inp_values_p"], p_tmp_addr))
                self.flush_schedule()
                for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_tmp_addr, p_base_addr) in enumerate(parallel_vec_scratch):
                    self.emit("store", ("vstore", p_base_addr, p_idx))
                    self.emit("store", ("vstore", p_tmp_addr, p_val))
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
