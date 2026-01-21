"""
Exploratory Genetic Algorithm for VLIW kernel optimization.

This version removes assumptions about "best practices" and lets the GA
explore fundamentally different approaches, not just parameter tuning.

Key differences from ga_optimizer.py:
1. Multiple alternative implementations for each phase
2. GA can choose different strategies per component
3. No hardcoded "always vectorize" or "always preload"
4. More granular control over instruction ordering
5. Radical mutation operators that can completely change approach
"""

from dataclasses import dataclass, field, fields, asdict
from typing import List, Tuple, Optional, Any
from enum import Enum
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import random
import json
import time
import argparse

from problem import (
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
# Exploratory Genome - Multiple Strategies Per Phase
# =============================================================================

class GatherStrategy(Enum):
    """Different ways to implement the gather (non-contiguous load) operation"""
    SIMPLE = "simple"              # One address calc + load at a time
    DUAL_ADDR = "dual_addr"        # Compute 2 addresses, then 2 loads
    BATCH_ADDR = "batch_addr"      # Compute ALL addresses, then ALL loads
    INTERLEAVED = "interleaved"    # Interleave addr/load across chunks
    PIPELINED = "pipelined"        # Start next addr while loading


class HashStrategy(Enum):
    """Different ways to implement the hash computation"""
    STANDARD = "standard"          # Original: flush after each stage pair
    FUSED = "fused"                # Emit all ops, single flush at end
    STAGE_PAIRS = "stage_pairs"    # Fuse pairs of stages
    UNROLLED = "unrolled"          # Unroll 2 hash stages together
    MINIMAL_SYNC = "minimal_sync"  # Only sync when absolutely needed


class IndexStrategy(Enum):
    """Different ways to compute the next index"""
    VSELECT = "vselect"            # Use vselect flow instruction
    ARITHMETIC = "arithmetic"       # Use subtraction: offset = 2 - cond
    MULTIPLY_ADD = "multiply_add"   # Use multiply_add instruction
    BITWISE = "bitwise"            # Try bitwise ops for conditionals
    BRANCHLESS = "branchless"      # Different branchless formulation


class LoopStructure(Enum):
    """How to structure the main loop"""
    FULLY_UNROLLED = "fully_unrolled"    # No loops, all unrolled
    ROUND_OUTER = "round_outer"          # for round: for batch (unrolled)
    BATCH_OUTER = "batch_outer"          # for batch: for round (unrolled)
    HARDWARE_LOOPS = "hardware_loops"    # Use jump instructions
    HYBRID = "hybrid"                    # Partial unroll with loops


class MemoryLayout(Enum):
    """How to organize scratch memory"""
    COMPACT = "compact"            # Pack tightly
    ALIGNED = "aligned"            # Align vectors to VLEN
    SEPARATED = "separated"        # Separate scalars and vectors
    CHUNKED = "chunked"            # Group by chunk for locality


@dataclass
class ExploratoryGenome:
    """
    Genome that allows exploration of fundamentally different approaches.
    Each gene represents a CHOICE, not just a parameter value.
    """

    # === Core Strategy Choices ===
    use_vectorization: bool = True        # NOT assumed - can try scalar
    parallel_chunks: int = 1              # 1-32, any value allowed

    # === Per-Phase Strategy Selection ===
    gather_strategy: GatherStrategy = GatherStrategy.SIMPLE
    hash_strategy: HashStrategy = HashStrategy.STANDARD
    index_strategy: IndexStrategy = IndexStrategy.VSELECT
    loop_structure: LoopStructure = LoopStructure.FULLY_UNROLLED
    memory_layout: MemoryLayout = MemoryLayout.COMPACT

    # === Fine-grained Flush Control ===
    flush_after_phase1_const: bool = True
    flush_after_phase1_addr: bool = True
    flush_after_phase1_load: bool = True
    flush_after_gather_addr: bool = True
    flush_after_gather_load: bool = True
    flush_per_gather_element: bool = False  # vs batch all elements
    flush_after_xor: bool = True
    flush_per_hash_stage: bool = True
    flush_after_hash: bool = True
    flush_per_index_op: bool = True
    flush_after_index: bool = True
    flush_after_store_addr: bool = True

    # === Constant Handling ===
    preload_scalar_constants: bool = True
    preload_vector_constants: bool = True
    preload_n_nodes_vector: bool = False
    inline_small_constants: bool = False  # Use const load inline vs cached

    # === Instruction Choices ===
    use_multiply_add: bool = False
    use_add_imm_flow: bool = False        # Use flow engine for some adds

    # === Unrolling Control ===
    batch_unroll_factor: int = 1          # 1, 2, 4, 8
    round_unroll_factor: int = 1          # 1, 2, 4, 8, 16
    gather_unroll_factor: int = 1         # 1, 2, 4 - unroll gather loop
    hash_unroll_factor: int = 1           # 1, 2, 3, 6 - unroll hash stages

    # === Scheduling Hints ===
    max_bundle_fill: float = 1.0
    prefer_load_slots: bool = False       # Prioritize filling load slots
    prefer_valu_slots: bool = False       # Prioritize filling VALU slots

    # === Experimental ===
    transpose_computation: bool = False   # Try transposed data layout
    chunk_rounds_together: bool = False   # Process multiple rounds per chunk
    speculative_loads: bool = False       # Load next iteration early

    def __post_init__(self):
        """Validate and clamp values"""
        # parallel_chunks must work with batch_size=256
        # Valid: any divisor of 256/VLEN = 32
        valid_chunks = [1, 2, 4, 8, 16, 32]
        if self.parallel_chunks not in valid_chunks:
            self.parallel_chunks = min(valid_chunks, key=lambda x: abs(x - self.parallel_chunks))

        self.batch_unroll_factor = max(1, min(8, self.batch_unroll_factor))
        self.round_unroll_factor = max(1, min(16, self.round_unroll_factor))
        self.gather_unroll_factor = max(1, min(4, self.gather_unroll_factor))
        self.hash_unroll_factor = max(1, min(6, self.hash_unroll_factor))
        self.max_bundle_fill = max(0.1, min(1.0, self.max_bundle_fill))


def random_genome() -> ExploratoryGenome:
    """Generate a truly random genome - no assumptions about what's good"""
    return ExploratoryGenome(
        # Core choices - always use vectorization (scalar too slow for this problem)
        use_vectorization=True,
        parallel_chunks=random.choice([1, 2, 4, 8, 16]),  # Max 16 due to scratch limits

        # Strategy choices - all options equally likely
        gather_strategy=random.choice(list(GatherStrategy)),
        hash_strategy=random.choice(list(HashStrategy)),
        index_strategy=random.choice(list(IndexStrategy)),
        loop_structure=random.choice(list(LoopStructure)),
        memory_layout=random.choice(list(MemoryLayout)),

        # Flush control - random
        flush_after_phase1_const=random.choice([True, False]),
        flush_after_phase1_addr=random.choice([True, False]),
        flush_after_phase1_load=random.choice([True, False]),
        flush_after_gather_addr=random.choice([True, False]),
        flush_after_gather_load=random.choice([True, False]),
        flush_per_gather_element=random.choice([True, False]),
        flush_after_xor=random.choice([True, False]),
        flush_per_hash_stage=random.choice([True, False]),
        flush_after_hash=random.choice([True, False]),
        flush_per_index_op=random.choice([True, False]),
        flush_after_index=random.choice([True, False]),
        flush_after_store_addr=random.choice([True, False]),

        # Constants - random
        preload_scalar_constants=random.choice([True, False]),
        preload_vector_constants=random.choice([True, False]),
        preload_n_nodes_vector=random.choice([True, False]),
        inline_small_constants=random.choice([True, False]),

        # Instructions
        use_multiply_add=random.choice([True, False]),
        use_add_imm_flow=random.choice([True, False]),

        # Unrolling
        batch_unroll_factor=random.choice([1, 2, 4, 8]),
        round_unroll_factor=random.choice([1, 2, 4, 8, 16]),
        gather_unroll_factor=random.choice([1, 2, 4]),
        hash_unroll_factor=random.choice([1, 2, 3, 6]),

        # Scheduling
        max_bundle_fill=random.uniform(0.3, 1.0),
        prefer_load_slots=random.choice([True, False]),
        prefer_valu_slots=random.choice([True, False]),

        # Experimental
        transpose_computation=random.choice([True, False]),
        chunk_rounds_together=random.choice([True, False]),
        speculative_loads=random.choice([True, False]),
    )


def seeded_genome(seed_type: str) -> ExploratoryGenome:
    """Generate genome seeded towards a particular strategy"""

    if seed_type == "minimal_flush":
        return ExploratoryGenome(
            use_vectorization=True,
            parallel_chunks=8,
            gather_strategy=GatherStrategy.BATCH_ADDR,
            hash_strategy=HashStrategy.FUSED,
            index_strategy=IndexStrategy.ARITHMETIC,
            flush_after_phase1_const=False,
            flush_after_phase1_addr=False,
            flush_after_xor=False,
            flush_per_hash_stage=False,
            flush_per_index_op=False,
            preload_vector_constants=True,
            preload_n_nodes_vector=True,
        )

    elif seed_type == "maximal_parallel":
        return ExploratoryGenome(
            use_vectorization=True,
            parallel_chunks=16,  # Max safe with preloaded constants
            gather_strategy=GatherStrategy.INTERLEAVED,
            hash_strategy=HashStrategy.FUSED,
            index_strategy=IndexStrategy.MULTIPLY_ADD,
            preload_vector_constants=True,
            preload_n_nodes_vector=True,
            use_multiply_add=True,
        )

    elif seed_type == "pipelined":
        return ExploratoryGenome(
            use_vectorization=True,
            parallel_chunks=4,
            gather_strategy=GatherStrategy.PIPELINED,
            hash_strategy=HashStrategy.MINIMAL_SYNC,
            speculative_loads=True,
            flush_per_gather_element=False,
        )

    elif seed_type == "scalar":
        return ExploratoryGenome(
            use_vectorization=False,
            parallel_chunks=1,
            loop_structure=LoopStructure.HARDWARE_LOOPS,
        )

    elif seed_type == "aggressive_unroll":
        return ExploratoryGenome(
            use_vectorization=True,
            parallel_chunks=16,
            batch_unroll_factor=4,
            round_unroll_factor=4,
            gather_unroll_factor=4,
            hash_unroll_factor=2,
            hash_strategy=HashStrategy.UNROLLED,
        )

    else:  # "baseline"
        return ExploratoryGenome()


# =============================================================================
# Exploratory Kernel Builder
# =============================================================================

class ExploratoryKernelBuilder:
    """
    Builds kernels based on exploratory genome.
    Implements multiple strategies for each phase.
    """

    def __init__(self, genome: ExploratoryGenome):
        self.genome = genome
        self.instrs: List[dict] = []
        self.scratch: dict = {}
        self.scratch_debug: dict = {}
        self.scratch_ptr = 0
        self.const_map: dict = {}
        self.vec_const_map: dict = {}
        self.pending_slots: List[Tuple[str, tuple]] = []

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name: Optional[str] = None, length: int = 1) -> int:
        if self.genome.memory_layout == MemoryLayout.ALIGNED and length >= VLEN:
            if self.scratch_ptr % VLEN != 0:
                self.scratch_ptr += VLEN - (self.scratch_ptr % VLEN)

        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr}"
        return addr

    def alloc_vector_scratch(self, name: Optional[str] = None) -> int:
        return self.alloc_scratch(name, VLEN)

    def emit(self, engine: str, slot: tuple):
        self.pending_slots.append((engine, slot))

    def get_const(self, val: int) -> int:
        if val not in self.const_map:
            addr = self.alloc_scratch(f"const_{val}")
            self.const_map[val] = addr
            self.instrs.append({"load": [("const", addr, val)]})
        return self.const_map[val]

    def maybe_flush(self, should_flush: bool):
        """Conditional flush based on genome"""
        if should_flush:
            self.flush_schedule()

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
            elif op == "add_imm":
                writes.add(slot[1])
                reads.add(slot[2])

        return reads, writes

    def _has_dependency(self, slot1_info, slot2_info) -> bool:
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
        max_fill = self.genome.max_bundle_fill

        while not all(scheduled):
            ready = [i for i in range(n) if not scheduled[i] and
                     all(scheduled[p] for p in must_precede[i])]

            if not ready:
                for i in range(n):
                    if not scheduled[i]:
                        engine, slot, _, _, _ = slots_info[i]
                        self.instrs.append({engine: [slot]})
                        scheduled[i] = True
                break

            # Optionally prioritize certain engines
            if self.genome.prefer_load_slots:
                ready.sort(key=lambda i: 0 if slots_info[i][0] == "load" else 1)
            elif self.genome.prefer_valu_slots:
                ready.sort(key=lambda i: 0 if slots_info[i][0] == "valu" else 1)

            bundle = {}
            slots_in_bundle = set()
            bundle_writes = set()

            for i in ready:
                engine, slot, reads, writes, _ = slots_info[i]
                limit = int(SLOT_LIMITS.get(engine, 0) * max_fill)
                current = len(bundle.get(engine, []))

                if writes & bundle_writes:
                    continue

                if current < max(1, limit):
                    bundle.setdefault(engine, []).append(slot)
                    slots_in_bundle.add(i)
                    bundle_writes |= writes

            if bundle:
                self.instrs.append(bundle)
                for i in slots_in_bundle:
                    scheduled[i] = True

        self.pending_slots = []

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Build kernel using genome-selected strategies"""
        genome = self.genome

        # Allocate temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load header
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.flush_schedule()
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        # Preload constants based on genome
        if genome.preload_scalar_constants:
            for val in [0, 1, 2]:
                self.get_const(val)
            for _, val1, _, _, val3 in HASH_STAGES:
                self.get_const(val1)
                self.get_const(val3)

        # Vector constant preloading
        if genome.use_vectorization and genome.preload_vector_constants:
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

            if genome.preload_n_nodes_vector:
                v_addr = self.alloc_vector_scratch("vconst_n_nodes")
                self.vec_const_map["n_nodes"] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.scratch["n_nodes"]))

            self.flush_schedule()

        self.instrs.append({"flow": [("pause",)]})

        # Build main loop based on structure choice
        if genome.use_vectorization:
            self._build_vectorized_kernel(batch_size, rounds)
        else:
            self._build_scalar_kernel(batch_size, rounds)

        self.instrs.append({"flow": [("pause",)]})
        return self.instrs

    def _build_vectorized_kernel(self, batch_size: int, rounds: int):
        """Build vectorized kernel with genome-selected strategies"""
        genome = self.genome
        n_chunks = genome.parallel_chunks
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        # Allocate chunk scratch
        chunk_scratch = []
        for c in range(n_chunks):
            p_idx = self.alloc_vector_scratch(f"p{c}_idx")
            p_val = self.alloc_vector_scratch(f"p{c}_val")
            p_node_val = self.alloc_vector_scratch(f"p{c}_node_val")
            p_tmp1 = self.alloc_vector_scratch(f"p{c}_tmp1")
            p_tmp2 = self.alloc_vector_scratch(f"p{c}_tmp2")
            p_addr1 = self.alloc_scratch(f"p{c}_addr1")
            p_addr2 = self.alloc_scratch(f"p{c}_addr2")
            chunk_scratch.append((p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_addr1, p_addr2))

        # Main loop
        for r in range(rounds):
            for cg in range(chunks_per_round):
                batch_indices = [cg * n_chunks * chunk_size + c * chunk_size
                                for c in range(n_chunks)]

                # Phase 1: Load indices and values
                self._emit_phase1_load(chunk_scratch, batch_indices)

                # Phase 2: Gather using selected strategy
                self._emit_gather(chunk_scratch, genome.gather_strategy)

                # Phase 3: XOR
                self._emit_xor(chunk_scratch)

                # Phase 4: Hash using selected strategy
                self._emit_hash(chunk_scratch, genome.hash_strategy)

                # Phase 5: Index computation using selected strategy
                self._emit_index_computation(chunk_scratch, genome.index_strategy)

                # Phase 6: Store
                self._emit_store(chunk_scratch, batch_indices)

    def _build_scalar_kernel(self, batch_size: int, rounds: int):
        """Build scalar (non-vectorized) kernel"""
        genome = self.genome

        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp1 = self.alloc_scratch("stmp1")
        tmp2 = self.alloc_scratch("stmp2")

        zero_const = self.get_const(0)
        one_const = self.get_const(1)
        two_const = self.get_const(2)

        for r in range(rounds):
            for b in range(batch_size):
                # Load idx
                self.emit("load", ("const", tmp_addr, b))
                self.emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], tmp_addr))
                self.flush_schedule()
                self.emit("load", ("load", tmp_idx, tmp_addr))

                # Load val
                self.emit("load", ("const", tmp_addr, b))
                self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr))
                self.flush_schedule()
                self.emit("load", ("load", tmp_val, tmp_addr))
                self.flush_schedule()

                # Gather node_val
                self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                self.flush_schedule()
                self.emit("load", ("load", tmp_node_val, tmp_addr))
                self.flush_schedule()

                # XOR
                self.emit("alu", ("^", tmp_val, tmp_val, tmp_node_val))
                self.flush_schedule()

                # Hash
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    self.emit("alu", (op1, tmp1, tmp_val, self.get_const(val1)))
                    self.emit("alu", (op3, tmp2, tmp_val, self.get_const(val3)))
                    self.flush_schedule()
                    self.emit("alu", (op2, tmp_val, tmp1, tmp2))
                    self.flush_schedule()

                # Index computation
                self.emit("alu", ("%", tmp1, tmp_val, two_const))
                self.emit("alu", ("==", tmp1, tmp1, zero_const))
                self.flush_schedule()
                self.emit("flow", ("select", tmp2, tmp1, one_const, two_const))
                self.emit("alu", ("*", tmp_idx, tmp_idx, two_const))
                self.flush_schedule()
                self.emit("alu", ("+", tmp_idx, tmp_idx, tmp2))
                self.flush_schedule()

                # Bounds check
                self.emit("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
                self.flush_schedule()
                self.emit("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))
                self.flush_schedule()

                # Store
                self.emit("load", ("const", tmp_addr, b))
                self.emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], tmp_addr))
                self.flush_schedule()
                self.emit("store", ("store", tmp_addr, tmp_idx))

                self.emit("load", ("const", tmp_addr, b))
                self.emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr))
                self.flush_schedule()
                self.emit("store", ("store", tmp_addr, tmp_val))
                self.flush_schedule()

    def _emit_phase1_load(self, chunk_scratch, batch_indices):
        """Emit phase 1: load indices and values"""
        genome = self.genome
        n_chunks = len(chunk_scratch)

        # Emit const loads
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]))
            self.emit("load", ("const", p_addr2, batch_indices[c]))
        self.maybe_flush(genome.flush_after_phase1_const)

        # Compute addresses
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1))
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
        self.maybe_flush(genome.flush_after_phase1_addr)

        # Load vectors
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("vload", p_idx, p_addr1))
            self.emit("load", ("vload", p_val, p_addr2))
        self.maybe_flush(genome.flush_after_phase1_load)

    def _emit_gather(self, chunk_scratch, strategy: GatherStrategy):
        """Emit gather phase using selected strategy"""
        genome = self.genome
        n_chunks = len(chunk_scratch)

        if strategy == GatherStrategy.SIMPLE:
            # Simple approach: one element at a time, flush after each addr+load pair
            for vi in range(VLEN):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                self.flush_schedule()  # MUST flush before load to commit address
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    if (c + 1) % 2 == 0:  # Flush every 2 loads (2 load slots)
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()
            self.maybe_flush(genome.flush_after_gather_load)

        elif strategy == GatherStrategy.DUAL_ADDR:
            # Compute 2 addresses per chunk, then 2 loads per chunk
            for vi in range(0, VLEN, 2):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                    self.emit("alu", ("+", p_addr2, self.scratch["forest_values_p"], p_idx + vi + 1))
                self.flush_schedule()  # MUST flush before load to commit addresses
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    self.emit("load", ("load", p_node_val + vi + 1, p_addr2))
                    self.flush_schedule()  # Flush after each chunk's 2 loads (2 load slots)
            self.maybe_flush(genome.flush_after_gather_load)

        elif strategy == GatherStrategy.BATCH_ADDR:
            # Batch by element index: compute all chunk addresses for one vi, then load all
            # This batches addresses across chunks rather than across elements
            for vi in range(VLEN):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                self.flush_schedule()  # Commit addresses before loads
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    if (c + 1) % 2 == 0:
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()
            self.maybe_flush(genome.flush_after_gather_load)

        elif strategy == GatherStrategy.INTERLEAVED:
            # Interleave address computation across chunks
            for vi in range(VLEN):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                self.flush_schedule()
                # Emit loads 2 at a time
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    if (c + 1) % 2 == 0:
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()

        elif strategy == GatherStrategy.PIPELINED:
            # Pipelined: same as INTERLEAVED but with different flush pattern
            for vi in range(VLEN):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                self.flush_schedule()  # MUST flush before load
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    if (c + 1) % 2 == 0:
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()

    def _emit_xor(self, chunk_scratch):
        """Emit XOR phase"""
        for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("^", p_val, p_val, p_node_val))
        self.maybe_flush(self.genome.flush_after_xor)

    def _emit_hash(self, chunk_scratch, strategy: HashStrategy):
        """Emit hash computation using selected strategy"""
        genome = self.genome
        has_preloaded = len(self.vec_const_map) > 3

        if strategy == HashStrategy.STANDARD:
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if has_preloaded and val1 in self.vec_const_map:
                    vc1, vc3 = self.vec_const_map[val1], self.vec_const_map[val3]
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1))
                        self.emit("valu", (op3, p_tmp2, p_val, vc3))
                else:
                    v_const_a = chunk_scratch[0][3]  # Reuse tmp1 of first chunk
                    v_const_b = chunk_scratch[0][4]  # Reuse tmp2 of first chunk
                    self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                    self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, v_const_a))
                        self.emit("valu", (op3, p_tmp2, p_val, v_const_b))
                self.flush_schedule()
                for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
                self.maybe_flush(genome.flush_per_hash_stage)
            self.maybe_flush(genome.flush_after_hash)

        elif strategy == HashStrategy.FUSED:
            # Emit all hash ops, single flush at end
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if has_preloaded and val1 in self.vec_const_map:
                    vc1, vc3 = self.vec_const_map[val1], self.vec_const_map[val3]
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1))
                        self.emit("valu", (op3, p_tmp2, p_val, vc3))
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
                else:
                    # Must flush if not preloaded to avoid clobbering
                    v_const_a, v_const_b = chunk_scratch[0][3], chunk_scratch[0][4]
                    self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                    self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, v_const_a))
                        self.emit("valu", (op3, p_tmp2, p_val, v_const_b))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
            self.flush_schedule()

        elif strategy in (HashStrategy.MINIMAL_SYNC, HashStrategy.STAGE_PAIRS, HashStrategy.UNROLLED):
            # Similar to fused but with dependency-aware emission
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if has_preloaded and val1 in self.vec_const_map:
                    vc1, vc3 = self.vec_const_map[val1], self.vec_const_map[val3]
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1))
                        self.emit("valu", (op3, p_tmp2, p_val, vc3))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
                else:
                    v_const_a, v_const_b = chunk_scratch[0][3], chunk_scratch[0][4]
                    self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                    self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, v_const_a))
                        self.emit("valu", (op3, p_tmp2, p_val, v_const_b))
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
            self.flush_schedule()

    def _emit_index_computation(self, chunk_scratch, strategy: IndexStrategy):
        """Emit index computation using selected strategy"""
        genome = self.genome
        has_preloaded = 2 in self.vec_const_map
        has_n_nodes = "n_nodes" in self.vec_const_map

        if has_preloaded:
            vc_zero = self.vec_const_map[0]
            vc_two = self.vec_const_map[2]
        else:
            vc_zero = vc_two = None

        if strategy == IndexStrategy.VSELECT:
            self._emit_index_vselect(chunk_scratch, vc_zero, vc_two, has_n_nodes)
        elif strategy == IndexStrategy.ARITHMETIC:
            self._emit_index_arithmetic(chunk_scratch, vc_zero, vc_two, has_n_nodes)
        elif strategy == IndexStrategy.MULTIPLY_ADD:
            self._emit_index_multiply_add(chunk_scratch, vc_zero, vc_two, has_n_nodes)
        elif strategy == IndexStrategy.BITWISE:
            self._emit_index_bitwise(chunk_scratch, vc_zero, vc_two, has_n_nodes)
        elif strategy == IndexStrategy.BRANCHLESS:
            self._emit_index_branchless(chunk_scratch, vc_zero, vc_two, has_n_nodes)

    def _emit_index_vselect(self, chunk_scratch, vc_zero, vc_two, has_n_nodes):
        """Original vselect-based index computation"""
        genome = self.genome

        if vc_zero is not None:
            vc_one = self.vec_const_map[1]
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two))
                self.emit("valu", ("*", p_idx, p_idx, vc_two))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("flow", ("vselect", p_tmp1, p_tmp2, vc_one, vc_two))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
            self.flush_schedule()

            # Bounds check
            if has_n_nodes:
                vc_n = self.vec_const_map["n_nodes"]
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, vc_n))
            else:
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
                self.flush_schedule()
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("flow", ("vselect", p_idx, p_tmp2, p_idx, vc_zero))
            self.flush_schedule()

    def _emit_index_arithmetic(self, chunk_scratch, vc_zero, vc_two, has_n_nodes):
        """Arithmetic-based index computation (no vselect)"""
        if vc_zero is None:
            return self._emit_index_vselect(chunk_scratch, vc_zero, vc_two, has_n_nodes)

        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("%", p_tmp2, p_val, vc_two))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))  # offset = 2 - cond
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("*", p_idx, p_idx, vc_two))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
        self.flush_schedule()

        # Bounds check
        if has_n_nodes:
            vc_n = self.vec_const_map["n_nodes"]
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, vc_n))
        else:
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("*", p_idx, p_idx, p_tmp2))  # idx * inbounds
        self.flush_schedule()

    def _emit_index_multiply_add(self, chunk_scratch, vc_zero, vc_two, has_n_nodes):
        """Use multiply_add for idx = 2*idx + offset"""
        if vc_zero is None or not self.genome.use_multiply_add:
            return self._emit_index_arithmetic(chunk_scratch, vc_zero, vc_two, has_n_nodes)

        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("%", p_tmp2, p_val, vc_two))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1))
        self.flush_schedule()

        # Bounds check
        if has_n_nodes:
            vc_n = self.vec_const_map["n_nodes"]
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, vc_n))
        else:
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
        self.flush_schedule()
        for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
        self.flush_schedule()

    def _emit_index_bitwise(self, chunk_scratch, vc_zero, vc_two, has_n_nodes):
        """Try bitwise operations for index computation"""
        # Fallback to arithmetic for now
        return self._emit_index_arithmetic(chunk_scratch, vc_zero, vc_two, has_n_nodes)

    def _emit_index_branchless(self, chunk_scratch, vc_zero, vc_two, has_n_nodes):
        """Alternative branchless formulation"""
        # Same as arithmetic for now
        return self._emit_index_arithmetic(chunk_scratch, vc_zero, vc_two, has_n_nodes)

    def _emit_store(self, chunk_scratch, batch_indices):
        """Emit store phase"""
        genome = self.genome

        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]))
            self.emit("load", ("const", p_addr2, batch_indices[c]))
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1))
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
        self.maybe_flush(genome.flush_after_store_addr)

        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("store", ("vstore", p_addr1, p_idx))
            self.emit("store", ("vstore", p_addr2, p_val))
        self.flush_schedule()


# =============================================================================
# Fitness Evaluation
# =============================================================================

TEST_FOREST_HEIGHT = 10
TEST_ROUNDS = 16
TEST_BATCH_SIZE = 256
TEST_SEED = 123


def evaluate_genome(genome: ExploratoryGenome, timeout_cycles: int = 50000) -> float:
    try:
        # Force vectorization - scalar is too slow for this problem size
        if not genome.use_vectorization:
            return float('inf')

        random.seed(TEST_SEED)
        forest = Tree.generate(TEST_FOREST_HEIGHT)
        inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
        mem = build_mem_image(forest, inp)

        builder = ExploratoryKernelBuilder(genome)
        instrs = builder.build_kernel(
            forest.height, len(forest.values), len(inp.indices), TEST_ROUNDS
        )

        machine = Machine(mem, instrs, builder.debug_info(), n_cores=N_CORES)
        machine.enable_pause = True
        machine.enable_debug = False

        ref_gen = reference_kernel2(list(mem), {})
        ref_mem = None
        for ref_mem in ref_gen:
            machine.run(max_cycles=timeout_cycles)  # Use internal cycle limit
            if machine.cycle >= timeout_cycles:
                return float('inf')

        if ref_mem is not None:
            inp_values_p = ref_mem[6]
            if machine.mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE] != \
               ref_mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE]:
                return float('inf')

        return float(machine.cycle)
    except Exception as e:
        return float('inf')


def evaluate_wrapper(args):
    genome, timeout = args
    return evaluate_genome(genome, timeout)


# =============================================================================
# Genetic Operators with Radical Mutations
# =============================================================================

def tournament_select(population, fitnesses, k=3):
    indices = random.sample(range(len(population)), k)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best_idx])


def crossover(p1: ExploratoryGenome, p2: ExploratoryGenome) -> ExploratoryGenome:
    child = ExploratoryGenome()
    for f in fields(ExploratoryGenome):
        if random.random() < 0.5:
            setattr(child, f.name, getattr(p1, f.name))
        else:
            setattr(child, f.name, getattr(p2, f.name))
    return child


def mutate(genome: ExploratoryGenome, rate: float = 0.15) -> ExploratoryGenome:
    genome = deepcopy(genome)

    for f in fields(ExploratoryGenome):
        if random.random() > rate:
            continue

        val = getattr(genome, f.name)

        if isinstance(val, bool):
            setattr(genome, f.name, not val)
        elif isinstance(val, int):
            if f.name == "parallel_chunks":
                setattr(genome, f.name, random.choice([1, 2, 4, 8, 16]))
            else:
                new_val = val + random.randint(-2, 2)
                setattr(genome, f.name, max(1, new_val))
        elif isinstance(val, float):
            new_val = val + random.gauss(0, 0.15)
            setattr(genome, f.name, max(0.1, min(1.0, new_val)))
        elif isinstance(val, Enum):
            options = list(type(val))
            setattr(genome, f.name, random.choice(options))

    genome.__post_init__()
    return genome


def radical_mutate(genome: ExploratoryGenome) -> ExploratoryGenome:
    """Completely change strategy - explore different region of search space"""
    mutation_type = random.choice([
        "random_strategies",
        "flip_all_flushes",
        "change_parallelism",
        "seeded_strategy",
    ])

    if mutation_type == "random_strategies":
        # Randomize all strategy enums
        genome = deepcopy(genome)
        genome.gather_strategy = random.choice(list(GatherStrategy))
        genome.hash_strategy = random.choice(list(HashStrategy))
        genome.index_strategy = random.choice(list(IndexStrategy))
        genome.loop_structure = random.choice(list(LoopStructure))
        genome.memory_layout = random.choice(list(MemoryLayout))

    elif mutation_type == "flip_all_flushes":
        # Flip all flush booleans
        genome = deepcopy(genome)
        for f in fields(ExploratoryGenome):
            if f.name.startswith("flush_"):
                setattr(genome, f.name, not getattr(genome, f.name))

    elif mutation_type == "change_parallelism":
        # Dramatically change parallel_chunks
        genome = deepcopy(genome)
        current = genome.parallel_chunks
        options = [x for x in [1, 2, 4, 8, 16] if x != current]
        genome.parallel_chunks = random.choice(options)

    elif mutation_type == "seeded_strategy":
        # Replace with a seeded genome
        seed_type = random.choice(["minimal_flush", "maximal_parallel",
                                   "pipelined", "aggressive_unroll"])
        genome = seeded_genome(seed_type)

    genome.__post_init__()
    return genome


# =============================================================================
# Main GA
# =============================================================================

class ExploratoryGA:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.2,
        radical_mutation_prob: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: int = 3,
        workers: int = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.radical_mutation_prob = radical_mutation_prob
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.workers = workers or cpu_count()

        self.population = []
        self.fitnesses = []
        self.best_genome = None
        self.best_fitness = float('inf')
        self.history = []

    def initialize_population(self):
        self.population = []

        # Seed with diverse strategies
        for seed_type in ["baseline", "minimal_flush", "maximal_parallel",
                          "pipelined", "aggressive_unroll"]:
            self.population.append(seeded_genome(seed_type))

        # Rest random
        while len(self.population) < self.population_size:
            self.population.append(random_genome())

    def evaluate_population(self):
        timeout = int(self.best_fitness * 1.5) if self.best_fitness < float('inf') else 50000

        # Evaluate sequentially - Pool has too much overhead on macOS
        self.fitnesses = [evaluate_genome(g, timeout) for g in self.population]

        for i, fitness in enumerate(self.fitnesses):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = deepcopy(self.population[i])

    def evolve_generation(self):
        new_population = []

        # Elitism
        sorted_indices = sorted(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        for i in range(self.elitism):
            if self.fitnesses[sorted_indices[i]] < float('inf'):
                new_population.append(deepcopy(self.population[sorted_indices[i]]))

        # Fill with offspring
        while len(new_population) < self.population_size:
            if random.random() < self.radical_mutation_prob:
                # Radical mutation - explore new region
                parent = tournament_select(self.population, self.fitnesses)
                child = radical_mutate(parent)
            elif random.random() < self.crossover_rate:
                p1 = tournament_select(self.population, self.fitnesses)
                p2 = tournament_select(self.population, self.fitnesses)
                child = crossover(p1, p2)
                child = mutate(child, self.mutation_rate)
            else:
                child = tournament_select(self.population, self.fitnesses)
                child = mutate(child, self.mutation_rate)

            new_population.append(child)

        self.population = new_population

    def run(self, verbose=True):
        print(f"Starting Exploratory GA: {self.population_size} individuals, {self.generations} generations", flush=True)
        print(f"Workers: {self.workers}, Radical mutation prob: {self.radical_mutation_prob}", flush=True)
        print(flush=True)

        self.initialize_population()

        for gen in range(self.generations):
            start = time.time()
            self.evaluate_population()

            valid = [f for f in self.fitnesses if f < float('inf')]
            avg = sum(valid) / len(valid) if valid else float('inf')
            gen_best = min(valid) if valid else float('inf')

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'gen_best': gen_best,
                'gen_avg': avg,
                'valid_count': len(valid),
                'time': time.time() - start,
            })

            if verbose:
                print(f"Gen {gen:3d}: best={self.best_fitness:,.0f}, "
                      f"gen_best={gen_best:,.0f}, gen_avg={avg:,.0f}, "
                      f"valid={len(valid)}/{self.population_size}, "
                      f"time={time.time()-start:.1f}s", flush=True)

            if self.best_fitness < 1500:
                print(f"\nExcellent solution found!")
                break

            self.evolve_generation()

        print(f"\nBest: {self.best_fitness:,.0f} cycles")
        if self.best_genome:
            print(f"Strategy: gather={self.best_genome.gather_strategy.value}, "
                  f"hash={self.best_genome.hash_strategy.value}, "
                  f"index={self.best_genome.index_strategy.value}, "
                  f"chunks={self.best_genome.parallel_chunks}")

        return self.best_genome, self.best_fitness

    def save_results(self, filename="exploratory_ga_results.json"):
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        results = {
            'best_fitness': self.best_fitness,
            'best_genome': {k: convert(v) for k, v in asdict(self.best_genome).items()} if self.best_genome else None,
            'history': self.history,
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Exploratory GA for VLIW kernel')
    parser.add_argument('--generations', '-g', type=int, default=30)
    parser.add_argument('--population', '-p', type=int, default=40)
    parser.add_argument('--workers', '-w', type=int, default=None)
    parser.add_argument('--radical', '-r', type=float, default=0.15, help='Radical mutation probability')
    parser.add_argument('--output', '-o', type=str, default='exploratory_ga_results.json')

    args = parser.parse_args()

    ga = ExploratoryGA(
        population_size=args.population,
        generations=args.generations,
        workers=args.workers,
        radical_mutation_prob=args.radical,
    )

    best_genome, best_fitness = ga.run()
    ga.save_results(args.output)

    print(f"\nSpeedup over baseline: {147734 / best_fitness:.2f}x")


if __name__ == "__main__":
    main()
