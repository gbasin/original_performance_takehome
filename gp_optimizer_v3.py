"""
Genetic Programming optimizer V3 for VLIW kernel synthesis.

Extensions over V1/V2:
1. Software Pipelining - overlap loop iterations
2. Instruction Interleaving - mix operations from different phases
3. Memory Access Patterns - prefetching, store buffering
4. Fine-Grained Unrolling - per-phase unroll factors
5. Register Pressure Control - allocation strategies
6. Phase Ordering and Fusion - reorder or merge phases
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
from copy import deepcopy
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
    HASH_STAGES,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
)


# =============================================================================
# Type System for GP
# =============================================================================

class GPType(Enum):
    PROGRAM = auto()
    SETUP = auto()
    LOOP = auto()
    PHASE_SEQ = auto()
    GATHER = auto()
    HASH = auto()
    INDEX = auto()
    STORE = auto()
    PIPELINE = auto()
    MEMORY = auto()
    REGISTER = auto()
    INTERLEAVE = auto()


# =============================================================================
# AST Node Base
# =============================================================================

class GPNode(ABC):
    @property
    @abstractmethod
    def node_type(self) -> GPType:
        pass

    @abstractmethod
    def children(self) -> List['GPNode']:
        pass

    @abstractmethod
    def replace_child(self, index: int, new_child: 'GPNode') -> None:
        pass

    @abstractmethod
    def clone(self) -> 'GPNode':
        pass

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children())

    def depth(self) -> int:
        children = self.children()
        if not children:
            return 1
        return 1 + max(c.depth() for c in children)

    def all_nodes(self) -> List[Tuple['GPNode', Optional['GPNode'], int]]:
        result: List[Tuple['GPNode', Optional['GPNode'], int]] = [(self, None, -1)]
        for i, child in enumerate(self.children()):
            result.append((child, self, i))
            for node, parent, idx in child.all_nodes():
                if parent is None:
                    continue
                result.append((node, parent, idx))
        return result


# =============================================================================
# Extension 1: Software Pipelining
# =============================================================================

class PipelineSchedule(Enum):
    NONE = "none"                    # No pipelining
    GATHER_FIRST = "gather_first"    # Start next gather while hashing
    BALANCED = "balanced"            # Even overlap of all phases
    STORE_LAST = "store_last"        # Delay stores to overlap with next iter


@dataclass
class PipelineNode(GPNode):
    """Controls software pipelining of loop iterations"""
    enabled: bool = False
    pipeline_depth: int = 2          # How many iterations in flight (1-4)
    schedule: PipelineSchedule = PipelineSchedule.NONE
    prologue_unroll: int = 1         # Iterations before steady state
    epilogue_drain: bool = True      # Properly drain pipeline at end

    @property
    def node_type(self) -> GPType:
        return GPType.PIPELINE

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'PipelineNode':
        return PipelineNode(
            enabled=self.enabled,
            pipeline_depth=self.pipeline_depth,
            schedule=self.schedule,
            prologue_unroll=self.prologue_unroll,
            epilogue_drain=self.epilogue_drain
        )


# =============================================================================
# Extension 2: Instruction Interleaving
# =============================================================================

class InterleaveStrategy(Enum):
    NONE = "none"                    # No interleaving, strict phase boundaries
    GATHER_HASH = "gather_hash"      # Overlap gather and hash
    HASH_INDEX = "hash_index"        # Overlap hash and index
    ALL_PHASES = "all_phases"        # Maximum interleaving
    ADAPTIVE = "adaptive"            # Based on slot utilization


@dataclass
class InterleaveNode(GPNode):
    """Controls instruction interleaving between phases"""
    strategy: InterleaveStrategy = InterleaveStrategy.NONE
    lookahead_depth: int = 2         # How many ops ahead to look for interleaving
    min_slot_fill: float = 0.5       # Min fill ratio before forcing flush
    allow_cross_chunk: bool = False  # Allow interleaving across chunks

    @property
    def node_type(self) -> GPType:
        return GPType.INTERLEAVE

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'InterleaveNode':
        return InterleaveNode(
            strategy=self.strategy,
            lookahead_depth=self.lookahead_depth,
            min_slot_fill=self.min_slot_fill,
            allow_cross_chunk=self.allow_cross_chunk
        )


# =============================================================================
# Extension 3: Memory Access Patterns
# =============================================================================

class AddressGenStrategy(Enum):
    LAZY = "lazy"                    # Compute address just before load
    EAGER = "eager"                  # Compute all addresses upfront
    SPECULATIVE = "speculative"      # Compute ahead of need


class AccessOrder(Enum):
    SEQUENTIAL = "sequential"        # 0,1,2,3,4,5,6,7
    STRIDED = "strided"              # 0,2,4,6,1,3,5,7
    BLOCKED = "blocked"              # 0,1,2,3 then 4,5,6,7
    REVERSED = "reversed"            # 7,6,5,4,3,2,1,0


@dataclass
class MemoryNode(GPNode):
    """Controls memory access patterns"""
    prefetch_distance: int = 0       # Loads issued N iterations early
    store_buffer_depth: int = 0      # Delay stores to batch them
    address_gen: AddressGenStrategy = AddressGenStrategy.LAZY
    access_order: AccessOrder = AccessOrder.SEQUENTIAL
    coalesce_loads: bool = False     # Try to merge adjacent loads
    coalesce_stores: bool = False    # Try to merge adjacent stores

    @property
    def node_type(self) -> GPType:
        return GPType.MEMORY

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'MemoryNode':
        return MemoryNode(
            prefetch_distance=self.prefetch_distance,
            store_buffer_depth=self.store_buffer_depth,
            address_gen=self.address_gen,
            access_order=self.access_order,
            coalesce_loads=self.coalesce_loads,
            coalesce_stores=self.coalesce_stores
        )


# =============================================================================
# Extension 5: Register Pressure Control
# =============================================================================

class AllocationStrategy(Enum):
    DENSE = "dense"                  # Pack registers tightly
    SPARSE = "sparse"                # Spread out to reduce conflicts
    PHASED = "phased"                # Different regions per phase


class ReusePolicy(Enum):
    AGGRESSIVE = "aggressive"        # Reuse immediately when dead
    CONSERVATIVE = "conservative"    # Keep some slack
    LIFETIME = "lifetime"            # Based on liveness analysis


@dataclass
class RegisterNode(GPNode):
    """Controls register allocation strategy"""
    allocation: AllocationStrategy = AllocationStrategy.DENSE
    reuse_policy: ReusePolicy = ReusePolicy.AGGRESSIVE
    spill_threshold: int = 512       # When to write back intermediates
    vector_alignment: int = 8        # Align vector regs to this boundary
    reserve_temps: int = 4           # Number of temp registers to reserve

    @property
    def node_type(self) -> GPType:
        return GPType.REGISTER

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'RegisterNode':
        return RegisterNode(
            allocation=self.allocation,
            reuse_policy=self.reuse_policy,
            spill_threshold=self.spill_threshold,
            vector_alignment=self.vector_alignment,
            reserve_temps=self.reserve_temps
        )


# =============================================================================
# Extension 6: Phase Ordering and Fusion
# =============================================================================

class PhaseOrder(Enum):
    STANDARD = ["gather", "hash", "index", "store"]
    HASH_FIRST = ["hash", "gather", "index", "store"]  # Precompute hash
    INDEX_EARLY = ["gather", "index", "hash", "store"]
    STORE_EARLY = ["gather", "hash", "store", "index"]  # Speculative store


class FusionMode(Enum):
    NONE = "none"                    # No fusion
    GATHER_XOR = "gather_xor"        # Fuse gather with XOR
    HASH_INDEX = "hash_index"        # Fuse hash with index computation
    INDEX_STORE = "index_store"      # Fuse index with store
    FULL = "full"                    # Maximum fusion


# =============================================================================
# Extended Phase Nodes with Fine-Grained Unrolling (Extension 4)
# =============================================================================

class GatherStrategy(Enum):
    SEQUENTIAL = "sequential"
    DUAL_ADDR = "dual_addr"
    BATCH_ADDR = "batch_addr"
    PIPELINED = "pipelined"
    VECTORIZED = "vectorized"        # NEW: Use vector address generation


@dataclass
class GatherNode(GPNode):
    """Gather phase with fine-grained unrolling"""
    strategy: GatherStrategy = GatherStrategy.BATCH_ADDR
    flush_after_addr: bool = False
    flush_per_element: bool = False
    # Extension 4: Fine-grained unrolling
    inner_unroll: int = 1            # Unroll the per-element loop (1,2,4)
    vector_grouping: int = 1         # Process N vectors together (1,2,4)
    addr_compute_ahead: int = 0      # Compute addresses N elements ahead

    @property
    def node_type(self) -> GPType:
        return GPType.GATHER

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'GatherNode':
        return GatherNode(
            strategy=self.strategy,
            flush_after_addr=self.flush_after_addr,
            flush_per_element=self.flush_per_element,
            inner_unroll=self.inner_unroll,
            vector_grouping=self.vector_grouping,
            addr_compute_ahead=self.addr_compute_ahead
        )


class HashStrategy(Enum):
    STANDARD = "standard"
    FUSED = "fused"
    INTERLEAVED = "interleaved"
    UNROLLED = "unrolled"            # NEW: Unroll hash stages


@dataclass
class HashNode(GPNode):
    """Hash phase with fine-grained unrolling"""
    strategy: HashStrategy = HashStrategy.FUSED
    flush_per_stage: bool = False
    use_preloaded_consts: bool = True
    # Extension 4: Fine-grained unrolling
    stage_unroll: Tuple[int, ...] = (1, 1, 1, 1)  # Per-stage unroll factors
    fuse_xor_with_stage1: bool = False  # Combine XOR with first hash stage
    cross_chunk_interleave: bool = False  # Interleave stages across chunks

    @property
    def node_type(self) -> GPType:
        return GPType.HASH

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'HashNode':
        return HashNode(
            strategy=self.strategy,
            flush_per_stage=self.flush_per_stage,
            use_preloaded_consts=self.use_preloaded_consts,
            stage_unroll=self.stage_unroll,
            fuse_xor_with_stage1=self.fuse_xor_with_stage1,
            cross_chunk_interleave=self.cross_chunk_interleave
        )


class IndexStrategy(Enum):
    VSELECT = "vselect"
    ARITHMETIC = "arithmetic"
    MULTIPLY_ADD = "multiply_add"
    BRANCHLESS = "branchless"        # NEW: Alternative branchless


@dataclass
class IndexNode(GPNode):
    """Index computation with fine-grained unrolling"""
    strategy: IndexStrategy = IndexStrategy.MULTIPLY_ADD
    flush_per_op: bool = False
    use_preloaded_consts: bool = True
    # Extension 4: Fine-grained unrolling
    compute_unroll: int = 1          # Unroll index computation
    bounds_check_mode: str = "multiply"  # "multiply", "select", "mask"
    speculative: bool = False        # Compute both branches speculatively

    @property
    def node_type(self) -> GPType:
        return GPType.INDEX

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'IndexNode':
        return IndexNode(
            strategy=self.strategy,
            flush_per_op=self.flush_per_op,
            use_preloaded_consts=self.use_preloaded_consts,
            compute_unroll=self.compute_unroll,
            bounds_check_mode=self.bounds_check_mode,
            speculative=self.speculative
        )


@dataclass
class StoreNode(GPNode):
    """Store phase with fine-grained control"""
    flush_after_addr: bool = False
    # Extension 4: Fine-grained unrolling
    batch_stores: int = 1            # Batch N stores together
    store_order: str = "idx_first"   # "idx_first", "val_first", "interleaved"
    write_combining: bool = False    # Attempt write combining

    @property
    def node_type(self) -> GPType:
        return GPType.STORE

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'StoreNode':
        return StoreNode(
            flush_after_addr=self.flush_after_addr,
            batch_stores=self.batch_stores,
            store_order=self.store_order,
            write_combining=self.write_combining
        )


# =============================================================================
# Main Program Structure
# =============================================================================

@dataclass
class SetupNode(GPNode):
    preload_scalars: bool = True
    preload_vectors: bool = True
    preload_hash_consts: bool = True
    preload_n_nodes: bool = True

    @property
    def node_type(self) -> GPType:
        return GPType.SETUP

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'SetupNode':
        return SetupNode(
            preload_scalars=self.preload_scalars,
            preload_vectors=self.preload_vectors,
            preload_hash_consts=self.preload_hash_consts,
            preload_n_nodes=self.preload_n_nodes
        )


class LoopStructure(Enum):
    ROUND_BATCH = "round_batch"
    BATCH_ROUND = "batch_round"
    FUSED = "fused"
    CHUNKED = "chunked"


@dataclass
class PhaseSequenceNode(GPNode):
    """Phase sequence with ordering and fusion control"""
    gather: GatherNode
    hash_comp: HashNode
    index_comp: IndexNode
    store: StoreNode
    # Extension 6: Phase ordering and fusion
    phase_order: List[str] = field(default_factory=lambda: ["gather", "hash", "index", "store"])
    fusion_mode: FusionMode = FusionMode.NONE
    fused_phases: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def node_type(self) -> GPType:
        return GPType.PHASE_SEQ

    def children(self) -> List[GPNode]:
        return [self.gather, self.hash_comp, self.index_comp, self.store]

    def replace_child(self, index: int, new_child: GPNode):
        if index == 0 and isinstance(new_child, GatherNode):
            self.gather = new_child
        elif index == 1 and isinstance(new_child, HashNode):
            self.hash_comp = new_child
        elif index == 2 and isinstance(new_child, IndexNode):
            self.index_comp = new_child
        elif index == 3 and isinstance(new_child, StoreNode):
            self.store = new_child

    def clone(self) -> 'PhaseSequenceNode':
        return PhaseSequenceNode(
            gather=self.gather.clone(),
            hash_comp=self.hash_comp.clone(),
            index_comp=self.index_comp.clone(),
            store=self.store.clone(),
            phase_order=list(self.phase_order),
            fusion_mode=self.fusion_mode,
            fused_phases=list(self.fused_phases)
        )


@dataclass
class LoopNode(GPNode):
    """Loop with all extensions"""
    structure: LoopStructure
    parallel_chunks: int
    body: PhaseSequenceNode
    # Extension 1: Software pipelining
    pipeline: PipelineNode = field(default_factory=PipelineNode)
    # Extension 2: Instruction interleaving
    interleave: InterleaveNode = field(default_factory=InterleaveNode)
    # Extension 3: Memory access patterns
    memory: MemoryNode = field(default_factory=MemoryNode)
    # Extension 5: Register control
    registers: RegisterNode = field(default_factory=RegisterNode)
    # Additional loop controls
    outer_unroll: int = 1            # Unroll outer loop
    chunk_unroll: int = 1            # Unroll chunk processing

    @property
    def node_type(self) -> GPType:
        return GPType.LOOP

    def children(self) -> List[GPNode]:
        return [self.body, self.pipeline, self.interleave, self.memory, self.registers]

    def replace_child(self, index: int, new_child: GPNode):
        if index == 0 and isinstance(new_child, PhaseSequenceNode):
            self.body = new_child
        elif index == 1 and isinstance(new_child, PipelineNode):
            self.pipeline = new_child
        elif index == 2 and isinstance(new_child, InterleaveNode):
            self.interleave = new_child
        elif index == 3 and isinstance(new_child, MemoryNode):
            self.memory = new_child
        elif index == 4 and isinstance(new_child, RegisterNode):
            self.registers = new_child

    def clone(self) -> 'LoopNode':
        return LoopNode(
            structure=self.structure,
            parallel_chunks=self.parallel_chunks,
            body=self.body.clone(),
            pipeline=self.pipeline.clone(),
            interleave=self.interleave.clone(),
            memory=self.memory.clone(),
            registers=self.registers.clone(),
            outer_unroll=self.outer_unroll,
            chunk_unroll=self.chunk_unroll
        )


@dataclass
class ProgramNode(GPNode):
    setup: SetupNode
    main_loop: LoopNode

    @property
    def node_type(self) -> GPType:
        return GPType.PROGRAM

    def children(self) -> List[GPNode]:
        return [self.setup, self.main_loop]

    def replace_child(self, index: int, new_child: GPNode):
        if index == 0 and isinstance(new_child, SetupNode):
            self.setup = new_child
        elif index == 1 and isinstance(new_child, LoopNode):
            self.main_loop = new_child

    def clone(self) -> 'ProgramNode':
        return ProgramNode(
            setup=self.setup.clone(),
            main_loop=self.main_loop.clone()
        )


# =============================================================================
# Random Tree Generation
# =============================================================================

def random_pipeline() -> PipelineNode:
    return PipelineNode(
        enabled=random.choice([True, False]),
        pipeline_depth=random.choice([1, 2, 3]),
        schedule=random.choice(list(PipelineSchedule)),
        prologue_unroll=random.choice([1, 2]),
        epilogue_drain=random.choice([True, False])
    )


def random_interleave() -> InterleaveNode:
    return InterleaveNode(
        strategy=random.choice(list(InterleaveStrategy)),
        lookahead_depth=random.choice([1, 2, 3, 4]),
        min_slot_fill=random.choice([0.3, 0.5, 0.7]),
        allow_cross_chunk=random.choice([True, False])
    )


def random_memory() -> MemoryNode:
    return MemoryNode(
        prefetch_distance=random.choice([0, 1, 2]),
        store_buffer_depth=random.choice([0, 1, 2]),
        address_gen=random.choice(list(AddressGenStrategy)),
        access_order=random.choice(list(AccessOrder)),
        coalesce_loads=random.choice([True, False]),
        coalesce_stores=random.choice([True, False])
    )


def random_register() -> RegisterNode:
    return RegisterNode(
        allocation=random.choice(list(AllocationStrategy)),
        reuse_policy=random.choice(list(ReusePolicy)),
        spill_threshold=random.choice([256, 512, 768]),
        vector_alignment=random.choice([8, 16]),
        reserve_temps=random.choice([2, 4, 8])
    )


def random_setup() -> SetupNode:
    return SetupNode(
        preload_scalars=random.choice([True, False]),
        preload_vectors=random.choice([True, False]),
        preload_hash_consts=random.choice([True, False]),
        preload_n_nodes=random.choice([True, False])
    )


def random_gather() -> GatherNode:
    return GatherNode(
        strategy=random.choice(list(GatherStrategy)),
        flush_after_addr=random.choice([True, False]),
        flush_per_element=random.choice([True, False]),
        inner_unroll=random.choice([1, 2, 4]),
        vector_grouping=random.choice([1, 2]),
        addr_compute_ahead=random.choice([0, 1, 2])
    )


def random_hash() -> HashNode:
    return HashNode(
        strategy=random.choice(list(HashStrategy)),
        flush_per_stage=random.choice([True, False]),
        use_preloaded_consts=random.choice([True, False]),
        stage_unroll=tuple(random.choice([1, 2]) for _ in range(4)),
        fuse_xor_with_stage1=random.choice([True, False]),
        cross_chunk_interleave=random.choice([True, False])
    )


def random_index() -> IndexNode:
    return IndexNode(
        strategy=random.choice(list(IndexStrategy)),
        flush_per_op=random.choice([True, False]),
        use_preloaded_consts=random.choice([True, False]),
        compute_unroll=random.choice([1, 2]),
        bounds_check_mode=random.choice(["multiply", "select", "mask"]),
        speculative=random.choice([True, False])
    )


def random_store() -> StoreNode:
    return StoreNode(
        flush_after_addr=random.choice([True, False]),
        batch_stores=random.choice([1, 2]),
        store_order=random.choice(["idx_first", "val_first", "interleaved"]),
        write_combining=random.choice([True, False])
    )


def random_phase_sequence() -> PhaseSequenceNode:
    phase_orders = [
        ["gather", "hash", "index", "store"],
        ["gather", "index", "hash", "store"],
    ]
    fusion_modes = list(FusionMode)

    return PhaseSequenceNode(
        gather=random_gather(),
        hash_comp=random_hash(),
        index_comp=random_index(),
        store=random_store(),
        phase_order=random.choice(phase_orders),
        fusion_mode=random.choice(fusion_modes)
    )


def random_loop() -> LoopNode:
    return LoopNode(
        structure=random.choice(list(LoopStructure)),
        parallel_chunks=random.choice([4, 8, 16]),
        body=random_phase_sequence(),
        pipeline=random_pipeline(),
        interleave=random_interleave(),
        memory=random_memory(),
        registers=random_register(),
        outer_unroll=random.choice([1, 2]),
        chunk_unroll=random.choice([1, 2])
    )


def random_program() -> ProgramNode:
    return ProgramNode(
        setup=random_setup(),
        main_loop=random_loop()
    )


def seeded_program(seed_type: str) -> ProgramNode:
    """Generate a program seeded toward a particular strategy"""

    if seed_type == "minimal_flush":
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.BATCH_ADDR, False, False),
                    hash_comp=HashNode(HashStrategy.FUSED, False, True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, True),
                    store=StoreNode(False)
                ),
                pipeline=PipelineNode(enabled=False),
                interleave=InterleaveNode(strategy=InterleaveStrategy.NONE),
                memory=MemoryNode(),
                registers=RegisterNode()
            )
        )

    elif seed_type == "max_interleave":
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=16,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.PIPELINED, False, False,
                                     inner_unroll=2, addr_compute_ahead=2),
                    hash_comp=HashNode(HashStrategy.INTERLEAVED, False, True,
                                      cross_chunk_interleave=True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, True),
                    store=StoreNode(False),
                    fusion_mode=FusionMode.GATHER_XOR
                ),
                pipeline=PipelineNode(enabled=False),
                interleave=InterleaveNode(
                    strategy=InterleaveStrategy.ALL_PHASES,
                    lookahead_depth=4,
                    min_slot_fill=0.3
                ),
                memory=MemoryNode(address_gen=AddressGenStrategy.EAGER),
                registers=RegisterNode()
            )
        )

    elif seed_type == "pipelined":
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.BATCH_ADDR, False, False),
                    hash_comp=HashNode(HashStrategy.FUSED, False, True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, True),
                    store=StoreNode(False)
                ),
                pipeline=PipelineNode(
                    enabled=True,
                    pipeline_depth=2,
                    schedule=PipelineSchedule.GATHER_FIRST
                ),
                interleave=InterleaveNode(strategy=InterleaveStrategy.GATHER_HASH),
                memory=MemoryNode(prefetch_distance=1),
                registers=RegisterNode()
            )
        )

    elif seed_type == "memory_optimized":
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.BATCH_ADDR, False, False,
                                     addr_compute_ahead=2),
                    hash_comp=HashNode(HashStrategy.FUSED, False, True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, True),
                    store=StoreNode(False, batch_stores=2)
                ),
                pipeline=PipelineNode(enabled=False),
                interleave=InterleaveNode(strategy=InterleaveStrategy.NONE),
                memory=MemoryNode(
                    prefetch_distance=2,
                    store_buffer_depth=1,
                    address_gen=AddressGenStrategy.EAGER,
                    coalesce_stores=True
                ),
                registers=RegisterNode()
            )
        )

    else:  # baseline
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                body=random_phase_sequence(),
                pipeline=PipelineNode(),
                interleave=InterleaveNode(),
                memory=MemoryNode(),
                registers=RegisterNode()
            )
        )


# =============================================================================
# GP Operators
# =============================================================================

def get_nodes_of_type(tree: GPNode, target_type: GPType) -> List[Tuple[GPNode, GPNode, int]]:
    result = []
    for node, parent, idx in tree.all_nodes():
        if node.node_type == target_type:
            result.append((node, parent, idx))
    return result


def subtree_crossover(parent1: ProgramNode, parent2: ProgramNode) -> ProgramNode:
    child = parent1.clone()

    exchangeable_types = [
        GPType.SETUP, GPType.LOOP, GPType.PHASE_SEQ,
        GPType.GATHER, GPType.HASH, GPType.INDEX, GPType.STORE,
        GPType.PIPELINE, GPType.INTERLEAVE, GPType.MEMORY, GPType.REGISTER
    ]
    target_type = random.choice(exchangeable_types)

    child_nodes = get_nodes_of_type(child, target_type)
    parent2_nodes = get_nodes_of_type(parent2, target_type)

    if child_nodes and parent2_nodes:
        _, child_parent, child_idx = random.choice(child_nodes)
        donor_node, _, _ = random.choice(parent2_nodes)

        if child_parent is not None:
            child_parent.replace_child(child_idx, donor_node.clone())

    return child


def point_mutation(tree: ProgramNode, rate: float = 0.2) -> ProgramNode:
    tree = tree.clone()

    for node, parent, idx in tree.all_nodes():
        if random.random() > rate:
            continue

        if isinstance(node, SetupNode):
            field = random.choice(['preload_scalars', 'preload_vectors',
                                   'preload_hash_consts', 'preload_n_nodes'])
            setattr(node, field, not getattr(node, field))

        elif isinstance(node, LoopNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.structure = random.choice(list(LoopStructure))
            elif choice == 1:
                node.parallel_chunks = random.choice([4, 8, 16])
            elif choice == 2:
                node.outer_unroll = random.choice([1, 2])
            else:
                node.chunk_unroll = random.choice([1, 2])

        elif isinstance(node, PipelineNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.enabled = not node.enabled
            elif choice == 1:
                node.pipeline_depth = random.choice([1, 2, 3])
            elif choice == 2:
                node.schedule = random.choice(list(PipelineSchedule))
            else:
                node.prologue_unroll = random.choice([1, 2])

        elif isinstance(node, InterleaveNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.strategy = random.choice(list(InterleaveStrategy))
            elif choice == 1:
                node.lookahead_depth = random.choice([1, 2, 3, 4])
            elif choice == 2:
                node.min_slot_fill = random.choice([0.3, 0.5, 0.7])
            else:
                node.allow_cross_chunk = not node.allow_cross_chunk

        elif isinstance(node, MemoryNode):
            choice = random.randint(0, 5)
            if choice == 0:
                node.prefetch_distance = random.choice([0, 1, 2])
            elif choice == 1:
                node.store_buffer_depth = random.choice([0, 1, 2])
            elif choice == 2:
                node.address_gen = random.choice(list(AddressGenStrategy))
            elif choice == 3:
                node.access_order = random.choice(list(AccessOrder))
            elif choice == 4:
                node.coalesce_loads = not node.coalesce_loads
            else:
                node.coalesce_stores = not node.coalesce_stores

        elif isinstance(node, RegisterNode):
            choice = random.randint(0, 4)
            if choice == 0:
                node.allocation = random.choice(list(AllocationStrategy))
            elif choice == 1:
                node.reuse_policy = random.choice(list(ReusePolicy))
            elif choice == 2:
                node.spill_threshold = random.choice([256, 512, 768])
            elif choice == 3:
                node.vector_alignment = random.choice([8, 16])
            else:
                node.reserve_temps = random.choice([2, 4, 8])

        elif isinstance(node, GatherNode):
            choice = random.randint(0, 5)
            if choice == 0:
                node.strategy = random.choice(list(GatherStrategy))
            elif choice == 1:
                node.flush_after_addr = not node.flush_after_addr
            elif choice == 2:
                node.flush_per_element = not node.flush_per_element
            elif choice == 3:
                node.inner_unroll = random.choice([1, 2, 4])
            elif choice == 4:
                node.vector_grouping = random.choice([1, 2])
            else:
                node.addr_compute_ahead = random.choice([0, 1, 2])

        elif isinstance(node, HashNode):
            choice = random.randint(0, 5)
            if choice == 0:
                node.strategy = random.choice(list(HashStrategy))
            elif choice == 1:
                node.flush_per_stage = not node.flush_per_stage
            elif choice == 2:
                node.use_preloaded_consts = not node.use_preloaded_consts
            elif choice == 3:
                node.stage_unroll = tuple(random.choice([1, 2]) for _ in range(4))
            elif choice == 4:
                node.fuse_xor_with_stage1 = not node.fuse_xor_with_stage1
            else:
                node.cross_chunk_interleave = not node.cross_chunk_interleave

        elif isinstance(node, IndexNode):
            choice = random.randint(0, 5)
            if choice == 0:
                node.strategy = random.choice(list(IndexStrategy))
            elif choice == 1:
                node.flush_per_op = not node.flush_per_op
            elif choice == 2:
                node.use_preloaded_consts = not node.use_preloaded_consts
            elif choice == 3:
                node.compute_unroll = random.choice([1, 2])
            elif choice == 4:
                node.bounds_check_mode = random.choice(["multiply", "select", "mask"])
            else:
                node.speculative = not node.speculative

        elif isinstance(node, StoreNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.flush_after_addr = not node.flush_after_addr
            elif choice == 1:
                node.batch_stores = random.choice([1, 2])
            elif choice == 2:
                node.store_order = random.choice(["idx_first", "val_first", "interleaved"])
            else:
                node.write_combining = not node.write_combining

        elif isinstance(node, PhaseSequenceNode):
            choice = random.randint(0, 1)
            if choice == 0:
                node.fusion_mode = random.choice(list(FusionMode))
            else:
                phase_orders = [
                    ["gather", "hash", "index", "store"],
                    ["gather", "index", "hash", "store"],
                ]
                node.phase_order = random.choice(phase_orders)

    return tree


def subtree_mutation(tree: ProgramNode) -> ProgramNode:
    tree = tree.clone()

    replaceable = [
        (GPType.GATHER, random_gather),
        (GPType.HASH, random_hash),
        (GPType.INDEX, random_index),
        (GPType.STORE, random_store),
        (GPType.PHASE_SEQ, random_phase_sequence),
        (GPType.PIPELINE, random_pipeline),
        (GPType.INTERLEAVE, random_interleave),
        (GPType.MEMORY, random_memory),
        (GPType.REGISTER, random_register),
    ]

    target_type, generator = random.choice(replaceable)
    nodes = get_nodes_of_type(tree, target_type)

    if nodes:
        _, parent, idx = random.choice(nodes)
        if parent is not None:
            parent.replace_child(idx, generator())

    return tree


# =============================================================================
# Code Generator with All Extensions
# =============================================================================

class GPKernelBuilderV3:
    """Compiles a GP V3 tree into VLIW instructions"""

    def __init__(self, program: ProgramNode):
        self.program = program
        self.instrs: List[dict] = []
        self.scratch: dict = {}
        self.scratch_debug: dict = {}
        self.scratch_ptr = 0
        self.const_map: dict = {}
        self.vec_const_map: dict = {}
        self.pending_slots: List[Tuple[str, tuple]] = []

        # Track interleaving state
        self.interleave_buffer: List[Tuple[str, tuple, str]] = []  # (engine, slot, phase)
        self.current_phase: str = ""

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

    def emit(self, engine: str, slot: tuple, phase: str = ""):
        # Always use pending_slots directly - interleave_buffer is for explicit interleaving
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
        """Dependency-aware VLIW scheduling with O(n) dependency detection.

        Uses last_writer map instead of O(n²) pairwise comparison.
        This enables larger scheduling windows for better ILP.
        """
        if not self.pending_slots:
            return

        slots_info = []
        for engine, slot in self.pending_slots:
            reads, writes = self._get_slot_reads_writes(engine, slot)
            slots_info.append([engine, slot, reads, writes, False])

        n = len(slots_info)
        must_precede = [set() for _ in range(n)]

        # O(n) dependency detection using last_writer map
        last_writer = {}  # addr -> instruction index

        for i, (engine, slot, reads, writes, _) in enumerate(slots_info):
            # RAW dependency: we read something that was written earlier
            for addr in reads:
                if addr in last_writer:
                    must_precede[i].add(last_writer[addr])

            # WAW dependency: we write something that was written earlier
            for addr in writes:
                if addr in last_writer:
                    must_precede[i].add(last_writer[addr])
                # Update last writer
                last_writer[addr] = i

        scheduled = [False] * n

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

    def flush_interleaved(self, interleave: InterleaveNode):
        """Flush with interleaving support"""
        if interleave.strategy == InterleaveStrategy.NONE:
            # No interleaving, just flush normally
            for engine, slot, phase in self.interleave_buffer:
                self.pending_slots.append((engine, slot))
            self.interleave_buffer = []
            self.flush_schedule()
            return

        # Group by phase
        phase_ops = {}
        for engine, slot, phase in self.interleave_buffer:
            phase_ops.setdefault(phase, []).append((engine, slot))

        # Interleave phases that can be combined
        if interleave.strategy == InterleaveStrategy.ALL_PHASES:
            # Mix all phases together
            all_ops = [(e, s) for e, s, p in self.interleave_buffer]
            for e, s in all_ops:
                self.pending_slots.append((e, s))
            self.interleave_buffer = []
            self.flush_schedule()

        elif interleave.strategy == InterleaveStrategy.GATHER_HASH:
            # Interleave gather and hash
            gather_ops = phase_ops.get("gather", [])
            hash_ops = phase_ops.get("hash", [])
            other_ops = []
            for p, ops in phase_ops.items():
                if p not in ("gather", "hash"):
                    other_ops.extend(ops)

            # Interleave gather and hash
            max_len = max(len(gather_ops), len(hash_ops))
            for i in range(max_len):
                if i < len(gather_ops):
                    self.pending_slots.append(gather_ops[i])
                if i < len(hash_ops):
                    self.pending_slots.append(hash_ops[i])

            # Flush interleaved
            self.flush_schedule()

            # Then other phases
            for e, s in other_ops:
                self.pending_slots.append((e, s))
            self.interleave_buffer = []
            self.flush_schedule()

        else:
            # Default: sequential by phase
            for phase in ["gather", "hash", "index", "store"]:
                for e, s in phase_ops.get(phase, []):
                    self.pending_slots.append((e, s))
            self.interleave_buffer = []
            self.flush_schedule()

    def maybe_flush(self, should: bool):
        if should:
            self.flush_schedule()

    def build(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
              strip_pauses: bool = False) -> List[dict]:
        """Compile the GP tree into VLIW instructions.

        Args:
            strip_pauses: If True, don't emit pause instructions. Use for final
                         submission where pauses waste cycles. Keep False during
                         GP evaluation where pauses sync with reference_kernel2.
        """

        # Allocate temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load header
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.flush_schedule()
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        # Setup phase
        self._compile_setup(self.program.setup)

        if not strip_pauses:
            self.instrs.append({"flow": [("pause",)]})

        # Main loop
        self._compile_loop(self.program.main_loop, batch_size, rounds)

        if not strip_pauses:
            self.instrs.append({"flow": [("pause",)]})

        return self.instrs

    def _compile_setup(self, setup: SetupNode):
        if setup.preload_scalars:
            for val in [0, 1, 2]:
                self.get_const(val)
            for _, val1, _, _, val3 in HASH_STAGES:
                self.get_const(val1)
                self.get_const(val3)

        if setup.preload_vectors:
            for val in [0, 1, 2]:
                v_addr = self.alloc_vector(f"vconst_{val}")
                self.vec_const_map[val] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.get_const(val)))

            if setup.preload_hash_consts:
                for _, val1, _, _, val3 in HASH_STAGES:
                    if val1 not in self.vec_const_map:
                        v_addr = self.alloc_vector(f"vconst_{val1:x}")
                        self.vec_const_map[val1] = v_addr
                        self.emit("valu", ("vbroadcast", v_addr, self.get_const(val1)))
                    if val3 not in self.vec_const_map:
                        v_addr = self.alloc_vector(f"vconst_{val3}")
                        self.vec_const_map[val3] = v_addr
                        self.emit("valu", ("vbroadcast", v_addr, self.get_const(val3)))

            if setup.preload_n_nodes:
                v_addr = self.alloc_vector("vconst_n_nodes")
                self.vec_const_map["n_nodes"] = v_addr
                self.emit("valu", ("vbroadcast", v_addr, self.scratch["n_nodes"]))

            self.flush_schedule()

    def _compile_loop(self, loop: LoopNode, batch_size: int, rounds: int):
        n_chunks = loop.parallel_chunks
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        # Allocate chunk scratch based on register strategy
        chunk_scratch = []
        for c in range(n_chunks):
            p_idx = self.alloc_vector(f"p{c}_idx")
            p_val = self.alloc_vector(f"p{c}_val")
            p_node_val = self.alloc_vector(f"p{c}_node_val")
            p_tmp1 = self.alloc_vector(f"p{c}_tmp1")
            p_tmp2 = self.alloc_vector(f"p{c}_tmp2")
            p_addr1 = self.alloc_scratch(f"p{c}_addr1")
            p_addr2 = self.alloc_scratch(f"p{c}_addr2")
            chunk_scratch.append((p_idx, p_val, p_node_val, p_tmp1, p_tmp2, p_addr1, p_addr2))

        # Main iteration with optional outer unrolling
        outer_unroll = loop.outer_unroll

        for r in range(rounds):
            for cg in range(0, chunks_per_round, outer_unroll):
                for u in range(min(outer_unroll, chunks_per_round - cg)):
                    batch_indices = [(cg + u) * n_chunks * chunk_size + c * chunk_size
                                    for c in range(n_chunks)]

                    self._compile_phases(loop.body, chunk_scratch, batch_indices, loop)

    def _compile_phases(self, seq: PhaseSequenceNode, chunk_scratch, batch_indices, loop: LoopNode):
        n_chunks = len(chunk_scratch)
        interleave = loop.interleave
        memory = loop.memory
        coalesce_loads = memory.coalesce_loads if memory else False
        store_buffer_depth = memory.store_buffer_depth if memory else 0

        # Phase 1: Load indices and values (no phase tag - always immediate)
        # With coalesce_loads, we try to group loads more aggressively
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]))
            self.emit("load", ("const", p_addr2, batch_indices[c]))
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1))
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
        self.flush_schedule()

        if coalesce_loads:
            # Coalesced loads: emit all loads together before flushing
            # This allows the scheduler to group adjacent memory accesses
            for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                self.emit("load", ("vload", p_idx, p_addr1))
            for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                self.emit("load", ("vload", p_val, p_addr2))
            self.flush_schedule()
        else:
            # Original: interleaved idx/val loads per chunk
            for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                self.emit("load", ("vload", p_idx, p_addr1))
                self.emit("load", ("vload", p_val, p_addr2))
            self.flush_schedule()

        # Track pending stores for store_buffer_depth
        self._store_buffer_depth = store_buffer_depth
        self._pending_store_ops = []

        # Execute phases based on order
        for phase_name in seq.phase_order:
            if phase_name == "gather":
                self._compile_gather(seq.gather, chunk_scratch, interleave, loop.memory)
                # XOR after gather - skip only if fused with hash stage 1
                # If fusion_mode is GATHER_XOR but fuse_xor_with_stage1 is False,
                # we still need to do the XOR here
                xor_will_be_in_hash = (seq.fusion_mode == FusionMode.GATHER_XOR and
                                       seq.hash_comp.fuse_xor_with_stage1)
                if not xor_will_be_in_hash:
                    for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", ("^", p_val, p_val, p_node_val))
                    self.flush_schedule()

            elif phase_name == "hash":
                self._compile_hash(seq.hash_comp, chunk_scratch, interleave, seq)

            elif phase_name == "index":
                self._compile_index(seq.index_comp, chunk_scratch, interleave)

            elif phase_name == "store":
                self._compile_store(seq.store, chunk_scratch, batch_indices, interleave, loop.memory)

        # Flush any remaining interleaved instructions
        if self.interleave_buffer:
            self.flush_interleaved(interleave)

    def _compile_gather(self, gather: GatherNode, chunk_scratch, interleave: InterleaveNode,
                        memory: Optional[MemoryNode] = None):
        n_chunks = len(chunk_scratch)

        # Determine element order based on memory access_order setting
        if memory is not None and memory.access_order == AccessOrder.REVERSED:
            element_order = list(range(VLEN - 1, -1, -1))
        elif memory is not None and memory.access_order == AccessOrder.STRIDED:
            # Process even indices first, then odd: 0,2,4,6,1,3,5,7
            element_order = list(range(0, VLEN, 2)) + list(range(1, VLEN, 2))
        elif memory is not None and memory.access_order == AccessOrder.BLOCKED:
            # Process in blocks: first half then second half
            element_order = list(range(VLEN))  # Same as sequential for gather
        else:
            element_order = list(range(VLEN))

        if gather.strategy == GatherStrategy.SEQUENTIAL:
            for vi in element_order:
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi), "gather")
                self.flush_schedule()
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1), "gather")
                    if (c + 1) % 2 == 0:
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()

        elif gather.strategy == GatherStrategy.DUAL_ADDR:
            for vi in range(0, VLEN, 2):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi), "gather")
                    self.emit("alu", ("+", p_addr2, self.scratch["forest_values_p"], p_idx + vi + 1), "gather")
                self.flush_schedule()
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1), "gather")
                    self.emit("load", ("load", p_node_val + vi + 1, p_addr2), "gather")
                    self.flush_schedule()

        elif gather.strategy in (GatherStrategy.BATCH_ADDR, GatherStrategy.PIPELINED, GatherStrategy.VECTORIZED):
            # With inner_unroll: process 2 elements at a time using both address registers
            # Note: inner_unroll > 2 is effectively the same as 2 due to register constraints
            inner_unroll = gather.inner_unroll

            if inner_unroll >= 2:
                # Process 2 elements at a time using both address registers (p_addr1, p_addr2)
                for vi_base in range(0, len(element_order), 2):
                    vi_pair = element_order[vi_base:vi_base + 2]

                    # Compute addresses for pair
                    for ui, vi in enumerate(vi_pair):
                        for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                            addr_reg = p_addr1 if ui == 0 else p_addr2
                            self.emit("alu", ("+", addr_reg, self.scratch["forest_values_p"], p_idx + vi), "gather")

                    self.maybe_flush(gather.flush_after_addr)

                    # Issue loads for pair - must flush before next pair's address computation
                    for ui, vi in enumerate(vi_pair):
                        for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                            addr_reg = p_addr1 if ui == 0 else p_addr2
                            self.emit("load", ("load", p_node_val + vi, addr_reg), "gather")
                    self.flush_schedule()
            else:
                # No unrolling: original per-element processing
                for vi in element_order:
                    # Compute addresses
                    for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                        self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi), "gather")

                    self.maybe_flush(gather.flush_after_addr)

                    # Issue loads
                    for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                        self.emit("load", ("load", p_node_val + vi, p_addr1), "gather")
                        if (c + 1) % 2 == 0:
                            self.flush_schedule()
                    if n_chunks % 2 != 0:
                        self.flush_schedule()

    def _compile_hash(self, hash_node: HashNode, chunk_scratch, interleave: InterleaveNode, seq: PhaseSequenceNode):
        has_preloaded = hash_node.use_preloaded_consts and len(self.vec_const_map) > 3
        fuse_xor = seq.fusion_mode == FusionMode.GATHER_XOR and hash_node.fuse_xor_with_stage1
        stage_unroll = hash_node.stage_unroll
        cross_chunk = hash_node.cross_chunk_interleave

        # Allocate dedicated scratch for hash constants when not preloaded
        # This avoids the bug where chunk 0's temps were used as constants AND overwritten
        if not has_preloaded:
            if not hasattr(self, '_hash_const_a'):
                self._hash_const_a = self.alloc_vector("hash_const_a")
                self._hash_const_b = self.alloc_vector("hash_const_b")
            v_const_a = self._hash_const_a
            v_const_b = self._hash_const_b
        else:
            v_const_a = v_const_b = None  # Will use vec_const_map instead

        if cross_chunk:
            # Cross-chunk interleaving: process operations from different chunks together
            # This improves ILP by having more independent operations in flight
            for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                unroll_factor = stage_unroll[stage_idx] if stage_idx < len(stage_unroll) else 1

                # XOR fusion with first stage
                if fuse_xor and stage_idx == 0:
                    for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", ("^", p_val, p_val, p_node_val), "hash")
                    self.flush_schedule()

                if has_preloaded and val1 in self.vec_const_map:
                    vc1 = self.vec_const_map[val1]
                    vc3 = self.vec_const_map[val3]
                    # Interleave: emit op1 for all chunks, then op3 for all chunks
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1), "hash")
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op3, p_tmp2, p_val, vc3), "hash")
                else:
                    # Use dedicated scratch for constants (not chunk 0's temps)
                    self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)), "hash")
                    self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)), "hash")
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, v_const_a), "hash")
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op3, p_tmp2, p_val, v_const_b), "hash")

                self.flush_schedule()

                # Final XOR for stage
                for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", (op2, p_val, p_tmp1, p_tmp2), "hash")

                # Only flush if unroll factor is 1 or at unroll boundary
                if unroll_factor == 1 or (stage_idx + 1) % unroll_factor == 0:
                    self.flush_schedule()
                self.maybe_flush(hash_node.flush_per_stage)

        else:
            # Original: process stages sequentially with optional stage unrolling
            stages_to_process = []
            for stage_idx, stage_data in enumerate(HASH_STAGES):
                unroll_factor = stage_unroll[stage_idx] if stage_idx < len(stage_unroll) else 1
                stages_to_process.append((stage_idx, stage_data, unroll_factor))

            for stage_idx, (op1, val1, op2, op3, val3), unroll_factor in stages_to_process:
                # XOR fusion with first stage
                if fuse_xor and stage_idx == 0:
                    for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", ("^", p_val, p_val, p_node_val), "hash")

                if has_preloaded and val1 in self.vec_const_map:
                    vc1 = self.vec_const_map[val1]
                    vc3 = self.vec_const_map[val3]
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, vc1), "hash")
                        self.emit("valu", (op3, p_tmp2, p_val, vc3), "hash")
                else:
                    # Use dedicated scratch for constants (not chunk 0's temps)
                    self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)), "hash")
                    self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)), "hash")
                    self.flush_schedule()
                    for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                        self.emit("valu", (op1, p_tmp1, p_val, v_const_a), "hash")
                        self.emit("valu", (op3, p_tmp2, p_val, v_const_b), "hash")

                # Only flush between stages if unroll factor is 1
                if unroll_factor == 1:
                    self.flush_schedule()

                for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", (op2, p_val, p_tmp1, p_tmp2), "hash")

                # Flush based on unroll factor - only at unroll boundaries
                if unroll_factor > 1 and (stage_idx + 1) % unroll_factor == 0:
                    self.flush_schedule()
                elif unroll_factor == 1:
                    self.maybe_flush(hash_node.flush_per_stage)

        self.flush_schedule()

    def _compile_index(self, index_node: IndexNode, chunk_scratch, interleave: InterleaveNode):
        has_preloaded = index_node.use_preloaded_consts and 2 in self.vec_const_map
        has_n_nodes = "n_nodes" in self.vec_const_map
        compute_unroll = index_node.compute_unroll
        speculative = index_node.speculative

        if has_preloaded:
            vc_zero = self.vec_const_map[0]
            vc_one = self.vec_const_map[1]
            vc_two = self.vec_const_map[2]
        else:
            vc_zero = vc_one = vc_two = None

        # Helper to conditionally flush based on unroll factor
        def maybe_flush_unroll(step: int):
            if compute_unroll == 1 or step % compute_unroll == 0:
                self.flush_schedule()

        if speculative and has_preloaded:
            # Speculative execution: compute both left (idx*2+1) and right (idx*2+2) children
            # then select based on hash value
            # This trades more compute for potentially better ILP

            # Compute hash % 2 to determine branch
            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two), "index")
            maybe_flush_unroll(1)

            # Compute left child = idx * 2 + 1
            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_tmp1, p_idx, vc_two), "index")
            maybe_flush_unroll(2)

            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_tmp1, p_tmp1, vc_one), "index")  # left = idx*2+1
            maybe_flush_unroll(3)

            # Compute right child = idx * 2 + 2 (reuse p_idx for this)
            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, vc_two), "index")
            maybe_flush_unroll(4)

            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, vc_two), "index")  # right = idx*2+2
            maybe_flush_unroll(5)

            # Select based on hash: if hash%2==0 take left (tmp1), else take right (p_idx)
            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero), "index")
            self.flush_schedule()

            for c, (p_idx, p_val, p_node_val, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("flow", ("vselect", p_idx, p_tmp2, p_tmp1, p_idx), "index")
            self.flush_schedule()

        elif index_node.strategy == IndexStrategy.VSELECT and has_preloaded:
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two), "index")
                self.emit("valu", ("*", p_idx, p_idx, vc_two), "index")
            maybe_flush_unroll(1)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero), "index")
            maybe_flush_unroll(2)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("flow", ("vselect", p_tmp1, p_tmp2, vc_one, vc_two), "index")
            maybe_flush_unroll(3)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1), "index")
            self.flush_schedule()

        elif index_node.strategy == IndexStrategy.ARITHMETIC and has_preloaded:
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two), "index")
            maybe_flush_unroll(1)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero), "index")
            maybe_flush_unroll(2)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2), "index")
            maybe_flush_unroll(3)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, vc_two), "index")
            maybe_flush_unroll(4)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1), "index")
            self.flush_schedule()

        elif index_node.strategy in (IndexStrategy.MULTIPLY_ADD, IndexStrategy.BRANCHLESS) and has_preloaded:
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two), "index")
            maybe_flush_unroll(1)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero), "index")
            maybe_flush_unroll(2)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2), "index")
            maybe_flush_unroll(3)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1), "index")
            self.flush_schedule()

        else:
            # Fallback
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.get_const(2)), "index")
            maybe_flush_unroll(1)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, p_tmp1), "index")
                self.emit("valu", ("*", p_idx, p_idx, p_tmp1), "index")
            maybe_flush_unroll(2)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.get_const(0)), "index")
            maybe_flush_unroll(3)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, p_tmp1), "index")
            maybe_flush_unroll(4)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, p_tmp1, p_tmp2), "index")
            maybe_flush_unroll(5)
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1), "index")
            self.flush_schedule()

        # Bounds check
        if has_n_nodes:
            vc_n = self.vec_const_map["n_nodes"]
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, vc_n), "index")
        else:
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]), "index")
            self.flush_schedule()
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1), "index")
        self.flush_schedule()

        # Bounds application based on mode
        if index_node.bounds_check_mode == "multiply":
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, p_tmp2), "index")
        elif index_node.bounds_check_mode == "select":
            if has_preloaded:
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("flow", ("vselect", p_idx, p_tmp2, p_idx, vc_zero), "index")
            else:
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("*", p_idx, p_idx, p_tmp2), "index")
        else:  # mask
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, p_tmp2), "index")
        self.flush_schedule()

    def _compile_store(self, store: StoreNode, chunk_scratch, batch_indices, interleave: InterleaveNode,
                       memory: Optional[MemoryNode] = None):
        batch_stores = store.batch_stores
        write_combining = store.write_combining
        n_chunks = len(chunk_scratch)
        coalesce_stores = memory.coalesce_stores if memory else False
        store_buffer_depth = memory.store_buffer_depth if memory else 0

        # Address computation for all chunks
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]), "store")
            self.emit("load", ("const", p_addr2, batch_indices[c]), "store")
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1), "store")
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2), "store")
        self.maybe_flush(store.flush_after_addr)

        # With store_buffer_depth, we delay flushing to accumulate more stores
        # depth 0 = normal flushing, depth 1 = batch all stores of same type, depth 2 = batch all stores
        flush_threshold = store_buffer_depth + 1

        # If coalesce_stores is enabled, override store_order to group by type (idx first, then val)
        effective_store_order = "idx_first" if coalesce_stores else store.store_order

        # Helper to determine if we should flush based on store_buffer_depth
        stores_emitted = [0]  # use list to allow modification in closure

        def maybe_flush_buffered():
            stores_emitted[0] += 1
            if stores_emitted[0] >= flush_threshold * n_chunks:
                self.flush_schedule()
                stores_emitted[0] = 0

        if batch_stores >= 2 and n_chunks >= batch_stores:
            # Batch stores: process batch_stores chunks together before flushing
            # This can improve memory throughput by having more stores in flight

            if effective_store_order == "idx_first":
                # Issue all index stores in batches
                for batch_start in range(0, n_chunks, batch_stores):
                    batch_end = min(batch_start + batch_stores, n_chunks)
                    for c in range(batch_start, batch_end):
                        p_idx, p_val, _, _, _, p_addr1, p_addr2 = chunk_scratch[c]
                        self.emit("store", ("vstore", p_addr1, p_idx), "store")
                    if write_combining:
                        # Flush after each batch when write combining to allow coalescing
                        self.flush_schedule()
                # Then all value stores in batches
                for batch_start in range(0, n_chunks, batch_stores):
                    batch_end = min(batch_start + batch_stores, n_chunks)
                    for c in range(batch_start, batch_end):
                        p_idx, p_val, _, _, _, p_addr1, p_addr2 = chunk_scratch[c]
                        self.emit("store", ("vstore", p_addr2, p_val), "store")
                    if write_combining:
                        self.flush_schedule()

            elif effective_store_order == "val_first":
                # Value stores first in batches
                for batch_start in range(0, n_chunks, batch_stores):
                    batch_end = min(batch_start + batch_stores, n_chunks)
                    for c in range(batch_start, batch_end):
                        p_idx, p_val, _, _, _, p_addr1, p_addr2 = chunk_scratch[c]
                        self.emit("store", ("vstore", p_addr2, p_val), "store")
                    if write_combining:
                        self.flush_schedule()
                # Then index stores in batches
                for batch_start in range(0, n_chunks, batch_stores):
                    batch_end = min(batch_start + batch_stores, n_chunks)
                    for c in range(batch_start, batch_end):
                        p_idx, p_val, _, _, _, p_addr1, p_addr2 = chunk_scratch[c]
                        self.emit("store", ("vstore", p_addr1, p_idx), "store")
                    if write_combining:
                        self.flush_schedule()

            else:  # interleaved with batching
                for batch_start in range(0, n_chunks, batch_stores):
                    batch_end = min(batch_start + batch_stores, n_chunks)
                    for c in range(batch_start, batch_end):
                        p_idx, p_val, _, _, _, p_addr1, p_addr2 = chunk_scratch[c]
                        self.emit("store", ("vstore", p_addr1, p_idx), "store")
                        self.emit("store", ("vstore", p_addr2, p_val), "store")
                    if write_combining:
                        self.flush_schedule()

        else:
            # Original store ordering without batching
            if effective_store_order == "idx_first":
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr1, p_idx), "store")
                if write_combining:
                    self.flush_schedule()
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr2, p_val), "store")
            elif effective_store_order == "val_first":
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr2, p_val), "store")
                if write_combining:
                    self.flush_schedule()
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr1, p_idx), "store")
            else:  # interleaved
                for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("store", ("vstore", p_addr1, p_idx), "store")
                    self.emit("store", ("vstore", p_addr2, p_val), "store")

        self.flush_schedule()


# =============================================================================
# Fitness Evaluation
# =============================================================================

TEST_FOREST_HEIGHT = 10
TEST_ROUNDS = 16
TEST_BATCH_SIZE = 256
TEST_SEED = 123


def evaluate_program(program: ProgramNode, timeout_cycles: int = 50000) -> float:
    try:
        random.seed(TEST_SEED)
        forest = Tree.generate(TEST_FOREST_HEIGHT)
        inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
        mem = build_mem_image(forest, inp)

        builder = GPKernelBuilderV3(program)
        instrs = builder.build(forest.height, len(forest.values),
                               len(inp.indices), TEST_ROUNDS)

        try:
            from numba_machine import NumbaMachine
            machine = NumbaMachine(mem, instrs)
            machine.enable_pause = True

            ref_gen = reference_kernel2(list(mem), {})
            ref_mem = None
            for ref_mem in ref_gen:
                machine.run(max_cycles=timeout_cycles)
                if machine.cycle >= timeout_cycles:
                    return float('inf')

            if ref_mem is not None:
                inp_values_p = ref_mem[6]
                if list(machine.mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE]) != \
                   ref_mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE]:
                    return float('inf')

            return float(machine.cycle)
        except ImportError:
            return float('inf')
    except Exception as e:
        return float('inf')


# =============================================================================
# GP Algorithm with Diversity
# =============================================================================

def compute_signature(program: ProgramNode) -> str:
    """Compute a signature for diversity tracking"""
    loop = program.main_loop
    body = loop.body
    return (
        f"{loop.structure.value}_{loop.parallel_chunks}_"
        f"{body.gather.strategy.value}_{body.hash_comp.strategy.value}_"
        f"{body.index_comp.strategy.value}_{loop.interleave.strategy.value}_"
        f"{loop.pipeline.enabled}_{loop.memory.prefetch_distance}"
    )


def fitness_sharing(fitnesses: List[float], programs: List[ProgramNode],
                    sigma: float = 0.5) -> List[float]:
    """Apply fitness sharing to promote diversity"""
    n = len(fitnesses)
    signatures = [compute_signature(p) for p in programs]

    shared = []
    for i in range(n):
        if fitnesses[i] == float('inf'):
            shared.append(float('inf'))
            continue

        niche_count = sum(1 for j in range(n)
                        if signatures[i] == signatures[j] and fitnesses[j] < float('inf'))
        shared.append(fitnesses[i] * (1 + sigma * (niche_count - 1)))

    return shared


class GeneticProgrammingV3:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.7,
        point_mutation_rate: float = 0.2,
        subtree_mutation_rate: float = 0.1,
        elitism: int = 3,
        tournament_size: int = 3,
        max_depth: int = 12,
        random_immigrant_rate: float = 0.1,
        use_fitness_sharing: bool = True,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.point_mutation_rate = point_mutation_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.random_immigrant_rate = random_immigrant_rate
        self.use_fitness_sharing = use_fitness_sharing

        self.population: List[ProgramNode] = []
        self.fitnesses: List[float] = []
        self.best_program: Optional[ProgramNode] = None
        self.best_fitness: float = float('inf')
        self.history: List[dict] = []

    def initialize_population(self):
        self.population = []

        # Seed with different strategies
        for seed_type in ["baseline", "minimal_flush", "max_interleave",
                          "pipelined", "memory_optimized"]:
            self.population.append(seeded_program(seed_type))

        # Fill with random
        while len(self.population) < self.population_size:
            self.population.append(random_program())

    def tournament_select(self, use_shared: bool = True) -> ProgramNode:
        indices = random.sample(range(len(self.population)), self.tournament_size)
        fitness_list = self.shared_fitnesses if use_shared and self.use_fitness_sharing else self.fitnesses
        best_idx = min(indices, key=lambda i: fitness_list[i])
        return self.population[best_idx].clone()

    def evaluate_population(self):
        timeout = int(self.best_fitness * 1.5) if self.best_fitness < float('inf') else 50000

        self.fitnesses = [evaluate_program(p, timeout) for p in self.population]

        if self.use_fitness_sharing:
            self.shared_fitnesses = fitness_sharing(self.fitnesses, self.population)
        else:
            self.shared_fitnesses = self.fitnesses

        for i, fitness in enumerate(self.fitnesses):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_program = self.population[i].clone()

    def evolve_generation(self):
        new_population = []

        # Elitism (using raw fitness)
        sorted_indices = sorted(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        for i in range(self.elitism):
            if self.fitnesses[sorted_indices[i]] < float('inf'):
                new_population.append(self.population[sorted_indices[i]].clone())

        # Random immigrants
        n_immigrants = int(self.population_size * self.random_immigrant_rate)
        for _ in range(n_immigrants):
            new_population.append(random_program())

        # Fill with offspring
        while len(new_population) < self.population_size:
            op = random.random()

            if op < self.crossover_rate:
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                child = subtree_crossover(p1, p2)

            elif op < self.crossover_rate + self.subtree_mutation_rate:
                parent = self.tournament_select()
                child = subtree_mutation(parent)

            else:
                parent = self.tournament_select()
                child = point_mutation(parent, self.point_mutation_rate)

            if child.depth() <= self.max_depth:
                new_population.append(child)
            else:
                new_population.append(self.tournament_select())

        self.population = new_population

    def run(self, verbose: bool = True) -> Tuple[ProgramNode, float]:
        print(f"Starting GP V3: {self.population_size} individuals, {self.generations} generations")
        print(f"Extensions: Pipeline, Interleave, Memory, Unroll, Register, Phase Fusion")
        print()

        self.initialize_population()

        for gen in range(self.generations):
            start = time.time()
            self.evaluate_population()

            valid = [f for f in self.fitnesses if f < float('inf')]
            avg = sum(valid) / len(valid) if valid else float('inf')
            gen_best = min(valid) if valid else float('inf')

            # Count unique strategies
            signatures = set(compute_signature(p) for p, f in
                           zip(self.population, self.fitnesses) if f < float('inf'))

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'gen_best': gen_best,
                'gen_avg': avg,
                'valid_count': len(valid),
                'unique_strategies': len(signatures),
                'time': time.time() - start,
            })

            if verbose:
                print(f"Gen {gen:3d}: best={self.best_fitness:,.0f}, "
                      f"gen_best={gen_best:,.0f}, gen_avg={avg:,.0f}, "
                      f"valid={len(valid)}/{self.population_size}, "
                      f"unique={len(signatures)}, "
                      f"time={time.time()-start:.1f}s")

            if self.best_fitness < 1500:
                print(f"\nExcellent solution found!")
                break

            self.evolve_generation()

        print(f"\nBest: {self.best_fitness:,.0f} cycles")
        if self.best_program:
            loop = self.best_program.main_loop
            phases = loop.body
            print(f"Structure: {loop.structure.value}, chunks={loop.parallel_chunks}")
            print(f"  Gather: {phases.gather.strategy.value}")
            print(f"  Hash: {phases.hash_comp.strategy.value}")
            print(f"  Index: {phases.index_comp.strategy.value}")
            print(f"  Interleave: {loop.interleave.strategy.value}")
            print(f"  Pipeline: enabled={loop.pipeline.enabled}, depth={loop.pipeline.pipeline_depth}")
            print(f"  Memory: prefetch={loop.memory.prefetch_distance}, addr_gen={loop.memory.address_gen.value}")

        return self.best_program, self.best_fitness

    def save_results(self, filename: str = "gp_v3_results.json"):
        def serialize_node(node: GPNode) -> dict:
            if isinstance(node, ProgramNode):
                return {
                    'type': 'Program',
                    'setup': serialize_node(node.setup),
                    'main_loop': serialize_node(node.main_loop)
                }
            elif isinstance(node, SetupNode):
                return {
                    'type': 'Setup',
                    'preload_scalars': node.preload_scalars,
                    'preload_vectors': node.preload_vectors,
                    'preload_hash_consts': node.preload_hash_consts,
                    'preload_n_nodes': node.preload_n_nodes
                }
            elif isinstance(node, LoopNode):
                return {
                    'type': 'Loop',
                    'structure': node.structure.value,
                    'parallel_chunks': node.parallel_chunks,
                    'outer_unroll': node.outer_unroll,
                    'chunk_unroll': node.chunk_unroll,
                    'body': serialize_node(node.body),
                    'pipeline': serialize_node(node.pipeline),
                    'interleave': serialize_node(node.interleave),
                    'memory': serialize_node(node.memory),
                    'registers': serialize_node(node.registers)
                }
            elif isinstance(node, PipelineNode):
                return {
                    'type': 'Pipeline',
                    'enabled': node.enabled,
                    'pipeline_depth': node.pipeline_depth,
                    'schedule': node.schedule.value,
                    'prologue_unroll': node.prologue_unroll,
                    'epilogue_drain': node.epilogue_drain
                }
            elif isinstance(node, InterleaveNode):
                return {
                    'type': 'Interleave',
                    'strategy': node.strategy.value,
                    'lookahead_depth': node.lookahead_depth,
                    'min_slot_fill': node.min_slot_fill,
                    'allow_cross_chunk': node.allow_cross_chunk
                }
            elif isinstance(node, MemoryNode):
                return {
                    'type': 'Memory',
                    'prefetch_distance': node.prefetch_distance,
                    'store_buffer_depth': node.store_buffer_depth,
                    'address_gen': node.address_gen.value,
                    'access_order': node.access_order.value,
                    'coalesce_loads': node.coalesce_loads,
                    'coalesce_stores': node.coalesce_stores
                }
            elif isinstance(node, RegisterNode):
                return {
                    'type': 'Register',
                    'allocation': node.allocation.value,
                    'reuse_policy': node.reuse_policy.value,
                    'spill_threshold': node.spill_threshold,
                    'vector_alignment': node.vector_alignment,
                    'reserve_temps': node.reserve_temps
                }
            elif isinstance(node, PhaseSequenceNode):
                return {
                    'type': 'PhaseSequence',
                    'phase_order': node.phase_order,
                    'fusion_mode': node.fusion_mode.value,
                    'gather': serialize_node(node.gather),
                    'hash': serialize_node(node.hash_comp),
                    'index': serialize_node(node.index_comp),
                    'store': serialize_node(node.store)
                }
            elif isinstance(node, GatherNode):
                return {
                    'type': 'Gather',
                    'strategy': node.strategy.value,
                    'flush_after_addr': node.flush_after_addr,
                    'flush_per_element': node.flush_per_element,
                    'inner_unroll': node.inner_unroll,
                    'vector_grouping': node.vector_grouping,
                    'addr_compute_ahead': node.addr_compute_ahead
                }
            elif isinstance(node, HashNode):
                return {
                    'type': 'Hash',
                    'strategy': node.strategy.value,
                    'flush_per_stage': node.flush_per_stage,
                    'use_preloaded_consts': node.use_preloaded_consts,
                    'stage_unroll': list(node.stage_unroll),
                    'fuse_xor_with_stage1': node.fuse_xor_with_stage1,
                    'cross_chunk_interleave': node.cross_chunk_interleave
                }
            elif isinstance(node, IndexNode):
                return {
                    'type': 'Index',
                    'strategy': node.strategy.value,
                    'flush_per_op': node.flush_per_op,
                    'use_preloaded_consts': node.use_preloaded_consts,
                    'compute_unroll': node.compute_unroll,
                    'bounds_check_mode': node.bounds_check_mode,
                    'speculative': node.speculative
                }
            elif isinstance(node, StoreNode):
                return {
                    'type': 'Store',
                    'flush_after_addr': node.flush_after_addr,
                    'batch_stores': node.batch_stores,
                    'store_order': node.store_order,
                    'write_combining': node.write_combining
                }
            return {}

        results = {
            'best_fitness': self.best_fitness,
            'best_program': serialize_node(self.best_program) if self.best_program else None,
            'history': self.history,
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='GP V3 optimizer for VLIW kernel')
    parser.add_argument('--generations', '-g', type=int, default=30)
    parser.add_argument('--population', '-p', type=int, default=50)
    parser.add_argument('--output', '-o', type=str, default='gp_v3_results.json')

    args = parser.parse_args()

    gp = GeneticProgrammingV3(
        population_size=args.population,
        generations=args.generations,
    )

    best_program, best_fitness = gp.run()
    gp.save_results(args.output)

    print(f"\nSpeedup over baseline: {147734 / best_fitness:.2f}x")


if __name__ == "__main__":
    main()
