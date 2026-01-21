"""
Enhanced Genetic Programming optimizer for VLIW kernel synthesis.

V2 improvements:
- Richer node parameters (unrolling, scheduling hints, more flush control)
- Deeper tree structure (nested loops, phase reordering)
- Diversity mechanisms (random immigrants, fitness sharing)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    build_mem_image,
    reference_kernel2,
)


# =============================================================================
# Type System
# =============================================================================

class GPType(Enum):
    PROGRAM = auto()
    SETUP = auto()
    LOOP = auto()
    PHASE_SEQ = auto()
    PHASE_GROUP = auto()  # New: group of phases that can be reordered
    GATHER = auto()
    HASH = auto()
    INDEX = auto()
    STORE = auto()
    TRANSFORM = auto()  # New: wrapper for transformations


# =============================================================================
# Base Node
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
# Program Structure Nodes
# =============================================================================

@dataclass
class ProgramNode(GPNode):
    """Root node"""
    setup: 'SetupNode'
    main_loop: 'LoopNode'

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
        return ProgramNode(self.setup.clone(), self.main_loop.clone())


@dataclass
class SetupNode(GPNode):
    """Setup phase with more options"""
    preload_scalars: bool = True
    preload_vectors: bool = True
    preload_hash_consts: bool = True
    preload_n_nodes: bool = False
    # New: memory alignment
    align_vectors: bool = True
    # New: scratch organization
    separate_temps: bool = False

    @property
    def node_type(self) -> GPType:
        return GPType.SETUP

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'SetupNode':
        return SetupNode(
            self.preload_scalars, self.preload_vectors,
            self.preload_hash_consts, self.preload_n_nodes,
            self.align_vectors, self.separate_temps
        )


class LoopStructure(Enum):
    ROUND_BATCH = "round_batch"
    BATCH_ROUND = "batch_round"
    FUSED = "fused"
    CHUNKED = "chunked"
    TILED = "tiled"  # New: 2D tiling


@dataclass
class LoopNode(GPNode):
    """Loop structure with unrolling and nesting"""
    structure: LoopStructure
    parallel_chunks: int
    # New: unrolling factors
    batch_unroll: int = 1
    round_unroll: int = 1
    # New: optional nested loop
    inner_loop: Optional['LoopNode'] = None
    # Body
    body: 'PhaseSequenceNode' = None

    def __post_init__(self):
        if self.body is None:
            self.body = random_phase_sequence()

    @property
    def node_type(self) -> GPType:
        return GPType.LOOP

    def children(self) -> List[GPNode]:
        kids = [self.body]
        if self.inner_loop is not None:
            kids.append(self.inner_loop)
        return kids

    def replace_child(self, index: int, new_child: GPNode):
        if index == 0 and isinstance(new_child, PhaseSequenceNode):
            self.body = new_child
        elif index == 1 and isinstance(new_child, LoopNode):
            self.inner_loop = new_child

    def clone(self) -> 'LoopNode':
        return LoopNode(
            structure=self.structure,
            parallel_chunks=self.parallel_chunks,
            batch_unroll=self.batch_unroll,
            round_unroll=self.round_unroll,
            inner_loop=self.inner_loop.clone() if self.inner_loop else None,
            body=self.body.clone()
        )


class PhaseOrder(Enum):
    """Order of phases within a sequence"""
    STANDARD = "standard"          # gather -> hash -> index -> store
    HASH_FIRST = "hash_first"      # hash -> gather -> index -> store (speculative)
    INTERLEAVED = "interleaved"    # interleave operations
    PIPELINED = "pipelined"        # overlap phases


@dataclass
class PhaseSequenceNode(GPNode):
    """Sequence of phases with ordering options"""
    gather: 'GatherNode'
    hash_comp: 'HashNode'
    index_comp: 'IndexNode'
    store: 'StoreNode'
    # New: phase ordering
    order: PhaseOrder = PhaseOrder.STANDARD
    # New: inter-phase flush control
    flush_after_gather: bool = True
    flush_after_hash: bool = True
    flush_after_index: bool = True

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
            order=self.order,
            flush_after_gather=self.flush_after_gather,
            flush_after_hash=self.flush_after_hash,
            flush_after_index=self.flush_after_index
        )


# =============================================================================
# Phase Nodes (Enhanced)
# =============================================================================

class GatherStrategy(Enum):
    SEQUENTIAL = "sequential"
    DUAL_ADDR = "dual_addr"
    BATCH_ADDR = "batch_addr"
    PIPELINED = "pipelined"
    VECTORIZED = "vectorized"  # New


@dataclass
class GatherNode(GPNode):
    """Gather phase with more parameters"""
    strategy: GatherStrategy
    # Flush control
    flush_after_addr: bool = True
    flush_per_element: bool = False
    flush_per_chunk: bool = False
    # New: unrolling
    unroll_factor: int = 1
    # New: prefetch distance
    prefetch_distance: int = 0
    # New: address computation
    use_add_imm: bool = False

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
            flush_per_chunk=self.flush_per_chunk,
            unroll_factor=self.unroll_factor,
            prefetch_distance=self.prefetch_distance,
            use_add_imm=self.use_add_imm
        )


class HashStrategy(Enum):
    STANDARD = "standard"
    FUSED = "fused"
    INTERLEAVED = "interleaved"
    UNROLLED = "unrolled"  # New
    PIPELINED = "pipelined"  # New


@dataclass
class HashNode(GPNode):
    """Hash computation with more parameters"""
    strategy: HashStrategy
    # Flush control
    flush_per_stage: bool = True
    flush_after_xor: bool = True
    # Constants
    use_preloaded_consts: bool = True
    inline_consts: bool = False
    # New: unrolling
    unroll_stages: int = 1  # 1, 2, 3, or 6
    # New: operation fusion
    fuse_xor_with_stage1: bool = False

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
            flush_after_xor=self.flush_after_xor,
            use_preloaded_consts=self.use_preloaded_consts,
            inline_consts=self.inline_consts,
            unroll_stages=self.unroll_stages,
            fuse_xor_with_stage1=self.fuse_xor_with_stage1
        )


class IndexStrategy(Enum):
    VSELECT = "vselect"
    ARITHMETIC = "arithmetic"
    MULTIPLY_ADD = "multiply_add"
    BITWISE = "bitwise"  # New
    BRANCHLESS = "branchless"  # New


@dataclass
class IndexNode(GPNode):
    """Index computation with more parameters"""
    strategy: IndexStrategy
    # Flush control
    flush_per_op: bool = True
    flush_after_mod: bool = True
    flush_after_mul: bool = True
    # Constants
    use_preloaded_consts: bool = True
    # New: bounds check
    speculative_bounds: bool = False  # Check bounds before/after
    # New: index computation fusion
    fuse_mul_add: bool = False

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
            flush_after_mod=self.flush_after_mod,
            flush_after_mul=self.flush_after_mul,
            use_preloaded_consts=self.use_preloaded_consts,
            speculative_bounds=self.speculative_bounds,
            fuse_mul_add=self.fuse_mul_add
        )


@dataclass
class StoreNode(GPNode):
    """Store phase with more parameters"""
    # Flush control
    flush_after_addr: bool = True
    flush_per_store: bool = False
    # New: store ordering
    store_indices_first: bool = True
    # New: address computation
    use_add_imm: bool = False
    # New: coalescing
    coalesce_stores: bool = False

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
            flush_per_store=self.flush_per_store,
            store_indices_first=self.store_indices_first,
            use_add_imm=self.use_add_imm,
            coalesce_stores=self.coalesce_stores
        )


# =============================================================================
# Tree Generation
# =============================================================================

def random_bool() -> bool:
    return random.choice([True, False])


def random_setup() -> SetupNode:
    return SetupNode(
        preload_scalars=random_bool(),
        preload_vectors=random_bool(),
        preload_hash_consts=random_bool(),
        preload_n_nodes=random_bool(),
        align_vectors=random_bool(),
        separate_temps=random_bool()
    )


def random_gather() -> GatherNode:
    return GatherNode(
        strategy=random.choice(list(GatherStrategy)),
        flush_after_addr=random_bool(),
        flush_per_element=random_bool(),
        flush_per_chunk=random_bool(),
        unroll_factor=random.choice([1, 2, 4]),
        prefetch_distance=random.choice([0, 1, 2]),
        use_add_imm=random_bool()
    )


def random_hash() -> HashNode:
    return HashNode(
        strategy=random.choice(list(HashStrategy)),
        flush_per_stage=random_bool(),
        flush_after_xor=random_bool(),
        use_preloaded_consts=random_bool(),
        inline_consts=random_bool(),
        unroll_stages=random.choice([1, 2, 3, 6]),
        fuse_xor_with_stage1=random_bool()
    )


def random_index() -> IndexNode:
    return IndexNode(
        strategy=random.choice(list(IndexStrategy)),
        flush_per_op=random_bool(),
        flush_after_mod=random_bool(),
        flush_after_mul=random_bool(),
        use_preloaded_consts=random_bool(),
        speculative_bounds=random_bool(),
        fuse_mul_add=random_bool()
    )


def random_store() -> StoreNode:
    return StoreNode(
        flush_after_addr=random_bool(),
        flush_per_store=random_bool(),
        store_indices_first=random_bool(),
        use_add_imm=random_bool(),
        coalesce_stores=random_bool()
    )


def random_phase_sequence() -> PhaseSequenceNode:
    return PhaseSequenceNode(
        gather=random_gather(),
        hash_comp=random_hash(),
        index_comp=random_index(),
        store=random_store(),
        order=random.choice(list(PhaseOrder)),
        flush_after_gather=random_bool(),
        flush_after_hash=random_bool(),
        flush_after_index=random_bool()
    )


def random_loop(max_depth: int = 3, current_depth: int = 0) -> LoopNode:
    """Generate random loop, possibly with nested inner loop"""
    has_inner = current_depth < max_depth - 1 and random.random() < 0.3

    return LoopNode(
        structure=random.choice(list(LoopStructure)),
        parallel_chunks=random.choice([4, 8, 16]),
        batch_unroll=random.choice([1, 2, 4]),
        round_unroll=random.choice([1, 2, 4, 8]),
        inner_loop=random_loop(max_depth, current_depth + 1) if has_inner else None,
        body=random_phase_sequence()
    )


def random_program(max_depth: int = 4) -> ProgramNode:
    return ProgramNode(
        setup=random_setup(),
        main_loop=random_loop(max_depth)
    )


def seeded_program(seed_type: str) -> ProgramNode:
    """Generate seeded programs for diversity"""
    if seed_type == "minimal_flush":
        return ProgramNode(
            setup=SetupNode(True, True, True, True, True, False),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=16,
                batch_unroll=1,
                round_unroll=1,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.BATCH_ADDR, False, False, False, 1, 0, False),
                    hash_comp=HashNode(HashStrategy.FUSED, False, False, True, False, 1, False),
                    index_comp=IndexNode(IndexStrategy.ARITHMETIC, False, False, False, True, False, False),
                    store=StoreNode(False, False, True, False, False),
                    order=PhaseOrder.STANDARD,
                    flush_after_gather=False,
                    flush_after_hash=False,
                    flush_after_index=False
                )
            )
        )

    elif seed_type == "max_parallel":
        return ProgramNode(
            setup=SetupNode(True, True, True, True, True, False),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=16,
                batch_unroll=4,
                round_unroll=4,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.PIPELINED, True, False, False, 2, 1, False),
                    hash_comp=HashNode(HashStrategy.INTERLEAVED, False, False, True, False, 2, True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, False, False, True, False, True),
                    store=StoreNode(True, False, True, False, True),
                    order=PhaseOrder.PIPELINED,
                    flush_after_gather=True,
                    flush_after_hash=False,
                    flush_after_index=False
                )
            )
        )

    elif seed_type == "deep_nested":
        inner = LoopNode(
            structure=LoopStructure.FUSED,
            parallel_chunks=8,
            batch_unroll=2,
            round_unroll=2,
            body=random_phase_sequence()
        )
        return ProgramNode(
            setup=SetupNode(True, True, True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.TILED,
                parallel_chunks=4,
                batch_unroll=1,
                round_unroll=1,
                inner_loop=inner,
                body=random_phase_sequence()
            )
        )

    else:  # baseline
        return ProgramNode(
            setup=SetupNode(True, True, True, False, True, False),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                batch_unroll=1,
                round_unroll=1,
                body=random_phase_sequence()
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
    """Type-safe subtree crossover"""
    child = parent1.clone()

    exchangeable_types = [GPType.SETUP, GPType.LOOP, GPType.PHASE_SEQ,
                         GPType.GATHER, GPType.HASH, GPType.INDEX, GPType.STORE]
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
    """Mutate individual parameters"""
    tree = tree.clone()

    for node, parent, idx in tree.all_nodes():
        if random.random() > rate:
            continue

        if isinstance(node, SetupNode):
            field = random.choice(['preload_scalars', 'preload_vectors',
                                   'preload_hash_consts', 'preload_n_nodes',
                                   'align_vectors', 'separate_temps'])
            setattr(node, field, not getattr(node, field))

        elif isinstance(node, LoopNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.structure = random.choice(list(LoopStructure))
            elif choice == 1:
                node.parallel_chunks = random.choice([4, 8, 16])
            elif choice == 2:
                node.batch_unroll = random.choice([1, 2, 4])
            else:
                node.round_unroll = random.choice([1, 2, 4, 8])

        elif isinstance(node, PhaseSequenceNode):
            choice = random.randint(0, 3)
            if choice == 0:
                node.order = random.choice(list(PhaseOrder))
            elif choice == 1:
                node.flush_after_gather = not node.flush_after_gather
            elif choice == 2:
                node.flush_after_hash = not node.flush_after_hash
            else:
                node.flush_after_index = not node.flush_after_index

        elif isinstance(node, GatherNode):
            choice = random.randint(0, 6)
            if choice == 0:
                node.strategy = random.choice(list(GatherStrategy))
            elif choice == 1:
                node.flush_after_addr = not node.flush_after_addr
            elif choice == 2:
                node.flush_per_element = not node.flush_per_element
            elif choice == 3:
                node.flush_per_chunk = not node.flush_per_chunk
            elif choice == 4:
                node.unroll_factor = random.choice([1, 2, 4])
            elif choice == 5:
                node.prefetch_distance = random.choice([0, 1, 2])
            else:
                node.use_add_imm = not node.use_add_imm

        elif isinstance(node, HashNode):
            choice = random.randint(0, 6)
            if choice == 0:
                node.strategy = random.choice(list(HashStrategy))
            elif choice == 1:
                node.flush_per_stage = not node.flush_per_stage
            elif choice == 2:
                node.flush_after_xor = not node.flush_after_xor
            elif choice == 3:
                node.use_preloaded_consts = not node.use_preloaded_consts
            elif choice == 4:
                node.inline_consts = not node.inline_consts
            elif choice == 5:
                node.unroll_stages = random.choice([1, 2, 3, 6])
            else:
                node.fuse_xor_with_stage1 = not node.fuse_xor_with_stage1

        elif isinstance(node, IndexNode):
            choice = random.randint(0, 6)
            if choice == 0:
                node.strategy = random.choice(list(IndexStrategy))
            elif choice == 1:
                node.flush_per_op = not node.flush_per_op
            elif choice == 2:
                node.flush_after_mod = not node.flush_after_mod
            elif choice == 3:
                node.flush_after_mul = not node.flush_after_mul
            elif choice == 4:
                node.use_preloaded_consts = not node.use_preloaded_consts
            elif choice == 5:
                node.speculative_bounds = not node.speculative_bounds
            else:
                node.fuse_mul_add = not node.fuse_mul_add

        elif isinstance(node, StoreNode):
            choice = random.randint(0, 4)
            if choice == 0:
                node.flush_after_addr = not node.flush_after_addr
            elif choice == 1:
                node.flush_per_store = not node.flush_per_store
            elif choice == 2:
                node.store_indices_first = not node.store_indices_first
            elif choice == 3:
                node.use_add_imm = not node.use_add_imm
            else:
                node.coalesce_stores = not node.coalesce_stores

    return tree


def subtree_mutation(tree: ProgramNode, max_depth: int = 4) -> ProgramNode:
    """Replace a subtree with a new random one"""
    tree = tree.clone()

    replaceable = [
        (GPType.GATHER, random_gather),
        (GPType.HASH, random_hash),
        (GPType.INDEX, random_index),
        (GPType.STORE, random_store),
        (GPType.PHASE_SEQ, random_phase_sequence),
        (GPType.LOOP, lambda: random_loop(max_depth)),
    ]

    target_type, generator = random.choice(replaceable)
    nodes = get_nodes_of_type(tree, target_type)

    if nodes:
        _, parent, idx = random.choice(nodes)
        if parent is not None:
            parent.replace_child(idx, generator())

    return tree


def add_nested_loop(tree: ProgramNode) -> ProgramNode:
    """Add a nested loop to increase depth"""
    tree = tree.clone()
    loop_nodes = get_nodes_of_type(tree, GPType.LOOP)

    for node, _, _ in loop_nodes:
        if isinstance(node, LoopNode) and node.inner_loop is None:
            if random.random() < 0.5:
                node.inner_loop = random_loop(max_depth=2)
                break

    return tree


def remove_nested_loop(tree: ProgramNode) -> ProgramNode:
    """Remove a nested loop to decrease depth"""
    tree = tree.clone()
    loop_nodes = get_nodes_of_type(tree, GPType.LOOP)

    for node, _, _ in loop_nodes:
        if isinstance(node, LoopNode) and node.inner_loop is not None:
            if random.random() < 0.5:
                node.inner_loop = None
                break

    return tree


# =============================================================================
# Code Generator (simplified - uses same structure as v1 but with new params)
# =============================================================================

class GPKernelBuilderV2:
    """Compiles enhanced GP tree to VLIW instructions"""

    def __init__(self, program: ProgramNode):
        self.program = program
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
        # Alignment for vectors if enabled
        setup = self.program.setup
        if setup.align_vectors and length >= VLEN:
            if self.scratch_ptr % VLEN != 0:
                self.scratch_ptr += VLEN - (self.scratch_ptr % VLEN)

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
                _, _, _, writes1, _ = slots_info[i]
                _, _, reads2, writes2, _ = slots_info[j]
                if (writes1 & reads2) or (writes1 & writes2):
                    must_precede[j].add(i)

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

    def maybe_flush(self, should: bool):
        if should:
            self.flush_schedule()

    def build(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int) -> List[dict]:
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp1, i))
            self.flush_schedule()
            self.emit("load", ("load", self.scratch[v], tmp1))
            self.flush_schedule()

        self._compile_setup(self.program.setup)
        self.instrs.append({"flow": [("pause",)]})
        self._compile_loop(self.program.main_loop, batch_size, rounds)
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

        for r in range(rounds):
            for cg in range(chunks_per_round):
                batch_indices = [cg * n_chunks * chunk_size + c * chunk_size
                                for c in range(n_chunks)]
                self._compile_phases(loop.body, chunk_scratch, batch_indices)

    def _compile_phases(self, seq: PhaseSequenceNode, chunk_scratch, batch_indices):
        n_chunks = len(chunk_scratch)

        # Phase 1: Load
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]))
            self.emit("load", ("const", p_addr2, batch_indices[c]))
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1))
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
        self.flush_schedule()
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("vload", p_idx, p_addr1))
            self.emit("load", ("vload", p_val, p_addr2))
        self.flush_schedule()

        # Gather
        self._compile_gather(seq.gather, chunk_scratch)
        self.maybe_flush(seq.flush_after_gather)

        # XOR
        for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("^", p_val, p_val, p_node_val))
        self.maybe_flush(seq.hash_comp.flush_after_xor)

        # Hash
        self._compile_hash(seq.hash_comp, chunk_scratch)
        self.maybe_flush(seq.flush_after_hash)

        # Index
        self._compile_index(seq.index_comp, chunk_scratch)
        self.maybe_flush(seq.flush_after_index)

        # Store
        self._compile_store(seq.store, chunk_scratch, batch_indices)

    def _compile_gather(self, gather: GatherNode, chunk_scratch):
        n_chunks = len(chunk_scratch)

        for vi in range(VLEN):
            for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
            self.maybe_flush(gather.flush_after_addr)
            for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                self.emit("load", ("load", p_node_val + vi, p_addr1))
                if (c + 1) % 2 == 0:
                    self.flush_schedule()
            if n_chunks % 2 != 0:
                self.flush_schedule()
            self.maybe_flush(gather.flush_per_element)

    def _compile_hash(self, hash_node: HashNode, chunk_scratch):
        has_preloaded = hash_node.use_preloaded_consts and len(self.vec_const_map) > 3

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if has_preloaded and val1 in self.vec_const_map:
                vc1 = self.vec_const_map[val1]
                vc3 = self.vec_const_map[val3]
                for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", (op1, p_tmp1, p_val, vc1))
                    self.emit("valu", (op3, p_tmp2, p_val, vc3))
            else:
                v_const_a = chunk_scratch[0][3]
                v_const_b = chunk_scratch[0][4]
                self.emit("valu", ("vbroadcast", v_const_a, self.get_const(val1)))
                self.emit("valu", ("vbroadcast", v_const_b, self.get_const(val3)))
                self.flush_schedule()
                for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", (op1, p_tmp1, p_val, v_const_a))
                    self.emit("valu", (op3, p_tmp2, p_val, v_const_b))

            self.flush_schedule()
            for c, (_, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", (op2, p_val, p_tmp1, p_tmp2))
            self.maybe_flush(hash_node.flush_per_stage)

        self.flush_schedule()

    def _compile_index(self, index_node: IndexNode, chunk_scratch):
        has_preloaded = index_node.use_preloaded_consts and 2 in self.vec_const_map
        has_n_nodes = "n_nodes" in self.vec_const_map

        if has_preloaded:
            vc_zero = self.vec_const_map[0]
            vc_one = self.vec_const_map[1]
            vc_two = self.vec_const_map[2]

            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two))
                self.emit("valu", ("*", p_idx, p_idx, vc_two))
            self.maybe_flush(index_node.flush_after_mod)

            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
            self.flush_schedule()

            if index_node.strategy == IndexStrategy.VSELECT:
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("flow", ("vselect", p_tmp1, p_tmp2, vc_one, vc_two))
                self.flush_schedule()
            else:
                for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))
                self.flush_schedule()

            self.maybe_flush(index_node.flush_after_mul)

            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
            self.flush_schedule()

            # Bounds
            if has_n_nodes:
                vc_n = self.vec_const_map["n_nodes"]
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, vc_n))
            else:
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
                self.flush_schedule()
                for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                    self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
            self.flush_schedule()

            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
            self.flush_schedule()

        else:
            # Fallback without preloaded constants
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.get_const(2)))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, p_tmp1))
                self.emit("valu", ("*", p_idx, p_idx, p_tmp1))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.get_const(0)))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, p_tmp1))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, p_tmp1, p_tmp2))
            self.flush_schedule()
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
            self.flush_schedule()

            # Bounds
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("vbroadcast", p_tmp1, self.scratch["n_nodes"]))
            self.flush_schedule()
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("<", p_tmp2, p_idx, p_tmp1))
            self.flush_schedule()
            for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
            self.flush_schedule()

    def _compile_store(self, store: StoreNode, chunk_scratch, batch_indices):
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("load", ("const", p_addr1, batch_indices[c]))
            self.emit("load", ("const", p_addr2, batch_indices[c]))
        for c, (p_idx, p_val, _, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
            self.emit("alu", ("+", p_addr1, self.scratch["inp_indices_p"], p_addr1))
            self.emit("alu", ("+", p_addr2, self.scratch["inp_values_p"], p_addr2))
        self.maybe_flush(store.flush_after_addr)

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


def evaluate_program(program: ProgramNode, timeout_cycles: int = 50000) -> float:
    try:
        random.seed(TEST_SEED)
        forest = Tree.generate(TEST_FOREST_HEIGHT)
        inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
        mem = build_mem_image(forest, inp)

        builder = GPKernelBuilderV2(program)
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
            pass

        # Fallback
        machine = Machine(mem, instrs, builder.debug_info(), n_cores=N_CORES)
        machine.enable_pause = True
        machine.enable_debug = False

        ref_gen = reference_kernel2(list(mem), {})
        ref_mem = None
        for ref_mem in ref_gen:
            machine.run(max_cycles=timeout_cycles)
            if machine.cycle >= timeout_cycles:
                return float('inf')

        if ref_mem is not None:
            inp_values_p = ref_mem[6]
            if machine.mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE] != \
               ref_mem[inp_values_p:inp_values_p + TEST_BATCH_SIZE]:
                return float('inf')

        return float(machine.cycle)
    except Exception:
        return float('inf')


# =============================================================================
# GP Algorithm with Diversity
# =============================================================================

def fitness_sharing(fitnesses: List[float], population: List[ProgramNode], sigma: float = 0.3) -> List[float]:
    """Apply fitness sharing to maintain diversity"""
    n = len(population)
    shared = fitnesses.copy()

    for i in range(n):
        if fitnesses[i] == float('inf'):
            continue
        niche_count = 1.0
        for j in range(n):
            if i != j and fitnesses[j] != float('inf'):
                # Simple similarity: same structure and chunks
                sim = 0.0
                if population[i].main_loop.structure == population[j].main_loop.structure:
                    sim += 0.3
                if population[i].main_loop.parallel_chunks == population[j].main_loop.parallel_chunks:
                    sim += 0.3
                if population[i].main_loop.body.gather.strategy == population[j].main_loop.body.gather.strategy:
                    sim += 0.2
                if population[i].main_loop.body.hash_comp.strategy == population[j].main_loop.body.hash_comp.strategy:
                    sim += 0.2

                if sim > sigma:
                    niche_count += 1.0 - (sim - sigma) / (1.0 - sigma)

        shared[i] = fitnesses[i] * niche_count

    return shared


class GeneticProgrammingV2:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.6,
        point_mutation_rate: float = 0.25,
        subtree_mutation_rate: float = 0.15,
        structure_mutation_rate: float = 0.1,  # Add/remove nested loops
        elitism: int = 2,
        tournament_size: int = 3,
        max_depth: int = 5,
        random_immigrant_rate: float = 0.1,  # Inject fresh random trees
        use_fitness_sharing: bool = True,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.point_mutation_rate = point_mutation_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.structure_mutation_rate = structure_mutation_rate
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

        # Seeded diversity
        for seed_type in ["baseline", "minimal_flush", "max_parallel", "deep_nested"]:
            self.population.append(seeded_program(seed_type))

        # Random with varying depths
        while len(self.population) < self.population_size:
            depth = random.randint(2, self.max_depth)
            self.population.append(random_program(depth))

    def tournament_select(self, use_shared: bool = True) -> ProgramNode:
        fits = self.shared_fitnesses if use_shared and self.use_fitness_sharing else self.fitnesses
        indices = random.sample(range(len(self.population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: fits[i])
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

        # Elitism (use raw fitness, not shared)
        sorted_indices = sorted(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        for i in range(self.elitism):
            if self.fitnesses[sorted_indices[i]] < float('inf'):
                new_population.append(self.population[sorted_indices[i]].clone())

        # Random immigrants
        n_immigrants = int(self.population_size * self.random_immigrant_rate)
        for _ in range(n_immigrants):
            depth = random.randint(2, self.max_depth)
            new_population.append(random_program(depth))

        # Fill with offspring
        while len(new_population) < self.population_size:
            op = random.random()

            if op < self.crossover_rate:
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                child = subtree_crossover(p1, p2)

            elif op < self.crossover_rate + self.subtree_mutation_rate:
                parent = self.tournament_select()
                child = subtree_mutation(parent, self.max_depth)

            elif op < self.crossover_rate + self.subtree_mutation_rate + self.structure_mutation_rate:
                parent = self.tournament_select()
                if random.random() < 0.5:
                    child = add_nested_loop(parent)
                else:
                    child = remove_nested_loop(parent)

            else:
                parent = self.tournament_select()
                child = point_mutation(parent, self.point_mutation_rate)

            if child.depth() <= self.max_depth:
                new_population.append(child)
            else:
                new_population.append(self.tournament_select())

        self.population = new_population

    def run(self, verbose: bool = True) -> Tuple[ProgramNode, float]:
        print(f"Starting GP V2: {self.population_size} individuals, {self.generations} generations")
        print(f"  Random immigrants: {self.random_immigrant_rate*100:.0f}%, Fitness sharing: {self.use_fitness_sharing}")
        print()

        self.initialize_population()

        for gen in range(self.generations):
            start = time.time()
            self.evaluate_population()

            valid = [f for f in self.fitnesses if f < float('inf')]
            avg = sum(valid) / len(valid) if valid else float('inf')
            gen_best = min(valid) if valid else float('inf')

            # Diversity metric
            unique_strategies = len(set(
                (p.main_loop.structure, p.main_loop.parallel_chunks,
                 p.main_loop.body.gather.strategy, p.main_loop.body.hash_comp.strategy)
                for p in self.population
            ))

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'gen_best': gen_best,
                'gen_avg': avg,
                'valid_count': len(valid),
                'unique_strategies': unique_strategies,
                'time': time.time() - start,
            })

            if verbose:
                print(f"Gen {gen:3d}: best={self.best_fitness:,.0f}, "
                      f"gen_best={gen_best:,.0f}, avg={avg:,.0f}, "
                      f"valid={len(valid)}/{self.population_size}, "
                      f"diversity={unique_strategies}, "
                      f"time={time.time()-start:.1f}s")

            if self.best_fitness < 1500:
                print(f"\nExcellent solution found!")
                break

            self.evolve_generation()

        print(f"\nBest: {self.best_fitness:,.0f} cycles")
        if self.best_program:
            loop = self.best_program.main_loop
            print(f"Structure: {loop.structure.value}, chunks={loop.parallel_chunks}")
            print(f"  Unroll: batch={loop.batch_unroll}, round={loop.round_unroll}")
            print(f"  Gather: {loop.body.gather.strategy.value}")
            print(f"  Hash: {loop.body.hash_comp.strategy.value}")
            print(f"  Index: {loop.body.index_comp.strategy.value}")
            print(f"  Depth: {self.best_program.depth()}")

        return self.best_program, self.best_fitness

    def save_results(self, filename: str = "gp_v2_results.json"):
        results = {
            'best_fitness': self.best_fitness,
            'history': self.history,
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='GP V2 optimizer')
    parser.add_argument('--generations', '-g', type=int, default=30)
    parser.add_argument('--population', '-p', type=int, default=50)
    parser.add_argument('--immigrants', '-i', type=float, default=0.1)
    parser.add_argument('--no-sharing', action='store_true')
    parser.add_argument('--output', '-o', type=str, default='gp_v2_results.json')

    args = parser.parse_args()

    gp = GeneticProgrammingV2(
        population_size=args.population,
        generations=args.generations,
        random_immigrant_rate=args.immigrants,
        use_fitness_sharing=not args.no_sharing,
    )

    best_program, best_fitness = gp.run()
    gp.save_results(args.output)


if __name__ == "__main__":
    main()
