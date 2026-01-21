"""
Genetic Programming optimizer for VLIW kernel synthesis.

Instead of parameterizing a fixed template, this evolves the PROGRAM STRUCTURE
itself using tree-based GP with a grammar that captures VLIW constraints.

Key insight: The problem is program synthesis, not parameter tuning.
The genome should BE the program, not parameters TO a program generator.

The grammar ensures all generated programs are valid by construction:
- Type system prevents invalid compositions
- Grammar productions respect VLIW slot limits
- Crossover exchanges semantically meaningful subtrees
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
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
# Type System for GP
# =============================================================================

class GPType(Enum):
    """Types in our GP system - ensures valid compositions"""
    PROGRAM = auto()      # Complete program
    SETUP = auto()        # Setup/initialization phase
    LOOP = auto()         # Loop structure
    PHASE_SEQ = auto()    # Sequence of phases
    GATHER = auto()       # Gather phase (loads)
    HASH = auto()         # Hash computation phase
    INDEX = auto()        # Index computation phase
    STORE = auto()        # Store phase
    VALU_OP = auto()      # Vector ALU operation
    ALU_OP = auto()       # Scalar ALU operation
    ADDR_COMP = auto()    # Address computation
    FLUSH_POINT = auto()  # Synchronization point


# =============================================================================
# AST Node Base Classes
# =============================================================================

class GPNode(ABC):
    """Base class for all GP tree nodes"""

    @property
    @abstractmethod
    def node_type(self) -> GPType:
        """Return the type of this node"""
        pass

    @abstractmethod
    def children(self) -> List['GPNode']:
        """Return list of child nodes"""
        pass

    @abstractmethod
    def replace_child(self, index: int, new_child: 'GPNode') -> None:
        """Replace child at index"""
        pass

    @abstractmethod
    def clone(self) -> 'GPNode':
        """Deep copy this node and all descendants"""
        pass

    def size(self) -> int:
        """Count nodes in subtree"""
        return 1 + sum(c.size() for c in self.children())

    def depth(self) -> int:
        """Max depth of subtree"""
        children = self.children()
        if not children:
            return 1
        return 1 + max(c.depth() for c in children)

    def all_nodes(self) -> List[Tuple['GPNode', Optional['GPNode'], int]]:
        """Return all nodes with parent info: (node, parent, child_index)"""
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
    """Root node: complete program"""
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
        return ProgramNode(
            setup=self.setup.clone(),
            main_loop=self.main_loop.clone()
        )


@dataclass
class SetupNode(GPNode):
    """Setup phase: constant preloading"""
    preload_scalars: bool = True
    preload_vectors: bool = True
    preload_hash_consts: bool = True
    preload_n_nodes: bool = False

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
    """How to structure the iteration"""
    ROUND_BATCH = "round_batch"    # for round: for batch_chunk
    BATCH_ROUND = "batch_round"    # for batch_chunk: for round
    FUSED = "fused"                # Single fused loop
    CHUNKED = "chunked"            # Process chunks in parallel


@dataclass
class LoopNode(GPNode):
    """Loop structure node"""
    structure: LoopStructure
    parallel_chunks: int  # How many vector chunks to process together
    body: 'PhaseSequenceNode'

    @property
    def node_type(self) -> GPType:
        return GPType.LOOP

    def children(self) -> List[GPNode]:
        return [self.body]

    def replace_child(self, index: int, new_child: GPNode):
        if index == 0 and isinstance(new_child, PhaseSequenceNode):
            self.body = new_child

    def clone(self) -> 'LoopNode':
        return LoopNode(
            structure=self.structure,
            parallel_chunks=self.parallel_chunks,
            body=self.body.clone()
        )


@dataclass
class PhaseSequenceNode(GPNode):
    """Sequence of computational phases"""
    gather: 'GatherNode'
    hash_comp: 'HashNode'
    index_comp: 'IndexNode'
    store: 'StoreNode'

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
            store=self.store.clone()
        )


# =============================================================================
# Phase Implementation Nodes
# =============================================================================

class GatherStrategy(Enum):
    """Different gather implementations"""
    SEQUENTIAL = "sequential"      # One load at a time
    DUAL_ADDR = "dual_addr"        # Two addresses, two loads
    BATCH_ADDR = "batch_addr"      # All addresses, then all loads
    PIPELINED = "pipelined"        # Overlap address calc with loads


@dataclass
class GatherNode(GPNode):
    """Gather phase: load values from computed addresses"""
    strategy: GatherStrategy
    flush_after_addr: bool = True
    flush_per_element: bool = False

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
            flush_per_element=self.flush_per_element
        )


class HashStrategy(Enum):
    """Different hash implementations"""
    STANDARD = "standard"          # Flush after each stage
    FUSED = "fused"                # Single flush at end
    INTERLEAVED = "interleaved"    # Interleave across chunks


@dataclass
class HashNode(GPNode):
    """Hash computation phase"""
    strategy: HashStrategy
    flush_per_stage: bool = True
    use_preloaded_consts: bool = True

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
            use_preloaded_consts=self.use_preloaded_consts
        )


class IndexStrategy(Enum):
    """Different index computation approaches"""
    VSELECT = "vselect"            # Use flow vselect
    ARITHMETIC = "arithmetic"       # Pure arithmetic: offset = 2 - cond
    MULTIPLY_ADD = "multiply_add"   # Use multiply_add instruction


@dataclass
class IndexNode(GPNode):
    """Index computation phase"""
    strategy: IndexStrategy
    flush_per_op: bool = True
    use_preloaded_consts: bool = True

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
            use_preloaded_consts=self.use_preloaded_consts
        )


@dataclass
class StoreNode(GPNode):
    """Store phase"""
    flush_after_addr: bool = True

    @property
    def node_type(self) -> GPType:
        return GPType.STORE

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, index: int, new_child: GPNode):
        pass

    def clone(self) -> 'StoreNode':
        return StoreNode(flush_after_addr=self.flush_after_addr)


# =============================================================================
# Tree Generation (Grow/Full methods)
# =============================================================================

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
        flush_per_element=random.choice([True, False])
    )


def random_hash() -> HashNode:
    return HashNode(
        strategy=random.choice(list(HashStrategy)),
        flush_per_stage=random.choice([True, False]),
        use_preloaded_consts=random.choice([True, False])
    )


def random_index() -> IndexNode:
    return IndexNode(
        strategy=random.choice(list(IndexStrategy)),
        flush_per_op=random.choice([True, False]),
        use_preloaded_consts=random.choice([True, False])
    )


def random_store() -> StoreNode:
    return StoreNode(flush_after_addr=random.choice([True, False]))


def random_phase_sequence() -> PhaseSequenceNode:
    return PhaseSequenceNode(
        gather=random_gather(),
        hash_comp=random_hash(),
        index_comp=random_index(),
        store=random_store()
    )


def random_loop() -> LoopNode:
    return LoopNode(
        structure=random.choice(list(LoopStructure)),
        parallel_chunks=random.choice([1, 2, 4, 8, 16]),
        body=random_phase_sequence()
    )


def random_program() -> ProgramNode:
    """Generate a random program tree"""
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
                    index_comp=IndexNode(IndexStrategy.ARITHMETIC, False, True),
                    store=StoreNode(False)
                )
            )
        )

    elif seed_type == "max_parallel":
        return ProgramNode(
            setup=SetupNode(True, True, True, True),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=16,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.PIPELINED, True, False),
                    hash_comp=HashNode(HashStrategy.INTERLEAVED, False, True),
                    index_comp=IndexNode(IndexStrategy.MULTIPLY_ADD, False, True),
                    store=StoreNode(True)
                )
            )
        )

    elif seed_type == "conservative":
        return ProgramNode(
            setup=SetupNode(True, True, True, False),
            main_loop=LoopNode(
                structure=LoopStructure.ROUND_BATCH,
                parallel_chunks=4,
                body=PhaseSequenceNode(
                    gather=GatherNode(GatherStrategy.DUAL_ADDR, True, False),
                    hash_comp=HashNode(HashStrategy.STANDARD, True, True),
                    index_comp=IndexNode(IndexStrategy.VSELECT, True, True),
                    store=StoreNode(True)
                )
            )
        )

    else:  # baseline
        return ProgramNode(
            setup=SetupNode(True, True, True, False),
            main_loop=LoopNode(
                structure=LoopStructure.CHUNKED,
                parallel_chunks=8,
                body=random_phase_sequence()
            )
        )


# =============================================================================
# GP Operators: Crossover and Mutation
# =============================================================================

def get_nodes_of_type(tree: GPNode, target_type: GPType) -> List[Tuple[GPNode, GPNode, int]]:
    """Find all nodes of a given type with their parent info"""
    result = []
    for node, parent, idx in tree.all_nodes():
        if node.node_type == target_type:
            result.append((node, parent, idx))
    return result


def subtree_crossover(parent1: ProgramNode, parent2: ProgramNode) -> ProgramNode:
    """
    Subtree crossover: swap type-compatible subtrees between parents.
    This is the key GP operator - it preserves semantic validity because
    we only swap subtrees of the same type.
    """
    child = parent1.clone()

    # Pick a random type to exchange
    exchangeable_types = [GPType.SETUP, GPType.LOOP, GPType.PHASE_SEQ,
                         GPType.GATHER, GPType.HASH, GPType.INDEX, GPType.STORE]
    target_type = random.choice(exchangeable_types)

    # Find nodes of that type in both parents
    child_nodes = get_nodes_of_type(child, target_type)
    parent2_nodes = get_nodes_of_type(parent2, target_type)

    if child_nodes and parent2_nodes:
        # Pick random nodes to swap
        _, child_parent, child_idx = random.choice(child_nodes)
        donor_node, _, _ = random.choice(parent2_nodes)

        if child_parent is not None:
            child_parent.replace_child(child_idx, donor_node.clone())

    return child


def point_mutation(tree: ProgramNode, rate: float = 0.3) -> ProgramNode:
    """
    Point mutation: mutate individual node parameters.
    Unlike subtree mutation, this makes small local changes.
    """
    tree = tree.clone()

    for node, parent, idx in tree.all_nodes():
        if random.random() > rate:
            continue

        if isinstance(node, SetupNode):
            field = random.choice(['preload_scalars', 'preload_vectors',
                                   'preload_hash_consts', 'preload_n_nodes'])
            setattr(node, field, not getattr(node, field))

        elif isinstance(node, LoopNode):
            if random.random() < 0.5:
                node.structure = random.choice(list(LoopStructure))
            else:
                node.parallel_chunks = random.choice([1, 2, 4, 8, 16])

        elif isinstance(node, GatherNode):
            choice = random.randint(0, 2)
            if choice == 0:
                node.strategy = random.choice(list(GatherStrategy))
            elif choice == 1:
                node.flush_after_addr = not node.flush_after_addr
            else:
                node.flush_per_element = not node.flush_per_element

        elif isinstance(node, HashNode):
            choice = random.randint(0, 2)
            if choice == 0:
                node.strategy = random.choice(list(HashStrategy))
            elif choice == 1:
                node.flush_per_stage = not node.flush_per_stage
            else:
                node.use_preloaded_consts = not node.use_preloaded_consts

        elif isinstance(node, IndexNode):
            choice = random.randint(0, 2)
            if choice == 0:
                node.strategy = random.choice(list(IndexStrategy))
            elif choice == 1:
                node.flush_per_op = not node.flush_per_op
            else:
                node.use_preloaded_consts = not node.use_preloaded_consts

        elif isinstance(node, StoreNode):
            node.flush_after_addr = not node.flush_after_addr

    return tree


def subtree_mutation(tree: ProgramNode) -> ProgramNode:
    """
    Subtree mutation: replace a random subtree with a new random one.
    This provides large jumps in the search space.
    """
    tree = tree.clone()

    # Pick a random node type to replace
    replaceable = [
        (GPType.GATHER, random_gather),
        (GPType.HASH, random_hash),
        (GPType.INDEX, random_index),
        (GPType.STORE, random_store),
        (GPType.PHASE_SEQ, random_phase_sequence),
    ]

    target_type, generator = random.choice(replaceable)
    nodes = get_nodes_of_type(tree, target_type)

    if nodes:
        _, parent, idx = random.choice(nodes)
        if parent is not None:
            parent.replace_child(idx, generator())

    return tree


# =============================================================================
# Code Generator: AST -> VLIW Instructions
# =============================================================================

class GPKernelBuilder:
    """Compiles a GP tree into VLIW instructions"""

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
        """Extract data dependencies for scheduling"""
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
        """Dependency-aware VLIW scheduling"""
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
        """Compile the GP tree into VLIW instructions"""

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

        self.instrs.append({"flow": [("pause",)]})

        # Main loop
        self._compile_loop(self.program.main_loop, batch_size, rounds)

        self.instrs.append({"flow": [("pause",)]})

        return self.instrs

    def _compile_setup(self, setup: SetupNode):
        """Compile setup phase"""
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
        """Compile main loop"""
        n_chunks = loop.parallel_chunks
        chunk_size = VLEN
        chunks_per_round = batch_size // (chunk_size * n_chunks)

        # Allocate chunk scratch
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

        # Main iteration
        for r in range(rounds):
            for cg in range(chunks_per_round):
                batch_indices = [cg * n_chunks * chunk_size + c * chunk_size
                                for c in range(n_chunks)]

                self._compile_phases(loop.body, chunk_scratch, batch_indices)

    def _compile_phases(self, seq: PhaseSequenceNode, chunk_scratch, batch_indices):
        """Compile phase sequence"""
        n_chunks = len(chunk_scratch)

        # Phase 1: Load indices and values
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

        # Phase 2: Gather
        self._compile_gather(seq.gather, chunk_scratch)

        # XOR
        for c, (p_idx, p_val, p_node_val, _, _, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("^", p_val, p_val, p_node_val))
        self.flush_schedule()

        # Phase 3: Hash
        self._compile_hash(seq.hash_comp, chunk_scratch)

        # Phase 4: Index
        self._compile_index(seq.index_comp, chunk_scratch)

        # Phase 5: Store
        self._compile_store(seq.store, chunk_scratch, batch_indices)

    def _compile_gather(self, gather: GatherNode, chunk_scratch):
        """Compile gather phase based on strategy"""
        n_chunks = len(chunk_scratch)

        if gather.strategy == GatherStrategy.SEQUENTIAL:
            for vi in range(VLEN):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                self.flush_schedule()
                for c, (p_idx, _, p_node_val, _, _, p_addr1, _) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    if (c + 1) % 2 == 0:
                        self.flush_schedule()
                if n_chunks % 2 != 0:
                    self.flush_schedule()

        elif gather.strategy == GatherStrategy.DUAL_ADDR:
            for vi in range(0, VLEN, 2):
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("alu", ("+", p_addr1, self.scratch["forest_values_p"], p_idx + vi))
                    self.emit("alu", ("+", p_addr2, self.scratch["forest_values_p"], p_idx + vi + 1))
                self.flush_schedule()
                for c, (p_idx, _, p_node_val, _, _, p_addr1, p_addr2) in enumerate(chunk_scratch):
                    self.emit("load", ("load", p_node_val + vi, p_addr1))
                    self.emit("load", ("load", p_node_val + vi + 1, p_addr2))
                    self.flush_schedule()

        elif gather.strategy in (GatherStrategy.BATCH_ADDR, GatherStrategy.PIPELINED):
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

    def _compile_hash(self, hash_node: HashNode, chunk_scratch):
        """Compile hash phase based on strategy"""
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
        """Compile index computation based on strategy"""
        has_preloaded = index_node.use_preloaded_consts and 2 in self.vec_const_map
        has_n_nodes = "n_nodes" in self.vec_const_map

        if has_preloaded:
            vc_zero = self.vec_const_map[0]
            vc_one = self.vec_const_map[1]
            vc_two = self.vec_const_map[2]
        else:
            vc_zero = vc_one = vc_two = None

        if index_node.strategy == IndexStrategy.VSELECT and has_preloaded:
            # val % 2 and 2*idx
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two))
                self.emit("valu", ("*", p_idx, p_idx, vc_two))
            self.flush_schedule()
            # == 0
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
            self.flush_schedule()
            # vselect
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("flow", ("vselect", p_tmp1, p_tmp2, vc_one, vc_two))
            self.flush_schedule()
            # add
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
            self.flush_schedule()

        elif index_node.strategy == IndexStrategy.ARITHMETIC and has_preloaded:
            # val % 2
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two))
            self.flush_schedule()
            # == 0
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
            self.flush_schedule()
            # offset = 2 - cond
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))
            self.flush_schedule()
            # 2*idx
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("*", p_idx, p_idx, vc_two))
            self.flush_schedule()
            # add offset
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("+", p_idx, p_idx, p_tmp1))
            self.flush_schedule()

        elif index_node.strategy == IndexStrategy.MULTIPLY_ADD and has_preloaded:
            # val % 2
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("%", p_tmp2, p_val, vc_two))
            self.flush_schedule()
            # == 0
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("==", p_tmp2, p_tmp2, vc_zero))
            self.flush_schedule()
            # offset = 2 - cond
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("-", p_tmp1, vc_two, p_tmp2))
            self.flush_schedule()
            # multiply_add: idx = idx*2 + offset
            for c, (p_idx, p_val, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
                self.emit("valu", ("multiply_add", p_idx, p_idx, vc_two, p_tmp1))
            self.flush_schedule()

        else:
            # Fallback to VSELECT with broadcasts
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

        # Bounds check
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

        # idx = idx * inbounds (arithmetic bounds)
        for c, (p_idx, _, _, p_tmp1, p_tmp2, _, _) in enumerate(chunk_scratch):
            self.emit("valu", ("*", p_idx, p_idx, p_tmp2))
        self.flush_schedule()

    def _compile_store(self, store: StoreNode, chunk_scratch, batch_indices):
        """Compile store phase"""
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
    """Evaluate a GP tree by compiling and running it"""
    try:
        random.seed(TEST_SEED)
        forest = Tree.generate(TEST_FOREST_HEIGHT)
        inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
        mem = build_mem_image(forest, inp)

        builder = GPKernelBuilder(program)
        instrs = builder.build(forest.height, len(forest.values),
                               len(inp.indices), TEST_ROUNDS)

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
    except Exception as e:
        return float('inf')


# =============================================================================
# GP Algorithm
# =============================================================================

class GeneticProgramming:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.7,
        point_mutation_rate: float = 0.2,
        subtree_mutation_rate: float = 0.1,
        elitism: int = 3,
        tournament_size: int = 3,
        max_depth: int = 10,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.point_mutation_rate = point_mutation_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.max_depth = max_depth

        self.population: List[ProgramNode] = []
        self.fitnesses: List[float] = []
        self.best_program: Optional[ProgramNode] = None
        self.best_fitness: float = float('inf')
        self.history: List[dict] = []

    def initialize_population(self):
        """Create initial population with seeded diversity"""
        self.population = []

        # Seed with different strategies
        for seed_type in ["baseline", "minimal_flush", "max_parallel", "conservative"]:
            self.population.append(seeded_program(seed_type))

        # Fill with random
        while len(self.population) < self.population_size:
            self.population.append(random_program())

    def tournament_select(self) -> ProgramNode:
        """Tournament selection"""
        indices = random.sample(range(len(self.population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: self.fitnesses[i])
        return self.population[best_idx].clone()

    def evaluate_population(self):
        """Evaluate all programs"""
        timeout = int(self.best_fitness * 1.5) if self.best_fitness < float('inf') else 50000

        self.fitnesses = [evaluate_program(p, timeout) for p in self.population]

        for i, fitness in enumerate(self.fitnesses):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_program = self.population[i].clone()

    def evolve_generation(self):
        """Create next generation using GP operators"""
        new_population = []

        # Elitism
        sorted_indices = sorted(range(len(self.fitnesses)), key=lambda i: self.fitnesses[i])
        for i in range(self.elitism):
            if self.fitnesses[sorted_indices[i]] < float('inf'):
                new_population.append(self.population[sorted_indices[i]].clone())

        # Fill with offspring
        while len(new_population) < self.population_size:
            op = random.random()

            if op < self.crossover_rate:
                # Subtree crossover
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                child = subtree_crossover(p1, p2)

            elif op < self.crossover_rate + self.subtree_mutation_rate:
                # Subtree mutation
                parent = self.tournament_select()
                child = subtree_mutation(parent)

            else:
                # Point mutation
                parent = self.tournament_select()
                child = point_mutation(parent, self.point_mutation_rate)

            # Depth limiting
            if child.depth() <= self.max_depth:
                new_population.append(child)
            else:
                new_population.append(self.tournament_select())

        self.population = new_population

    def run(self, verbose: bool = True) -> Tuple[ProgramNode, float]:
        """Run the GP algorithm"""
        print(f"Starting GP: {self.population_size} individuals, {self.generations} generations")
        print()

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

        return self.best_program, self.best_fitness

    def save_results(self, filename: str = "gp_results.json"):
        """Save results to file"""
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
                    'body': serialize_node(node.body)
                }
            elif isinstance(node, PhaseSequenceNode):
                return {
                    'type': 'PhaseSequence',
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
                    'flush_per_element': node.flush_per_element
                }
            elif isinstance(node, HashNode):
                return {
                    'type': 'Hash',
                    'strategy': node.strategy.value,
                    'flush_per_stage': node.flush_per_stage,
                    'use_preloaded_consts': node.use_preloaded_consts
                }
            elif isinstance(node, IndexNode):
                return {
                    'type': 'Index',
                    'strategy': node.strategy.value,
                    'flush_per_op': node.flush_per_op,
                    'use_preloaded_consts': node.use_preloaded_consts
                }
            elif isinstance(node, StoreNode):
                return {
                    'type': 'Store',
                    'flush_after_addr': node.flush_after_addr
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
    parser = argparse.ArgumentParser(description='GP optimizer for VLIW kernel')
    parser.add_argument('--generations', '-g', type=int, default=30)
    parser.add_argument('--population', '-p', type=int, default=40)
    parser.add_argument('--output', '-o', type=str, default='gp_results.json')

    args = parser.parse_args()

    gp = GeneticProgramming(
        population_size=args.population,
        generations=args.generations,
    )

    best_program, best_fitness = gp.run()
    gp.save_results(args.output)

    print(f"\nSpeedup over baseline: {147734 / best_fitness:.2f}x")


if __name__ == "__main__":
    main()
