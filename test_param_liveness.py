"""
Test script to determine which node parameters actually affect code generation.
Tests MemoryNode, LoopNode, PipelineNode, InterleaveNode, and RegisterNode parameters.
"""

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Any

# Import from gp_optimizer_v3
from gp_optimizer_v3 import (
    ProgramNode, SetupNode, LoopNode, PhaseSequenceNode,
    GatherNode, HashNode, IndexNode, StoreNode,
    PipelineNode, InterleaveNode, MemoryNode, RegisterNode,
    LoopStructure, GatherStrategy, HashStrategy, IndexStrategy,
    InterleaveStrategy, AddressGenStrategy, AccessOrder,
    PipelineSchedule, AllocationStrategy, ReusePolicy,
    GPKernelBuilderV3, GPNode, FusionMode
)

def create_base_program(parallel_chunks=8, loop_order="round_first", skip_indices=False, outer_unroll=1):
    """Create a minimal valid program for testing."""
    setup = SetupNode(
        preload_tree_depth=1,
        preload_hash_consts=False,
        init_pointers=True,
        buffer_alignment=64,
        const_pool_size=8
    )
    
    gather = GatherNode(
        strategy=GatherStrategy.BATCH_ADDR,
        inner_unroll=2,
        max_addr_regs=4,
        prefetch=False,
        flush_after_addr=True,
        use_tree_cache=False
    )
    
    hash_node = HashNode(
        strategy=HashStrategy.FUSED,
        use_preloaded_consts=False,
        stage_unroll=[1, 1, 1],
        cross_chunk_interleave=False,
        fuse_xor_with_stage1=False
    )
    
    index = IndexNode(
        strategy=IndexStrategy.MULTIPLY_ADD,
        use_shift=True,
        unroll_factor=1,
        combine_with_hash=False
    )
    
    store = StoreNode(
        store_order="idx_first",
        batch_stores=1,
        write_combining=False,
        flush_after_addr=True
    )
    
    body = PhaseSequenceNode(
        gather=gather,
        hash_comp=hash_node,
        index_comp=index,
        store=store,
        phase_order=["gather", "hash", "index", "store"],
        fusion_mode=FusionMode.NONE
    )
    
    pipeline = PipelineNode(
        enabled=False,
        pipeline_depth=2,
        schedule=PipelineSchedule.NONE,
        prologue_unroll=1,
        epilogue_drain=True
    )
    
    interleave = InterleaveNode(
        strategy=InterleaveStrategy.NONE,
        lookahead_depth=2,
        min_slot_fill=0.5,
        allow_cross_chunk=False
    )
    
    memory = MemoryNode(
        prefetch_distance=0,
        store_buffer_depth=0,
        address_gen=AddressGenStrategy.LAZY,
        access_order=AccessOrder.SEQUENTIAL,
        coalesce_loads=False,
        coalesce_stores=False
    )
    
    registers = RegisterNode(
        allocation=AllocationStrategy.DENSE,
        reuse_policy=ReusePolicy.AGGRESSIVE,
        spill_threshold=512,
        vector_alignment=8,
        reserve_temps=4
    )
    
    loop = LoopNode(
        structure=LoopStructure.CHUNKED,
        parallel_chunks=parallel_chunks,
        body=body,
        pipeline=pipeline,
        interleave=interleave,
        memory=memory,
        registers=registers,
        outer_unroll=outer_unroll,
        chunk_unroll=1,
        loop_order=loop_order,
        skip_indices=skip_indices
    )
    
    return ProgramNode(setup=setup, main_loop=loop)

def generate_code(program: ProgramNode) -> str:
    """Generate code from a program and return as string."""
    try:
        builder = GPKernelBuilderV3(program)
        instrs = builder.compile()
        # Convert code to a stable string representation
        lines = []
        for instr in instrs:
            instr_str = str(sorted([(k, str(v)) for k, v in instr.items()]))
            lines.append(instr_str)
        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n{traceback.format_exc()}"

def test_parameter(param_name: str, values: List[Any], modify_func) -> Tuple[str, bool, str]:
    """
    Test if a parameter affects code generation.
    
    Args:
        param_name: Name of the parameter
        values: List of values to test
        modify_func: Function that takes (program, value) and modifies the program
        
    Returns:
        Tuple of (param_name, is_live, details)
    """
    codes = {}
    for val in values:
        program = create_base_program()
        modify_func(program, val)
        code = generate_code(program)
        codes[str(val)] = code
    
    # Check if all codes are the same
    unique_codes = set(codes.values())
    is_live = len(unique_codes) > 1
    
    if is_live:
        # Find which values produce different code
        val_to_code = {}
        for val_str, code in codes.items():
            code_hash = hash(code)
            if code_hash not in val_to_code:
                val_to_code[code_hash] = []
            val_to_code[code_hash].append(val_str)
        details = f"{len(unique_codes)} unique outputs"
    else:
        # Check if there was an error
        first_code = list(codes.values())[0]
        if first_code.startswith("ERROR:"):
            details = f"ERROR during generation: {first_code[:100]}"
            return (param_name, None, details)
        details = "all values produce identical code"
    
    return (param_name, is_live, details)

def main():
    print("=" * 70)
    print("PARAMETER LIVENESS TEST")
    print("Testing which parameters affect code generation in gp_optimizer_v3.py")
    print("=" * 70)
    print()
    
    results = []
    
    # ==========================================================================
    # MEMORY NODE TESTS
    # ==========================================================================
    print("Testing MemoryNode parameters...")
    
    # address_gen
    result = test_parameter(
        "MemoryNode.address_gen",
        [AddressGenStrategy.EAGER, AddressGenStrategy.LAZY],
        lambda p, v: setattr(p.main_loop.memory, 'address_gen', v)
    )
    results.append(result)
    
    # prefetch_distance
    result = test_parameter(
        "MemoryNode.prefetch_distance",
        [0, 2],
        lambda p, v: setattr(p.main_loop.memory, 'prefetch_distance', v)
    )
    results.append(result)
    
    # store_buffer_depth
    result = test_parameter(
        "MemoryNode.store_buffer_depth",
        [0, 2],
        lambda p, v: setattr(p.main_loop.memory, 'store_buffer_depth', v)
    )
    results.append(result)
    
    # access_order
    result = test_parameter(
        "MemoryNode.access_order",
        [AccessOrder.SEQUENTIAL, AccessOrder.STRIDED, AccessOrder.REVERSED],
        lambda p, v: setattr(p.main_loop.memory, 'access_order', v)
    )
    results.append(result)
    
    # coalesce_loads
    result = test_parameter(
        "MemoryNode.coalesce_loads",
        [True, False],
        lambda p, v: setattr(p.main_loop.memory, 'coalesce_loads', v)
    )
    results.append(result)
    
    # coalesce_stores
    result = test_parameter(
        "MemoryNode.coalesce_stores",
        [True, False],
        lambda p, v: setattr(p.main_loop.memory, 'coalesce_stores', v)
    )
    results.append(result)
    
    # ==========================================================================
    # LOOP NODE TESTS
    # ==========================================================================
    print("Testing LoopNode parameters...")
    
    # structure
    result = test_parameter(
        "LoopNode.structure",
        [LoopStructure.CHUNKED, LoopStructure.SIMPLE, LoopStructure.TILED],
        lambda p, v: setattr(p.main_loop, 'structure', v)
    )
    results.append(result)
    
    # parallel_chunks
    result = test_parameter(
        "LoopNode.parallel_chunks",
        [4, 8, 16],
        lambda p, v: setattr(p.main_loop, 'parallel_chunks', v)
    )
    results.append(result)
    
    # outer_unroll
    result = test_parameter(
        "LoopNode.outer_unroll",
        [1, 2, 4],
        lambda p, v: setattr(p.main_loop, 'outer_unroll', v)
    )
    results.append(result)
    
    # skip_indices (needs chunk_first)
    def set_skip_indices(p, v):
        p.main_loop.loop_order = "chunk_first"
        p.main_loop.skip_indices = v
    result = test_parameter(
        "LoopNode.skip_indices",
        [True, False],
        set_skip_indices
    )
    results.append(result)
    
    # loop_order
    result = test_parameter(
        "LoopNode.loop_order",
        ["chunk_first", "round_first"],
        lambda p, v: setattr(p.main_loop, 'loop_order', v)
    )
    results.append(result)
    
    # ==========================================================================
    # PIPELINE NODE TESTS
    # ==========================================================================
    print("Testing PipelineNode parameters...")
    
    # enabled
    result = test_parameter(
        "PipelineNode.enabled",
        [True, False],
        lambda p, v: setattr(p.main_loop.pipeline, 'enabled', v)
    )
    results.append(result)
    
    # ==========================================================================
    # INTERLEAVE NODE TESTS
    # ==========================================================================
    print("Testing InterleaveNode parameters...")
    
    # strategy
    result = test_parameter(
        "InterleaveNode.strategy",
        [InterleaveStrategy.NONE, InterleaveStrategy.ALL_PHASES],
        lambda p, v: setattr(p.main_loop.interleave, 'strategy', v)
    )
    results.append(result)
    
    # lookahead_depth
    result = test_parameter(
        "InterleaveNode.lookahead_depth",
        [2, 8],
        lambda p, v: setattr(p.main_loop.interleave, 'lookahead_depth', v)
    )
    results.append(result)
    
    # ==========================================================================
    # REGISTER NODE TESTS
    # ==========================================================================
    print("Testing RegisterNode parameters...")
    
    # allocation
    result = test_parameter(
        "RegisterNode.allocation",
        [AllocationStrategy.DENSE, AllocationStrategy.SPARSE],
        lambda p, v: setattr(p.main_loop.registers, 'allocation', v)
    )
    results.append(result)
    
    # reuse_policy
    result = test_parameter(
        "RegisterNode.reuse_policy",
        [ReusePolicy.AGGRESSIVE, ReusePolicy.CONSERVATIVE],
        lambda p, v: setattr(p.main_loop.registers, 'reuse_policy', v)
    )
    results.append(result)
    
    # spill_threshold
    result = test_parameter(
        "RegisterNode.spill_threshold",
        [256, 512],
        lambda p, v: setattr(p.main_loop.registers, 'spill_threshold', v)
    )
    results.append(result)
    
    # vector_alignment
    result = test_parameter(
        "RegisterNode.vector_alignment",
        [8, 16],
        lambda p, v: setattr(p.main_loop.registers, 'vector_alignment', v)
    )
    results.append(result)
    
    # reserve_temps
    result = test_parameter(
        "RegisterNode.reserve_temps",
        [4, 8],
        lambda p, v: setattr(p.main_loop.registers, 'reserve_temps', v)
    )
    results.append(result)
    
    # Test max_live_vectors if it exists
    if hasattr(RegisterNode, 'max_live_vectors'):
        result = test_parameter(
            "RegisterNode.max_live_vectors",
            [16, 32],
            lambda p, v: setattr(p.main_loop.registers, 'max_live_vectors', v)
        )
        results.append(result)
    else:
        results.append(("RegisterNode.max_live_vectors", None, "PARAMETER NOT FOUND"))
    
    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Parameter':<45} {'Status':<10} {'Details'}")
    print("-" * 70)
    
    live_count = 0
    dead_count = 0
    
    for param_name, is_live, details in results:
        if is_live is None:
            status = "N/A"
        elif is_live:
            status = "LIVE"
            live_count += 1
        else:
            status = "DEAD"
            dead_count += 1
        print(f"{param_name:<45} {status:<10} {details}")
    
    print()
    print("-" * 70)
    print(f"LIVE parameters: {live_count}")
    print(f"DEAD parameters: {dead_count}")
    print()
    
    # Print categorized results
    print("=" * 70)
    print("CATEGORIZED RESULTS")
    print("=" * 70)
    
    print("\nLIVE PARAMETERS (affect code generation):")
    for param_name, is_live, details in results:
        if is_live:
            print(f"  - {param_name}: {details}")
    
    print("\nDEAD PARAMETERS (no effect on code generation):")
    for param_name, is_live, details in results:
        if is_live is False:
            print(f"  - {param_name}")
    
    print("\nN/A PARAMETERS (not found or error):")
    for param_name, is_live, details in results:
        if is_live is None:
            print(f"  - {param_name}: {details}")

if __name__ == "__main__":
    main()
