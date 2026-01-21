"""
Test which IndexNode and StoreNode parameters actually affect code generation.
Uses isolated test pattern where ONLY the tested parameter changes between variants.
Version 2: More detailed analysis including instruction count and order.
"""

import sys
import random
from copy import deepcopy

# Import the necessary classes from gp_optimizer_v3
from gp_optimizer_v3 import (
    GPKernelBuilderV3, ProgramNode, LoopNode, SetupNode, PhaseSequenceNode,
    GatherNode, HashNode, IndexNode, StoreNode, PipelineNode, InterleaveNode,
    MemoryNode, RegisterNode, LoopStructure, GatherStrategy, HashStrategy,
    IndexStrategy, InterleaveStrategy, PipelineSchedule
)
from problem import Tree, Input, build_mem_image

# Test configuration
TEST_FOREST_HEIGHT = 10
TEST_ROUNDS = 16
TEST_BATCH_SIZE = 256
TEST_SEED = 123

def create_base_program() -> ProgramNode:
    """Create a known-good baseline program for testing."""
    setup = SetupNode(
        preload_scalars=True,
        preload_vectors=True,
        preload_hash_consts=True,
        preload_n_nodes=True,
        tree_cache_depth=0
    )
    
    gather = GatherNode(
        strategy=GatherStrategy.BATCH_ADDR,
        flush_after_addr=False,
        flush_per_element=False,
        inner_unroll=1,
        vector_grouping=1,
        addr_compute_ahead=0,
        max_addr_regs=2,
        use_tree_cache=False
    )
    
    hash_comp = HashNode(
        strategy=HashStrategy.FUSED,
        flush_per_stage=False,
        use_preloaded_consts=True,
        stage_unroll=(1, 1, 1, 1),
        fuse_xor_with_stage1=False,
        cross_chunk_interleave=False
    )
    
    # Base IndexNode - we'll vary parameters from this
    index_comp = IndexNode(
        strategy=IndexStrategy.MULTIPLY_ADD,
        flush_per_op=False,
        use_preloaded_consts=True,
        compute_unroll=1,
        bounds_check_mode="multiply",
        speculative=False,
        index_formula="original"
    )
    
    # Base StoreNode - we'll vary parameters from this
    store = StoreNode(
        flush_after_addr=False,
        batch_stores=1,
        store_order="idx_first",
        write_combining=False
    )
    
    body = PhaseSequenceNode(
        gather=gather,
        hash_comp=hash_comp,
        index_comp=index_comp,
        store=store
    )
    
    pipeline = PipelineNode(enabled=False)
    interleave = InterleaveNode(strategy=InterleaveStrategy.NONE)
    memory = MemoryNode()
    registers = RegisterNode()
    
    loop = LoopNode(
        structure=LoopStructure.CHUNKED,
        parallel_chunks=4,
        body=body,
        pipeline=pipeline,
        interleave=interleave,
        memory=memory,
        registers=registers,
        outer_unroll=1,
        chunk_unroll=1
    )
    
    return ProgramNode(setup=setup, main_loop=loop)


def generate_code_and_stats(program: ProgramNode) -> tuple:
    """Generate code from a program and return (code_string, cycle_count, instruction_count)."""
    random.seed(TEST_SEED)
    forest = Tree.generate(TEST_FOREST_HEIGHT)
    inp = Input.generate(forest, TEST_BATCH_SIZE, TEST_ROUNDS)
    mem = build_mem_image(forest, inp)
    
    builder = GPKernelBuilderV3(program)
    instrs = builder.build(forest.height, len(forest.values),
                           len(inp.indices), TEST_ROUNDS)
    
    # Count cycles and total instructions
    cycle_count = len(instrs)
    total_instrs = sum(len(cycle) for cycle in instrs)
    
    # Convert instructions to a canonical string representation preserving order
    code_lines = []
    for i, cycle_instrs in enumerate(instrs):
        for instr in cycle_instrs:
            code_lines.append(f"C{i:04d}: {str(instr)}")
    
    return "\n".join(code_lines), cycle_count, total_instrs


def compare_codes(code1: str, code2: str) -> dict:
    """Detailed comparison of two code strings."""
    lines1 = code1.split('\n')
    lines2 = code2.split('\n')
    
    # Check if identical
    identical = code1 == code2
    
    # Count line differences
    line_diffs = 0
    for l1, l2 in zip(lines1, lines2):
        if l1 != l2:
            line_diffs += 1
    line_diffs += abs(len(lines1) - len(lines2))
    
    # Check instruction set difference (unordered)
    set1 = set(lines1)
    set2 = set(lines2)
    symmetric_diff = len(set1.symmetric_difference(set2))
    
    return {
        "identical": identical,
        "line_count_diff": len(lines2) - len(lines1),
        "positional_diffs": line_diffs,
        "set_diffs": symmetric_diff
    }


def test_index_parameter(param_name: str, values: list) -> dict:
    """Test if a specific IndexNode parameter affects code generation."""
    codes = {}
    stats = {}
    
    for value in values:
        program = create_base_program()
        setattr(program.main_loop.body.index_comp, param_name, value)
        code, cycles, instrs = generate_code_and_stats(program)
        key = str(value)
        codes[key] = code
        stats[key] = {"cycles": cycles, "instructions": instrs}
    
    # Check if codes are actually identical
    unique_codes = len(set(codes.values()))
    is_live = unique_codes > 1
    
    # Detailed pairwise comparison
    comparisons = {}
    value_strs = [str(v) for v in values]
    for i, v1 in enumerate(value_strs):
        for v2 in value_strs[i+1:]:
            comp = compare_codes(codes[v1], codes[v2])
            comparisons[f"{v1} vs {v2}"] = comp
    
    return {
        "parameter": param_name,
        "values_tested": value_strs,
        "is_live": is_live,
        "unique_code_count": unique_codes,
        "stats": stats,
        "comparisons": comparisons
    }


def test_store_parameter(param_name: str, values: list) -> dict:
    """Test if a specific StoreNode parameter affects code generation."""
    codes = {}
    stats = {}
    
    for value in values:
        program = create_base_program()
        setattr(program.main_loop.body.store, param_name, value)
        code, cycles, instrs = generate_code_and_stats(program)
        key = str(value)
        codes[key] = code
        stats[key] = {"cycles": cycles, "instructions": instrs}
    
    # Check if codes are actually identical
    unique_codes = len(set(codes.values()))
    is_live = unique_codes > 1
    
    # Detailed pairwise comparison
    comparisons = {}
    value_strs = [str(v) for v in values]
    for i, v1 in enumerate(value_strs):
        for v2 in value_strs[i+1:]:
            comp = compare_codes(codes[v1], codes[v2])
            comparisons[f"{v1} vs {v2}"] = comp
    
    return {
        "parameter": param_name,
        "values_tested": value_strs,
        "is_live": is_live,
        "unique_code_count": unique_codes,
        "stats": stats,
        "comparisons": comparisons
    }


def main():
    print("=" * 70)
    print("INDEXNODE AND STORENODE PARAMETER LIVENESS TEST (V2)")
    print("=" * 70)
    print()
    
    # ===== IndexNode Tests =====
    print("=" * 70)
    print("INDEXNODE PARAMETER TESTS")
    print("=" * 70)
    print()
    
    index_tests = [
        ("strategy", [IndexStrategy.VSELECT, IndexStrategy.ARITHMETIC, IndexStrategy.MULTIPLY_ADD]),
        ("flush_per_op", [True, False]),
        ("use_preloaded_consts", [True, False]),
        ("compute_unroll", [1, 2]),
        ("speculative", [True, False]),
    ]
    
    index_results = []
    for param_name, values in index_tests:
        print(f"Testing IndexNode.{param_name}...")
        try:
            result = test_index_parameter(param_name, values)
            index_results.append(result)
            status = "LIVE" if result["is_live"] else "DEAD"
            print(f"  {param_name}: {status}")
            print(f"    Values: {result['values_tested']}")
            print(f"    Unique codes: {result['unique_code_count']}")
            print(f"    Stats per value:")
            for val, st in result["stats"].items():
                print(f"      {val}: {st['cycles']} cycles, {st['instructions']} instrs")
            print(f"    Comparisons:")
            for pair, comp in result["comparisons"].items():
                if comp["identical"]:
                    print(f"      {pair}: IDENTICAL")
                else:
                    print(f"      {pair}: DIFFERS - {comp['positional_diffs']} positional diffs, {comp['set_diffs']} set diffs")
            print()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            index_results.append({
                "parameter": param_name,
                "error": str(e),
                "is_live": None
            })
            print()
    
    # ===== StoreNode Tests =====
    print("=" * 70)
    print("STORENODE PARAMETER TESTS")
    print("=" * 70)
    print()
    
    store_tests = [
        ("flush_after_addr", [True, False]),
        ("store_order", ["idx_first", "val_first", "interleaved"]),
        ("batch_stores", [1, 4]),
        ("write_combining", [True, False]),
    ]
    
    store_results = []
    for param_name, values in store_tests:
        print(f"Testing StoreNode.{param_name}...")
        try:
            result = test_store_parameter(param_name, values)
            store_results.append(result)
            status = "LIVE" if result["is_live"] else "DEAD"
            print(f"  {param_name}: {status}")
            print(f"    Values: {result['values_tested']}")
            print(f"    Unique codes: {result['unique_code_count']}")
            print(f"    Stats per value:")
            for val, st in result["stats"].items():
                print(f"      {val}: {st['cycles']} cycles, {st['instructions']} instrs")
            print(f"    Comparisons:")
            for pair, comp in result["comparisons"].items():
                if comp["identical"]:
                    print(f"      {pair}: IDENTICAL")
                else:
                    print(f"      {pair}: DIFFERS - {comp['positional_diffs']} positional diffs, {comp['set_diffs']} set diffs")
            print()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            store_results.append({
                "parameter": param_name,
                "error": str(e),
                "is_live": None
            })
            print()
    
    # ===== Summary =====
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("INDEXNODE PARAMETERS:")
    live_index = [r["parameter"] for r in index_results if r.get("is_live") is True]
    dead_index = [r["parameter"] for r in index_results if r.get("is_live") is False]
    error_index = [r["parameter"] for r in index_results if r.get("is_live") is None]
    
    print(f"  LIVE ({len(live_index)}): {', '.join(live_index) if live_index else 'None'}")
    print(f"  DEAD ({len(dead_index)}): {', '.join(dead_index) if dead_index else 'None'}")
    if error_index:
        print(f"  ERROR ({len(error_index)}): {', '.join(error_index)}")
    print()
    
    print("STORENODE PARAMETERS:")
    live_store = [r["parameter"] for r in store_results if r.get("is_live") is True]
    dead_store = [r["parameter"] for r in store_results if r.get("is_live") is False]
    error_store = [r["parameter"] for r in store_results if r.get("is_live") is None]
    
    print(f"  LIVE ({len(live_store)}): {', '.join(live_store) if live_store else 'None'}")
    print(f"  DEAD ({len(dead_store)}): {', '.join(dead_store) if dead_store else 'None'}")
    if error_store:
        print(f"  ERROR ({len(error_store)}): {', '.join(error_store)}")


if __name__ == "__main__":
    main()
