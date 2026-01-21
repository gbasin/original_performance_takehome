# GP Search Space Analysis

*Last updated: January 2026 - Based on empirical testing of each parameter*

## Parameter Status Summary

### Legend
- **LIVE**: Parameter affects generated code (verified by instruction count changes)
- **DEAD**: Parameter declared but never read in code generation
- **CONDITIONAL**: Only affects code under specific conditions

---

## 1. LIVE Parameters (Actually Work)

### GatherNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `strategy` | SEQUENTIAL, DUAL_ADDR, BATCH_ADDR, PIPELINED, VECTORIZED | LIVE | BATCH_ADDR saves ~256 instrs vs SEQUENTIAL |
| `flush_after_addr` | bool | LIVE | False saves ~256 instrs |
| `inner_unroll` | [1, 2, 4] | LIVE | 1→2 saves ~256 instrs; 2→4 no change unless max_addr_regs=4 |
| `max_addr_regs` | [2, 4] | CONDITIONAL | Only matters when inner_unroll > 2; saves ~128 instrs with unroll=4 |

### HashNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `flush_per_stage` | bool | LIVE | Adds flush between hash stages |
| `use_preloaded_consts` | bool | LIVE | False adds inline vbroadcast ops |
| `fuse_xor_with_stage1` | bool | CONDITIONAL | Only works when fusion_mode=GATHER_XOR |
| `stage_unroll` | tuple[int, ...] | LIVE | Controls flush granularity between stages |
| `cross_chunk_interleave` | bool | LIVE | Interleaves ops across chunks for better ILP |

### IndexNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `strategy` | VSELECT, ARITHMETIC, MULTIPLY_ADD, BRANCHLESS | LIVE | MULTIPLY_ADD typically best (~500 cycle difference) |
| `use_preloaded_consts` | bool | LIVE | True saves ~384 cycles |
| `speculative` | bool | LIVE | Computes both branches, uses vselect |

### StoreNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `flush_after_addr` | bool | LIVE | False saves ~384 cycles |
| `write_combining` | bool | LIVE | Adds flushes after store batches |
| `batch_stores` | [1, 2, 4, 8] | CONDITIONAL | Only affects code when write_combining=True |
| `store_order` | idx_first, val_first, interleaved | CONDITIONAL | Only matters with write_combining + batch_stores>=2 |

### MemoryNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `access_order` | SEQUENTIAL, STRIDED, BLOCKED, REVERSED | LIVE | Changes gather element processing order |
| `coalesce_loads` | bool | LIVE | Groups all loads together before flush |

### LoopNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `parallel_chunks` | [4, 8, 16] | LIVE | All produce different instruction counts |
| `loop_order` | chunk_first, round_first | LIVE | Different iteration patterns |
| `skip_indices` | bool | CONDITIONAL | Only works with loop_order=chunk_first |

### SetupNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `preload_scalars` | bool | LIVE | Preloads common scalar constants |
| `preload_vectors` | bool | LIVE | Preloads vector constants (0, 1, 2) |
| `preload_hash_consts` | bool | LIVE | Preloads hash stage constants |

### PhaseSequenceNode

| Parameter | Values | Status | Notes |
|-----------|--------|--------|-------|
| `phase_order` | [gather,hash,index,store] or [gather,index,hash,store] | LIVE | Only 2 orderings implemented |
| `fusion_mode` | NONE, GATHER_XOR, ... | LIVE | Only GATHER_XOR actually implemented |

---

## 2. DEAD Parameters (15 Total)

### GatherNode (3 dead)

| Parameter | Why Dead |
|-----------|----------|
| `flush_per_element` | Declared but never read in `_compile_gather()` |
| `vector_grouping` | Declared but never read in `_compile_gather()` |
| `addr_compute_ahead` | Declared but never read in `_compile_gather()` |

### HashNode (1 dead)

| Parameter | Why Dead |
|-----------|----------|
| `strategy` | The enum (STANDARD, FUSED, INTERLEAVED, UNROLLED) is **never checked** in `_compile_hash()` - all paths use same logic |

### IndexNode (2 dead)

| Parameter | Why Dead |
|-----------|----------|
| `flush_per_op` | Declared but never read |
| `compute_unroll` | Declared but never read |

### MemoryNode (4 dead)

| Parameter | Why Dead |
|-----------|----------|
| `address_gen` | EAGER/LAZY/SPECULATIVE enum never checked |
| `prefetch_distance` | Declared but never read |
| `store_buffer_depth` | Read but doesn't change behavior (flush logic ineffective) |
| `coalesce_stores` | Read but overridden - only affects `effective_store_order` which itself is often dead |

### LoopNode (2 dead)

| Parameter | Why Dead |
|-----------|----------|
| `structure` | Only CHUNKED path implemented in `_compile_loop()` |
| `outer_unroll` | Declared but never affects code generation |

### PipelineNode (entire node dead)

| Parameter | Why Dead |
|-----------|----------|
| `enabled` | Never checked - no software pipelining implemented |
| `stages` | Never read |
| `initiation_interval` | Never read |
| `pipeline_depth` | Never read |
| `schedule` | Never read |

### InterleaveNode (2 dead)

| Parameter | Why Dead |
|-----------|----------|
| `lookahead_depth` | Never read in scheduling logic |
| `min_slot_fill` | Never read in scheduling logic |
| `strategy` | Enum exists but only basic buffering works, no true interleaving |

### RegisterNode (entire node dead)

| Parameter | Why Dead |
|-----------|----------|
| `allocation` | Never read - uses fixed sequential allocation |
| `reuse_policy` | Never read |
| `spill_threshold` | Never read |
| `vector_alignment` | Never read |
| `reserve_temps` | Never read |
| `max_live_vectors` | Never read |

### SetupNode (1 dead)

| Parameter | Why Dead |
|-----------|----------|
| `init_scratch` | Declared but never read |

---

## 3. Parameter Value Ranges

### Current vs Potential Expansion

| Parameter | Current | Could Be | Constraint |
|-----------|---------|----------|------------|
| `parallel_chunks` | [4, 8, 16] | [4, 8, 16, 32] | 32 chunks uses 90% of scratch memory |
| `inner_unroll` | [1, 2, 4] | [1, 2, 4, 8] | 8 = full VLEN, needs max_addr_regs=8 |
| `max_addr_regs` | [2, 4] | [2, 4, 8] | Limited by scratch space per chunk |
| `batch_stores` | [1, 2, 4] | [1, 2, 4, 8, 16] | Should match parallel_chunks |
| `stage_unroll` | each [1, 2] | each [1, 2, 4] | More aggressive unrolling |
| `outer_unroll` | [1, 2, 4] | Would need implementation | Currently dead |

### Hardware Constraints

```
VLEN = 8              # Vector length
SCRATCH_SIZE = 1536   # Total scratch memory (words)
SLOT_LIMITS = {
    'alu': 12,        # ALU ops per cycle
    'valu': 6,        # Vector ops per cycle
    'load': 2,        # Loads per cycle (main bottleneck)
    'store': 2,       # Stores per cycle
    'flow': 1,        # Control ops per cycle
}
```

### Scratch Usage Analysis

| Chunks | Addr Regs | Scratch Used | % of 1536 |
|--------|-----------|--------------|-----------|
| 8 | 2 | 386 | 25% |
| 8 | 4 | 402 | 26% |
| 16 | 2 | 722 | 47% |
| 16 | 4 | 754 | 49% |
| 32 | 2 | 1394 | 91% |
| 32 | 4 | 1458 | 95% |

---

## 4. Effective Search Space

### What's Actually Evolved (Affects Output)

| Category | Options | Count |
|----------|---------|-------|
| Loop parallel_chunks | 4, 8, 16 | 3 |
| Loop loop_order | chunk_first, round_first | 2 |
| Loop skip_indices | true, false | 2 |
| Gather strategy | 5 strategies | 5 |
| Gather flush_after_addr | bool | 2 |
| Gather inner_unroll | 1, 2, 4 | 3 |
| Gather max_addr_regs | 2, 4 | 2 |
| Hash flush_per_stage | bool | 2 |
| Hash use_preloaded_consts | bool | 2 |
| Hash fuse_xor_with_stage1 | bool | 2 |
| Hash stage_unroll | 2^4 combinations | 16 |
| Hash cross_chunk_interleave | bool | 2 |
| Index strategy | 4 strategies | 4 |
| Index use_preloaded_consts | bool | 2 |
| Index speculative | bool | 2 |
| Store flush_after_addr | bool | 2 |
| Store write_combining | bool | 2 |
| Store batch_stores | 1, 2, 4 | 3 |
| Store store_order | 3 options | 3 |
| Memory access_order | 4 options | 4 |
| Memory coalesce_loads | bool | 2 |
| Setup preload options | 2^3 | 8 |
| Phase order | 2 options | 2 |
| Fusion mode | 2 effective | 2 |

**Estimated unique programs: ~10^9**

### If Dead Parameters Were Implemented

Adding functional implementations of:
- `outer_unroll` (4 values)
- `vector_grouping` (3 values)
- `addr_compute_ahead` (4 values)
- `HashNode.strategy` (4 values)
- `store_buffer_depth` (3 values)
- PipelineNode (multiple params)
- InterleaveNode (multiple params)
- RegisterNode (multiple params)

**Estimated unique programs: ~10^15**

---

## 5. Implementation Priority

### High Impact (Should Implement)

1. **`outer_unroll`** - Loop unrolling is a classic optimization
2. **`HashNode.strategy`** - The enum exists, just needs the switch statement
3. **`addr_compute_ahead`** - Pipelining address computation with loads
4. **`vector_grouping`** - Processing multiple vectors together

### Medium Impact

5. **`store_buffer_depth`** - Delayed store flushing
6. **Expand `inner_unroll` to 8** - Full vector processing
7. **Expand `parallel_chunks` to 32** - More parallelism (memory permitting)

### Lower Priority (Complex)

8. **PipelineNode** - Requires significant new scheduling logic
9. **InterleaveNode** - Cross-phase instruction mixing
10. **RegisterNode** - Register allocation strategies

---

## 6. Structural Constraints (Not Parameters)

These are architectural limitations, not individual parameters:

1. **Phase Order** - Only 2 of 24 possible orderings
2. **Fixed Computation** - Hash algorithm, index formula are immutable
3. **No Nested Loops** - Single loop structure only
4. **Greedy Scheduling** - No exploration of scheduling heuristics
5. **Fixed Vector Width** - Always process full VLEN=8
6. **No Instruction Substitution** - Can't evolve `x*2` → `x<<1`

---

## Appendix: Verification Commands

```bash
# Test any parameter
source .venv/bin/activate
python -c "
from gp_optimizer_v3 import *

def make_prog(**overrides):
    # ... create minimal program with overrides
    pass

# Compare instruction counts
prog1 = make_prog(param=value1)
prog2 = make_prog(param=value2)
i1 = len(GPKernelBuilderV3(prog1).build(10, 1023, 256, 16))
i2 = len(GPKernelBuilderV3(prog2).build(10, 1023, 256, 16))
print(f'value1: {i1}, value2: {i2}, delta: {i2-i1}')
"
```
