# GP Search Space Analysis

## Current Constraints in GP V3

### 1. Hard-Coded Numeric Ranges

| Parameter | Current Range | Could Be |
|-----------|---------------|----------|
| `parallel_chunks` | [4, 8, 16] | [2, 4, 8, 16, 32] |
| `pipeline_depth` | [1, 2, 3] | [1, 2, 3, 4] |
| `outer_unroll` | [1, 2] | [1, 2, 4, 8] |
| `chunk_unroll` | [1, 2] | [1, 2, 4] |
| `inner_unroll` | [1, 2, 4] | [1, 2, 4, 8] |
| `vector_grouping` | [1, 2] | [1, 2, 4] |
| `addr_compute_ahead` | [0, 1, 2] | [0, 1, 2, 4, 8] |
| `stage_unroll` | each [1, 2] | each [1, 2, 4] |
| `compute_unroll` | [1, 2] | [1, 2, 4] |
| `batch_stores` | [1, 2] | [1, 2, 4] |
| `prefetch_distance` | [0, 1, 2] | [0, 1, 2, 4, 8] |
| `store_buffer_depth` | [0, 1, 2] | [0, 1, 2, 4] |
| `lookahead_depth` | [1, 2, 3, 4] | [1, 2, 4, 8] |
| `min_slot_fill` | [0.3, 0.5, 0.7] | [0.1, 0.3, 0.5, 0.7, 0.9] |
| `spill_threshold` | [256, 512, 768] | [128, 256, 512, 768, 1024] |
| `vector_alignment` | [8, 16] | [8, 16, 32] |
| `reserve_temps` | [2, 4, 8] | [0, 2, 4, 8, 16] |

### 2. Dead Parameters (Declared but Not Implemented)

| Node | Unused Parameters | Impact |
|------|-------------------|--------|
| **PipelineNode** | `enabled`, `pipeline_depth`, `schedule`, `prologue_unroll`, `epilogue_drain` | No software pipelining of loop iterations |
| **InterleaveNode** | `strategy`, `lookahead_depth`, `min_slot_fill`, `allow_cross_chunk` | No cross-phase instruction mixing |
| **RegisterNode** | `allocation`, `reuse_policy`, `spill_threshold`, `vector_alignment`, `reserve_temps` | Fixed sequential register allocation |
| **MemoryNode** | `store_buffer_depth`, `coalesce_loads`, `coalesce_stores` | No store buffering or coalescing |
| **GatherNode** | `inner_unroll`, `vector_grouping`, `addr_compute_ahead` | Fixed gather loop structure |
| **HashNode** | `stage_unroll`, `cross_chunk_interleave` | Fixed hash stage structure |
| **IndexNode** | `compute_unroll`, `speculative` | Fixed index computation |
| **StoreNode** | `batch_stores`, `write_combining` | Fixed store pattern |
| **PhaseSequenceNode** | `fused_phases` list | Only GATHER_XOR fusion works |

### 3. Structural Constraints

**Phase Order:**
- Only 2 orderings allowed:
  - `[gather, hash, index, store]`
  - `[gather, index, hash, store]`
- Could have 24 permutations (4!)
- Could allow phases to be skipped or repeated

**Tree Structure:**
- Fixed depth: Program → Setup + Loop → PhaseSequence → Phases
- No nested loops (V2 had `inner_loop`, V3 removed it)
- Single loop structure per program
- Can't represent shared computation between phases

---

## Hidden Constraints (Structural Assumptions)

### 4. Fixed Instruction Sequences

Each phase always generates the same *pattern* of instructions. We pick strategies but the actual ALU/VALU sequences are hardcoded.

**Example - Hash always does:**
```
op1(val, const1) → tmp1
op3(val, const3) → tmp2
flush
op2(tmp1, tmp2) → val
```

**Never explores:**
- `op1 → op2 → op3` ordering
- Fusing operations differently
- Different temporary register assignments

### 5. Greedy Scheduling

The scheduler packs bundles greedily by dependency order.

**Never explores:**
- Deliberately leaving slots empty to reduce register pressure
- Different tie-breaking heuristics when multiple ops are ready
- Scheduling for cache locality vs ILP tradeoffs

### 6. Fixed Computation (Algorithmic)

**Hash algorithm is immutable:**
- 4 stages with fixed operations
- `HASH_STAGES` constant from problem.py
- Could there be mathematically equivalent but faster formulations?

**Index formula is fixed:**
```python
idx = idx * 2 + (2 - (val % 2 == 0))
```

**Algebraic equivalences not explored:**
- `x * 2` could be `x + x` or `x << 1`
- `x % 2` could be `x & 1`
- `2 - condition` could be `1 + (1 - condition)`

### 7. Single Loop Nest

**Always:**
```
for round in rounds:
    for chunk_group in chunks_per_round:
        process(chunk_group)
```

**Never explores:**
- Loop tiling: `for round_tile: for chunk_tile: for r in tile: for c in tile`
- Loop peeling for first/last iterations
- Loop versioning for special cases
- Duff's device style unrolling

### 8. Fixed Data Flow

**Always:**
```
load → gather → xor → hash → index → store
```

**Never explores:**
- Computing indices speculatively while gathering
- Overlapping round N's store with round N+1's gather
- Splitting phases (half gather, half hash, other half gather, other half hash)
- Out-of-order phase execution based on data availability

### 9. Bundle Boundaries = Cycle Boundaries

We assume `flush_schedule()` = cycle boundary.

**Could explore:**
- "Soft" flushes that hint but don't force
- Partial flushes (only some engines)
- Speculative execution past flush points

### 10. Fixed Vector Width Usage

**Always process full VLEN=8 vectors.**

**Never explores:**
- Processing 4 elements at a time with 2x unroll
- Mixed scalar/vector execution
- Partial vector operations for edge cases

### 11. No Instruction Substitution

`multiply_add` vs separate `mul` + `add` is a strategy choice, but many other substitutions exist:

| Operation | Alternatives |
|-----------|-------------|
| `x * 2` | `x + x`, `x << 1` |
| `x % 2` | `x & 1` |
| `x == 0` | `!x`, `x < 1` |
| `x * y + z` | `multiply_add(x, y, z)` |
| `a ? b : c` | `select`, `b * a + c * (1-a)` |

### 12. The GP Tree Structure Itself

Tree structure forces hierarchical decomposition.

**Cannot represent:**
- Shared computation between phases
- Instruction-level dataflow graphs (like CGP would)
- Feedback loops / iteration-carried dependencies
- Dynamic control flow

---

## Expansion Opportunities

### High Impact

1. **Linear GP (LGP) Layer**
   - Evolve actual instruction sequences within phases
   - Much larger search space but could find novel optimizations

2. **Algebraic Equivalence Exploration**
   - Let GP discover `x & 1` == `x % 2`
   - Strength reduction: `x * 2` → `x << 1`

3. **Cross-Phase Data Flow**
   - Break strict phase boundaries
   - Overlap independent operations from different phases

4. **Scheduling Exploration**
   - Evolve the scheduler heuristics, not just the code
   - Multiple scheduling strategies as GP choices

### Medium Impact

5. **Full Phase Permutations**
   - Allow all 24 orderings of 4 phases
   - Allow phase repetition or skipping

6. **Loop Transformations**
   - Tiling, peeling, versioning as evolvable choices
   - Nested loop structures

7. **Implement Dead Parameters**
   - Software pipelining
   - Instruction interleaving
   - Register allocation strategies

### Lower Impact (but easy)

8. **Expand Numeric Ranges**
   - Larger unroll factors
   - More prefetch distances
   - Wider parallel_chunks range

9. **More Strategies per Phase**
   - Additional gather/hash/index/store implementations
   - Hybrid strategies

---

## Current Effective Search Space

**Actually evolved (affects output):**
- Loop structure (4 options)
- parallel_chunks (3 options)
- Gather strategy (5 options) + flush booleans
- Hash strategy (4 options) + flush/preload booleans
- Index strategy (4 options) + flush/preload booleans + bounds_check_mode
- Store flush + store_order
- Setup preload booleans
- Phase order (2 options)
- Fusion mode (5 options, only 1 implemented)
- Memory access_order, prefetch_distance

**Rough estimate:** ~10^8 unique programs

**With dead parameters implemented:** ~10^15 unique programs

**With structural expansions (LGP, etc):** ~10^50+ unique programs
