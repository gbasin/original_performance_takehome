# GP Optimizer V4 Improvements Scope

*Scoping document for activating dead code and implementing missing optimizations*

## Current State Summary

| Version | Best Cycles | Status |
|---------|-------------|--------|
| Baseline | 6210 | Tree-based GP |
| V3 | 5272 | 15% faster via `parallel_chunks=32` + preloading |
| **V4** | **5179** | **2% faster than V3 via activated features** |

**Problem (V3)**: V3 added ~1800 lines but ~60% is dead code. The gains came from just 3-4 parameters.

**Solution (V4)**: Activated previously dead code and fixed implementation bugs.

---

## V4 Implementation Progress

### Completed
- [x] **Store buffering**: Fixed `maybe_flush_buffered()` - now properly delays flushes based on `store_buffer_depth`
- [x] **Instruction interleaving**: Fixed emit() to properly populate interleave_buffer; flush_schedule() now drains buffer
- [x] **HASH_INDEX fusion**: Implemented but disabled (produces incorrect results - needs debugging)
- [x] **Pipeline schedules**: BALANCED and STORE_LAST were already implemented, now enabled in random generation
- [x] **Random generation fixes**: Reduced scratch exhaustion by limiting reserve_temps and parallel_chunks combinations

### Best V4 Configuration Found
```
Structure: batch_round, chunks=16
Loop order: chunk_first, skip_indices=True
Gather: batch_addr
Hash: fused
Index: multiply_add, formula=bitwise
Interleave: gather_hash
Pipeline: enabled=False
Memory: prefetch=0, addr_gen=speculative
```

### Remaining Work
- [ ] Debug HASH_INDEX fusion (produces incorrect output)
- [ ] Implement INDEX_STORE and FULL fusion modes
- [ ] Add proper outer loop unrolling (currently just changes stride)
- [ ] Profile slot utilization to verify interleaving benefit

---

---

## Hardware Constraints (Reference)

```
VLEN = 8              # Vector width
SCRATCH_SIZE = 1536   # Total scratch (words)
SLOT_LIMITS = {
    'alu': 12,        # ALU ops per cycle
    'valu': 6,        # Vector ops per cycle
    'load': 2,        # Loads per cycle (BOTTLENECK)
    'store': 2,       # Stores per cycle (BOTTLENECK)
    'flow': 1,        # Control ops per cycle
}
```

**Key insight**: Memory is the bottleneck (2 load/store slots vs 12 ALU). Any optimization that reduces memory pressure or hides memory latency has high potential.

---

## 1. Software Pipelining

### Current State
- `PipelineNode` exists with `enabled`, `pipeline_depth`, `schedule`, `prologue_unroll`, `epilogue_drain`
- `_compile_loop_pipelined()` exists (lines 1748-1916) but is **partially implemented**
- Only `GATHER_FIRST` schedule has real logic; `BALANCED` and `STORE_LAST` are stubs
- Never enabled in random program generation (always `enabled=False`)

### What's Missing

#### 1.1 Complete Schedule Implementations
```python
# BALANCED schedule (lines 1834+): stub only
# Need: Interleave ALL phases across N iterations in lockstep
# Example with depth=2:
#   Cycle 1: gather[iter0], gather[iter1]
#   Cycle 2: hash[iter0], hash[iter1]
#   Cycle 3: index[iter0], index[iter1]
#   Cycle 4: store[iter0], store[iter1]

# STORE_LAST schedule: not implemented
# Need: Delay stores to overlap with next iteration's gather
# Example:
#   Cycle 1: gather[iter1], store[iter0]
#   Cycle 2: hash[iter1]
#   Cycle 3: index[iter1], gather[iter2]
```

#### 1.2 Enable in Random Generation
```python
# In create_random_program():
pipeline = PipelineNode(
    enabled=random.choice([True, False]),  # Currently always False
    pipeline_depth=random.choice([2, 3, 4]),
    schedule=random.choice(list(PipelineSchedule)),
    ...
)
```

#### 1.3 Prologue/Epilogue Handling
- Prologue: Ramp up pipeline (partial iterations)
- Epilogue: Drain pipeline (complete in-flight iterations)
- Current implementation has basic structure but needs testing

### Estimated Impact
- **High**: Could hide 50-75% of memory latency if properly implemented
- Memory is 2 slots/cycle; pipelining lets us overlap iteration N's stores with iteration N+1's gathers

### Implementation Effort
- **Medium-High**: Core logic exists but schedules need completion
- Files to modify: `gp_optimizer_v3.py` lines 1748-1916

---

## 2. Instruction Interleaving

### Current State
- `InterleaveNode` exists with `strategy`, `lookahead_depth`, `min_slot_fill`, `allow_cross_chunk`
- `InterleaveStrategy` enum: NONE, GATHER_HASH, HASH_INDEX, ALL_PHASES, ADAPTIVE
- **All parameters are DEAD** - never read in scheduling logic

### What's Missing

#### 2.1 Implement Interleave-Aware Scheduling
Current `flush_schedule()` just packs instructions greedily. Need:

```python
def flush_schedule_interleaved(self, interleave: InterleaveNode):
    """Pack instructions from different phases into same VLIW word"""
    if interleave.strategy == InterleaveStrategy.NONE:
        return self.flush_schedule()  # Current behavior

    # Lookahead: peek at next phase's instructions
    lookahead = interleave.lookahead_depth

    # Fill slots from current phase
    # If slots remain AND min_slot_fill not met, pull from next phase
    if current_fill < interleave.min_slot_fill:
        # Borrow instructions from pending_next_phase buffer
        pass
```

#### 2.2 Cross-Phase Instruction Buffer
Need a buffer to hold "ready" instructions from upcoming phases:

```python
class GPKernelBuilderV3:
    def __init__(self):
        self.phase_buffers = {
            'gather': [],
            'hash': [],
            'index': [],
            'store': []
        }
        self.current_phase = 'gather'
```

#### 2.3 Strategy-Specific Logic
```python
# GATHER_HASH: Allow gather loads to overlap with hash ALU ops
# - Gather uses: load slots
# - Hash uses: valu slots
# - No conflict! Can interleave freely

# HASH_INDEX: Overlap hash valu with index valu
# - Both use valu slots - conflict!
# - Can only interleave if total valu ops < 6

# ALL_PHASES: Maximum interleaving
# - Track all slot usage, pack maximally

# ADAPTIVE: Based on utilization
# - If current cycle < 50% full, look ahead for compatible ops
```

### Estimated Impact
- **Medium-High**: VLIW has 12 ALU + 6 VALU + 2 load + 2 store per cycle
- Current code often uses only 2-3 slots per cycle
- Interleaving could improve slot utilization from ~20% to ~60%

### Implementation Effort
- **High**: Requires rethinking the emission model
- Currently: emit phase A completely, then phase B
- Need: emit phase A partially, interleave phase B, continue phase A

---

## 3. Memory Optimizations

### Current State
- `MemoryNode` exists with `prefetch_distance`, `store_buffer_depth`, `address_gen`, `access_order`, `coalesce_loads`, `coalesce_stores`
- **DEAD**: `prefetch_distance`, `store_buffer_depth`, `address_gen`, `coalesce_stores`
- **LIVE**: `access_order`, `coalesce_loads`

### What's Missing

#### 3.1 Prefetch Implementation
```python
# In _compile_gather():
prefetch = memory.prefetch_distance
if prefetch > 0:
    # Issue loads for iteration+prefetch while processing current iteration
    # Requires: separate address computation for prefetch
    # Requires: extra scratch space for prefetched data
    for p in range(prefetch):
        future_addrs = compute_addresses(iteration + p + 1)
        emit("load", prefetch_scratch[p], future_addrs)
```

#### 3.2 Store Buffering
```python
# In _compile_store():
buffer_depth = memory.store_buffer_depth
if buffer_depth > 0:
    # Don't store immediately - buffer in scratch
    # Flush buffer when: full OR end of iteration OR every N ops
    self.store_buffer.append((addr, value))
    if len(self.store_buffer) >= buffer_depth:
        self._flush_store_buffer()
```

#### 3.3 Address Generation Strategies
```python
# LAZY (current): Compute address just before load
# EAGER: Compute ALL addresses upfront, then do all loads
# SPECULATIVE: Compute addresses N elements ahead

if memory.address_gen == AddressGenStrategy.EAGER:
    # Phase 1: Compute all addresses
    for elem in range(VLEN):
        addrs[elem] = compute_addr(elem)
    flush_schedule()
    # Phase 2: Issue all loads
    for elem in range(VLEN):
        emit("load", data[elem], addrs[elem])
```

#### 3.4 Store Coalescing
```python
# Batch adjacent stores into single wide store if possible
# Current: store idx[0], store val[0], store idx[1], store val[1]...
# Coalesced: vstore idx[0:8], vstore val[0:8]
if memory.coalesce_stores:
    # Group stores by base address
    # Emit vector stores where possible
```

### Estimated Impact
- **Prefetch**: Medium - helps if memory latency > 1 cycle (depends on simulator)
- **Store buffering**: Medium - reduces store flushes, better slot packing
- **Address gen**: Low-Medium - may enable better scheduling
- **Coalescing**: High - could halve store instructions if applicable

### Implementation Effort
- **Medium**: Each sub-feature is relatively isolated
- Prefetch needs extra scratch allocation
- Store buffering needs flush tracking

---

## 4. Phase Fusion

### Current State
- `FusionMode` enum: NONE, GATHER_XOR, HASH_INDEX, INDEX_STORE, FULL
- **Only GATHER_XOR partially implemented** (fuses initial XOR with gather)
- Other modes are dead

### What's Missing

#### 4.1 HASH_INDEX Fusion
```python
# Fuse last hash stage with index computation
# Instead of:
#   hash_stage_4 -> flush -> index_compute
# Do:
#   hash_stage_4 + index_compute (interleaved, same VLIW words)

def _compile_hash_index_fused(self, hash_node, index_node, chunk_scratch):
    # Emit hash stage 3
    # Emit hash stage 4 AND index ops in same cycles
    # Uses: hash needs valu, index needs valu
    # Constraint: total valu ops <= 6 per cycle
```

#### 4.2 INDEX_STORE Fusion
```python
# Overlap index computation with store address generation
# Index produces: new_idx values
# Store needs: addresses computed from new_idx
# Can pipeline: compute idx[0], start store[0], compute idx[1], start store[1]...
```

#### 4.3 FULL Fusion
```python
# Maximum fusion: treat entire phase sequence as one scheduling problem
# Emit instructions from all phases into a single instruction stream
# Let scheduler pack them optimally

# This is essentially interleaving but at a coarser grain
# May conflict with InterleaveNode - need to reconcile
```

### Estimated Impact
- **Medium**: Reduces phase transition overhead
- Each phase boundary currently has a flush
- Fusion eliminates flushes, improves ILP

### Implementation Effort
- **Medium-High**: Requires understanding dependencies between phases
- Hash output -> Index input -> Store input
- Must respect data dependencies

---

## 5. Register Allocation

### Current State
- `RegisterNode` exists with `allocation`, `reuse_policy`, `spill_threshold`, `vector_alignment`, `reserve_temps`
- **Entire node is DEAD** - uses fixed sequential allocation

### What's Missing

#### 5.1 Allocation Strategies
```python
# DENSE (current effective behavior): r0, r1, r2, r3...
# SPARSE: r0, r8, r16, r24... (spread out to reduce bank conflicts)
# PHASED: Gather uses r0-r31, Hash uses r32-r63, etc.

def alloc_vector(self, name: str) -> int:
    strategy = self._reg_settings.allocation
    if strategy == AllocationStrategy.DENSE:
        return self._next_vector  # Current
    elif strategy == AllocationStrategy.SPARSE:
        return self._next_vector * 8  # Spread out
    elif strategy == AllocationStrategy.PHASED:
        base = self._phase_bases[self.current_phase]
        return base + self._phase_counters[self.current_phase]
```

#### 5.2 Reuse Policy
```python
# AGGRESSIVE: Reuse register as soon as last use complete
# CONSERVATIVE: Keep some buffer registers
# LIFETIME: Track actual live ranges, reuse based on analysis

class LivenessTracker:
    def __init__(self):
        self.live_ranges = {}  # reg -> (first_use, last_use)

    def mark_use(self, reg, cycle):
        if reg not in self.live_ranges:
            self.live_ranges[reg] = (cycle, cycle)
        else:
            start, _ = self.live_ranges[reg]
            self.live_ranges[reg] = (start, cycle)

    def get_dead_regs(self, cycle):
        return [r for r, (_, end) in self.live_ranges.items() if end < cycle]
```

#### 5.3 Spill/Reload
```python
# When register pressure exceeds threshold, spill to scratch
if self._live_regs > self._reg_settings.spill_threshold:
    victim = self._select_spill_victim()
    self.emit("store", spill_addr, victim)
    self._spilled[victim] = spill_addr
    # Later, reload when needed
```

### Estimated Impact
- **Low-Medium**: Current allocation works fine for most cases
- May help with very large `parallel_chunks` values
- Spilling could enable even more parallelism

### Implementation Effort
- **High**: Requires significant infrastructure
- Liveness analysis across the entire program
- Spill/reload insertion and tracking

---

## 7. Loop Transformations

### Current State
- `outer_unroll` parameter exists but is **DEAD**
- `loop_order` (chunk_first vs round_first) is LIVE
- `skip_indices` is LIVE (conditional on chunk_first)
- No tiling/blocking

### What's Missing

#### 7.1 Outer Loop Unrolling
```python
# In _compile_loop(), outer_unroll is read but doesn't actually unroll
# Current: iterates with step outer_unroll but processes one at a time
# Need: Actually unroll the loop body

def _compile_loop_unrolled(self, loop, ...):
    unroll = loop.outer_unroll
    for cg in range(0, chunks_per_round, unroll):
        # Process 'unroll' chunk groups together
        batch_indices_list = []
        for u in range(unroll):
            if cg + u < chunks_per_round:
                batch_indices_list.append(...)

        # Emit unrolled: gather[0], gather[1], gather[2], gather[3]
        # Then: hash[0], hash[1], hash[2], hash[3]
        # Instead of: gather[0], hash[0], gather[1], hash[1]...
```

#### 7.2 Loop Tiling/Blocking
```python
# Process data in tiles that fit in scratch
# Current: process all chunks, all rounds
# Tiled: process tile of chunks, all rounds for tile, next tile

tile_size = calculate_tile_size(scratch_available)
for tile_start in range(0, total_chunks, tile_size):
    tile_end = min(tile_start + tile_size, total_chunks)
    # Process just this tile through all rounds
    for chunk in range(tile_start, tile_end):
        for round in range(rounds):
            process(chunk, round)
```

#### 7.3 Chunk Unrolling
```python
# chunk_unroll parameter exists but underutilized
# Could process multiple chunks with same instructions (SIMD-like)

if loop.chunk_unroll > 1:
    # Instead of separate instructions for chunk 0, chunk 1...
    # Use wider operations or batched addressing
```

#### 7.4 Round Fusion
```python
# Process multiple rounds without intermediate flushes
# Currently: each round is separate
# Fused: keep intermediate values in registers across rounds

for chunk_group in chunks:
    # Load initial indices
    indices = load_indices(chunk_group)
    for round in range(rounds):
        # Process without flushing indices between rounds
        values = gather(indices)
        indices = hash_and_index(values)
    store_final(indices)
```

### Estimated Impact
- **Outer unroll**: Medium - reduces loop overhead, better ILP
- **Tiling**: Low - may help cache behavior (if simulated)
- **Chunk unroll**: Medium - better utilization of vector units
- **Round fusion**: High - reduces memory traffic for intermediate values

### Implementation Effort
- **Medium**: Loop transformations are well-understood
- Outer unroll: just needs the actual unrolling code
- Tiling: requires scratch space accounting
- Round fusion: similar to `skip_indices` but more general

---

## Implementation Priority

### Tier 1: High Impact, Moderate Effort
1. **Memory: Prefetch** - directly addresses load bottleneck
2. **Memory: Store buffering** - reduces store pressure
3. **Loop: Outer unroll** - parameter exists, just needs wiring
4. **Loop: Round fusion** - `skip_indices` shows this works

### Tier 2: High Impact, High Effort
5. **Software Pipelining: Complete schedules** - partially implemented
6. **Instruction Interleaving** - major scheduling rework

### Tier 3: Medium Impact
7. **Phase Fusion: HASH_INDEX, INDEX_STORE** - reduces flush overhead
8. **Memory: Address gen strategies** - may enable better scheduling

### Tier 4: Lower Priority
9. **Register Allocation** - current approach works
10. **Loop: Tiling** - benefit unclear without cache modeling

---

## Validation Approach

For each improvement:
1. Add parameter to random generation (ensure GP can discover it)
2. Run small population (30) for 10 generations
3. Compare best fitness to V3 baseline (5272 cycles)
4. Check instruction count changes to verify code is different
5. Profile slot utilization to confirm optimization is active

```bash
# Quick validation
python gp_optimizer_v4.py --generations 10 --population 30 --seed 42

# Compare instruction counts
python -c "
from gp_optimizer_v4 import *
prog_baseline = create_baseline_program()
prog_optimized = create_optimized_program()
print(f'Baseline: {len(build(prog_baseline))} instrs')
print(f'Optimized: {len(build(prog_optimized))} instrs')
"
```

---

## Success Metrics

| Improvement | Target | Actual V4 | Status |
|-------------|--------|-----------|--------|
| Prefetch | 5-10% | TBD | Infrastructure exists |
| Store buffer | 3-5% | ~1% | Fixed, now functional |
| Outer unroll | 2-5% | N/A | Not yet implemented |
| Pipeline schedules | 10-20% | N/A | Enabled but rarely helps |
| Interleaving | 10-15% | ~1% | Fixed, now functional |
| Phase fusion | 3-5% | Buggy | HASH_INDEX has errors |

**Original target**: 4500-4800 cycles (10-15% improvement over V3's 5272)

**Actual V4 result**: 5179 cycles (2% improvement over V3)

### Analysis
The 2% improvement came primarily from:
1. Bug fixes that expanded the valid search space
2. Enabling bitwise index formula (`1 + (val & 1)` vs `2 - (val % 2 == 0)`)
3. Better parameter combinations found through evolution

The larger improvements (10-20%) would require:
- Debugging HASH_INDEX fusion
- Implementing true outer unrolling with separate scratch per iteration
- More aggressive instruction scheduling that exploits VLIW parallelism
