# GA Optimization Progress

## Current State

### Best Results
- **Original GA (ga_optimizer.py)**: 6,971 cycles with 16 parallel chunks
- **Exploratory GA (exploratory_ga.py)**: ~6,834 cycles with maximal_parallel seeded genome

### Target
- Need to reach <2,164 cycles to pass the next test threshold
- Baseline: 147,734 cycles (15.13x improvement achieved so far)

## Files

### `ga_optimizer.py`
Original GA optimizer with basic genome:
- batch/round unroll factors
- vectorization flags
- pipeline depth
- parallel chunks (up to 16)

### `exploratory_ga.py`
New exploratory GA with more diverse strategies:

**Strategy Enums:**
- `GatherStrategy`: SIMPLE, DUAL_ADDR, BATCH_ADDR, INTERLEAVED, PIPELINED
- `HashStrategy`: STANDARD, FUSED, STAGE_PAIRS, UNROLLED, MINIMAL_SYNC
- `IndexStrategy`: VSELECT, ARITHMETIC, MULTIPLY_ADD, BITWISE, BRANCHLESS

**Key Genome Parameters:**
- `parallel_chunks`: 1-16 (limited by scratch space)
- `preload_vector_constants`: bool
- `preload_n_nodes_vector`: bool
- Fine-grained flush control (12+ boolean flags)

## Known Issues

1. **Scratch space limit**: 32 parallel chunks exceeds SCRATCH_SIZE (1536 words) when combined with preloaded constants. Limited to 16 max.

2. **Some genome configurations hang**: Certain combinations (e.g., 16 chunks + BATCH_ADDR gather + STAGE_PAIRS hash) generate 14,000+ instructions and hang during evaluation. The Machine simulator runs forever without reaching the second pause instruction.

3. **Gather strategies all produce similar results**: After fixing correctness bugs (must flush between address computation and loads), all gather strategies produce ~8,610 cycles with 8 chunks. The 2 load slots per cycle is the fundamental bottleneck.

4. **Multiprocessing Pool overhead**: On macOS, Pool has significant startup overhead. Disabled in favor of sequential evaluation.

## What Works

Seeded genome results (8 chunks):
- baseline: 21,538 cycles
- minimal_flush: 7,266 cycles
- maximal_parallel: 6,978 cycles (16 chunks)
- pipelined: 9,634 cycles
- aggressive_unroll: 8,034 cycles

Strategy impact (8 chunks, default other settings):
- Hash FUSED vs STANDARD: ~4.5% improvement (8,226 vs 8,610)
- Index ARITHMETIC vs VSELECT: ~8% improvement (7,906 vs 8,610)
- Parallel chunks: 68% improvement from 1 to 16 chunks (22,050 vs 7,170)

## Next Steps

1. Debug the hanging configurations - likely issue with code generation for certain strategy combinations
2. Explore more radical architectural changes (different instruction ordering, different flush patterns)
3. Profile where cycles are actually being spent
4. Consider fundamentally different approaches to the gather bottleneck

## Running the GA

```bash
# Test seeded genomes
python -c "
from exploratory_ga import seeded_genome, evaluate_genome
for s in ['baseline', 'minimal_flush', 'maximal_parallel', 'pipelined', 'aggressive_unroll']:
    g = seeded_genome(s)
    print(f'{s}: {evaluate_genome(g):.0f} cycles')
"

# Small GA run (may hang on some configs)
python -c "
from exploratory_ga import ExploratoryGA
ga = ExploratoryGA(population_size=10, generations=5)
ga.run()
"
```
