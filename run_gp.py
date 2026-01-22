#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)

from gp_optimizer_v4 import GeneticProgrammingV3

if __name__ == "__main__":
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f'Running GP with pop={pop}, gen={gen}', flush=True)
    gp = GeneticProgrammingV3(population_size=pop, generations=gen)
    best, fitness = gp.run()
    print(f'\nFinal: {fitness:.0f} cycles', flush=True)
    gp.save_results('gp_v4_results.json')
    print('Saved to gp_v4_results.json', flush=True)
