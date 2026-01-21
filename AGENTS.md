# Agent Instructions

## Environment Setup

This project requires Python with numba for the GP optimizer. Set up a virtual environment:

```bash
# One-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
python3 -c "from numba_machine import NumbaMachine; print('OK')"
```

## Subsequent Sessions

Always activate the venv before working:

```bash
source venv/bin/activate
```

## Running the GP Optimizer

```bash
# Quick test (few generations)
python3 gp_optimizer_v3.py --generations 5 --population 30

# Full run
python3 gp_optimizer_v3.py --generations 100 --population 50
```

## Key Files

- `gp_optimizer_v3.py` - Main GP optimizer with VLIW kernel synthesis
- `problem.py` - Problem definition and reference kernel
- `numba_machine.py` - JIT-compiled VLIW machine simulator
- `requirements.txt` - Python dependencies
