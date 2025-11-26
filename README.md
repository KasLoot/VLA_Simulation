# Simulation Guide

## Quick Start
### 1. Install Dependencies
```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run the Simulation
1. This will run the simulation with a sine wave reference signal.
```bash
python run_simulation.py
```

2. This will allow manual control the joint angles.
```bash
python manual_control.py
```