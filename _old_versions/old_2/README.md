# Simulation, Data Collection, and VLA Training
This repository contains code for simulating robotic arms, collecting data from their movements, and training a Variable Length Array (VLA) model.

## Setting Up the Environment
```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data Collection
The `collect_data_*.py` script allows you to collect data from robotic simulations. It initializes the simulation environment, sets up the robots, and collects joint positions and end-effector positions over time.
```bash
python collect_data_joint_ctrl.py
```

## Reinforcement Learning Training (Degugging)
The `train_rl.py` script is used to train a reinforcement learning model using the collected data. It sets up the training configuration, initializes the simulation, and runs the training loop.
```bash
python train_rl.py
```