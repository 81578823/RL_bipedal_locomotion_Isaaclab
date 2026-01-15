<p align="center">
    <img alt="SUSTech" src="./media/SUSTech_University_Logo.png" height="200">
    <img alt="CLEARLAB" src="./media/clearlab.png" height="200">
</p>

# Bipedal Robot RL Locomotion Learning Project

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository provides a modular framework for reinforcement learning (RL) research and development on bipedal robots [limxdynamics TRON1](https://www.limxdynamics.com/en/tron1), leveraging [IsaacLab](https://github.com/isaac-sim/IsaacLab) for high-fidelity simulation. It supports training, playing, and deployment of locomotion policies for various bipedal robots (pointfoot, solefoot, wheelfoot) across flat, rough, and stair terrains.

**Keywords:** isaaclab, locomotion, bipedal, pointfoot, TRON1

## Installation

- Install Isaaclab by following the [official installation guidance](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). We recommend using the miniconda installation as it simplifies calling Python scripts from the terminal.

**Attention:** Please switch to 2.1.0 branch in the official guidance if you want to use the learning project directly. **This project is designed for Isaaclab 2.1.0 + isaacsim 4.5.0.**

- Clone the repository:

```bash
# Option 1: HTTPS
git clone https://github.com/81578823/RL_bipedal_locomotion_Isaaclab.git

# Option 2: SSH
git clone git@github.com:81578823/RL_bipedal_locomotion_Isaaclab.git
```

```bash
# Enter the repository
conda activate isaaclab     # Or virtual environment you have created
cd RL_bipedal_locomotion_Isaaclab
```

- Using a python interpreter that has IsaacLab installed, install the library

```bash
python -m pip install -e exts/bipedal_locomotion
```

- To use the mlp branch, install the library

```bash
cd RL_bipedal_locomotion_Isaaclab/rsl_rl
python -m pip install -e .
```

## Set up IDE for convenience (Optional)

To setup the IDE, please follow these instructions:

- Replace the path in .vscode/settings.json with the Isaaclab and python paths used by the user. This way, when the user retrieves the official functions or variables of Isaaclab, they can directly jump into the definition of the configuration environment code.

And then you can use the ```F5``` key to debug this project, which will launch debug function following the setup in ```.vscode/launch.json```. You can modify launch.json to debug new tasks you design.

## Train a bipedal robot agent

- Use the `scripts/rsl_rl/train.py` script to run a simple robot training task, specifying the task:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Flat-v0 --headless

#or the following for rough terrains

python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Rough-v0 --headless
```

- The following arguments can be used to customize the training:
    * --headless: Run the simulation in headless mode
    * --num_envs: Number of parallel environments to run
    * --max_iterations: Maximum number of training iterations
    * --save_interval: Interval to save the model
    * --seed: Seed for the random number generator

## Play the trained model

- To play a trained model:

```bash
#this is for flat terrain
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --num_envs=10

#this is for multi-terrains
python3 scripts/rsl_rl/play.py   --task=Isaac-Limx-PF-Blind-Rough-Play-Near-v0   --checkpoint_path=/path_to_ckpt  --num_envs=1 --video --video_length 800
```

- You can customize `play.py` with:
  * `--num_envs`: number of parallel environments
  * `--headless`: run without rendering
  * `--checkpoint_path`: checkpoint to load
  * `--video`: enable video capture (recommended)
  * `--video_length`: number of steps to record

- Keyboard/gamepad controls:
  * Directions (WASD): W forward, A left, S back, D right
  * Gamepad: supported when connected
  * Push forces: 1 forward, 2 left, 3 back, 4 right (tunable in `play.py`)

- Also supports angular and linear velocity tracking.

## Run the exported model in mujoco (sim2sim)

- After playing the model, the policy has already been saved. You can export the policy to mujoco environment and run it in mujoco [tron1-mujoco-sim]((https://github.com/limxdynamics/tron1-mujoco-sim)) by using the [tron1-rl-deploy-python]((https://github.com/limxdynamics/tron1-rl-deploy-python)).

- Following the instructions to install it properly and replace the origin policy by your trained `policy.onnx` and `encoder.onnx`.

## Run the exported model in real robot (sim2real)

- Real deployment details see section https://support.limxdynamics.com/docs/tron-1-sdk/rl-training-results-deployment 8.1 ~ 8.2

## Overview of the learning framework

<p align="center">
    <img alt="Figure2 of CTS" src="./media/learning_frame.png">
</p>

- The policies are trained using PPO within an asymmetric actor-critic framework, with actions determined by history observations latent and proprioceptive observation. **Inspired by the paper CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion. ([H. Wang, H. Luo, W. Zhang, and H. Chen (2024)](https://doi.org/10.1109/LRA.2024.3457379))**


## Video Demonstration

### Simulation in IsaacLab
- **Multi-terrain traversal**  
  ![multi-terrain-traversal](./media/multi-terrain-traversal.gif)

- **High-accuracy speed tracking**  
  ![speed tracking](./media/speed_tracking.gif)

  <p align="left">
      <img alt="CLEARLAB" src="./media/speed_tracking.PNG" height="317">
  </p>

- **High pushing tolerance**  
  ![pushing](./media/push.gif)
