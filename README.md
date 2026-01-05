# Bipedal Robot RL Locomotion Learning Project

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository is used to train and simulate bipedal robots, such as [limxdynamics TRON1](https://www.limxdynamics.com/en/tron1).
With the help of [IsaacLab](https://github.com/isaac-sim/IsaacLab), we can train the bipedal robots to walk in different environments, such as flat, rough, and stairs.

**Keywords:** isaaclab, locomotion, bipedal, pointfoot, TRON1

## Installation

- Install Isaaclab by following the [official installation guidance](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). We recommend using the miniconda installation as it simplifies calling Python scripts from the terminal.

**Attention:** Please switch to 2.1.0 branch in the official guidance if you want to use the learning project directly. **This project is designed for Isaaclab 2.1.0 + isaacsim 4.5.0.**

- Clone the repository:

```bash
# Option 1: HTTPS
git clone https://github.com/81578823/RL_bipedal_locomotion.git

# Option 2: SSH
git clone git@github.com:81578823/RL_bipedal_locomotion.git
```

```bash
# Enter the repository
conda activate isaaclab     # Or virtual environment you have created
cd RL_bipedal_locomotion
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e exts/bipedal_locomotion
```

- To use the mlp branch, install the library

```bash
cd RL_bipedal_locomotion/rsl_rl
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
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --num_envs=10
```

- The following arguments can be used to customize the playing:
    * --num_envs: Number of parallel environments to run
    * --headless: Run the simulation in headless mode
    * --checkpoint_path: Path to the checkpoint to load

## Run the exported model in mujoco (sim2sim)

- After playing the model, the policy has already been saved. You can export the policy to mujoco environment and run it in mujoco [tron1-mujoco-sim]((https://github.com/limxdynamics/tron1-mujoco-sim)) by using the [tron1-rl-deploy-python]((https://github.com/limxdynamics/tron1-rl-deploy-python)).

- Following the instructions to install it properly and replace the origin policy by your trained `policy.onnx` and `encoder.onnx`.

## Run the exported model in real robot (sim2real)

- Real deployment details see section https://support.limxdynamics.com/docs/tron-1-sdk/rl-training-results-deployment 8.1 ~ 8.2

**Overview of the learning framework.**

<p align="center">
    <img alt="Figure2 of CTS" src="./media/learning_frame.png">
</p>

- The policies are trained using PPO within an asymmetric actor-critic framework, with actions determined by history observations latent and proprioceptive observation. **Inspired by the paper CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion. ([H. Wang, H. Luo, W. Zhang, and H. Chen (2024)](https://doi.org/10.1109/LRA.2024.3457379))**


## Video Demonstration

### Simulation in IsaacLab
- **Pointfoot Blind Flat**:

![play_isaaclab](./media/play_isaaclab.gif)
### Simulation in Mujoco
- **Pointfoot Blind Flat**:

![play_mujoco](./media/play_mujoco.gif)

## Acknowledgements

This project uses the following open-source libraries:
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl/tree/master)
- [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab)
- [tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab)

**Contributors:**
- Hongwei Xiong 
- Bobin Wang
- Wen
- Haoxiang Luo
- Junde Guo