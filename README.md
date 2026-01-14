# Scaling Model-Based Control with Latent Stochastic Neural Differential Equations

This repository introduces model-based RL (MBRL) with latent Stochastic neural differential equations (neural SDEs). The proposed code base builts upon a PyTorch replication of the DreamerV3 architecture  (https://github.com/NM512/dreamerv3-torch) and replaces the dynamics of the world model by latent neural SDEs More precisely, the discrete Recurrent-State-Space-Model (RSSM) is transformed into a continuous-time reformulation. This allows to compare the RSSM against latent neural SDEs on environments with irregular time data. 

## Overview
To get a general overview of the DreamerV3 code base, i recommend to read the documentation of the original code base located in Readme.md. This short overview presents the main features that were included to the frame work to integrate latent neural SDEs.

### Dynamics Replacement
The `World Model` class located in `models.py` contains an attribute called `dynamics`that is responsible for simulating the dynamics of the environment and generating imagination rollouts. The current dynamics `RSSM` implemented in `networks.py` is replaced by `ControlledLatentSDE`. This class represents the latent neural SDE with posterior, prior and diffusion functions. The instance of `ControlledLatentSDE` is not directly used as the dynamics of the `World Model`class, instead the wrapper `LatentSDEDreamerInterface` is defined as dynamics and provides the necessary interfaces for the world model for training, planning and interaction.

### Irregular Setting
To test the performance of the models in an environment with irregular time data, the environment definition from OpenAI Gymnasium implemented in `envs/gymnasium`is adapted to allow irregular decision times by action holds (an action is hold constant for `k` steps). A new data set structure implemented in the class `ReplayBuffer`allows to store trajectories with a fixed time grid (can be irregular) and therefore facilitates the SDE integration of a batch of sequences.

## Instructions

### 1: Install requirements
To use the code base, the python version 3.11 needs to be installed in the system. Packages such as `conda` can be used.

Clone the repository and a root level, run the following command to get all dependencies.
```
pip install -r requirements.txt
```

It might be necessary to change the rendering backend specified in the file `dreamer.py` by the code line 

```
os.environ["MUJOCO_GL"] = "osmesa"
````
dependent on the system where the the script runs. If `osmesa` does not work, try to use `egl` or `glfw`.

### 2. Edit configurations
To conifgure parameters for the environment and the models (World model, Task behavior, etc.), edit the  file `configs.yaml`. The file contains configurations for each type of environment and i recommend to read the header description of the file how to speicify configurations for the new proposed features. To switch between the `RSSM`and `ControlledLatentSDE`, use the flag `use_sde`.

### 3. Run dreamer agent
The file `dreamer.py`is the main script to run the dreamer agent. Dependent on which environment is used, the command is different. In the following example commands are shown

Run training on DMC Vision (Pixel-based observations)
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_vision_walker_walk
```
Other environments can be run by changing `dmc_walker_walk`to `dmc_<new_env_id>`

Run training on DMC Proprio (Vector-based observations)
```
python3 dreamer.py --configs dmc_proprio --task dmc_walker_walk --logdir ./logdir/dmc_proprio_walker_walk
```

Run training on Gymnasium Environment 

```
python3 dreamer.py --configs gym_pendulum --task gym_Pendulum-v1 --logdir ./logdir/gym_pendulum
```

### 4. Monitor results:
To monitor the performance metrics, use the Tensorboard tool. 

```
tensorboard --logdir ./logdir
```

In case that an Gymnasium environment is run, the log directory contains additional visualization plots (Posterior vs Prior vs Ground Truth).

## Acknowledgments
The code builts upon the PyTorch replication of the DreamerV3 architecture (https://github.com/NM512/dreamerv3-torch).

## World Model Training on Pendulum environment
The scripts `create_replay_buffer` allows to create random trajectories of the pendulum environment of OpenAI Gymnasium and stores them in the `ReplayBuffer` class. By this script, training and evaluation data sets can be created and used to train the RSSM/neural SDE based world models without control by the script `train_world_model.py`. Further details can be found in the corresponding files.