"""Random Trajectory Generator for Pendulum Environment
This script allows the user to generate random trajectories from the Pendulum environment of OpenAI Gymnasium.
The trajectories are stored in a replay buffer instance from the class *ReplayBuffer* and a modified version
of the environment implemented by the class *GymEnv* enables the creation of data that are sampled on an irregular 
time grid. 

Command-line arguments:
    -- batch_size: int              Number of trajectories per batch. This is relevant for batches with each 
                                    consisting of a different time grid.
    -- time_limit: int              Time limit per trajectory in seconds 
    -- irregular: optional          Enables to sample irregular data. Each batch will contain trajectories with a 
                                    fixed time grid that is non-uniform
    -- capacity: int                Capacity of the replay buffer in terms of maximum number of batches that can 
                                    be stored in the replay buffer
    -- seed: int                    Random seed to allow deterministic runs
    -- num_batches:                 Number of batches to store in the replay buffer      
    -- physical_step_size: float    Physical step size (frequency) of the pendulum environment
    -- directory: str               Output directory of the replay buffer
    -- action_hold_min: int         Minimum number of action holds in the environment. 
    -- action_hold_max: int         Maximum number of action holds in the environment. 

Example command:
python ./create_replay_buffer.py --batch_size 16 --time_limit 10 --irregular --capacity 500 --seed 0 --num_batches 100 --physical_step_size 0.05 -- directory data/test --action_hold_min 1 --action_hold_max 2
"""

import numpy as np
from envs.gymnasium import GymEnv
from replaybuffer import ReplayBuffer
from tqdm import tqdm
import os
import json
import argparse
import tools

def sample_trajectory(env: GymEnv, replayBuffer: ReplayBuffer):
    """Sample a trajectory from the pendulum environment using random actions.

    Parameters:
    -----------
    env: GymEnv
        environment instance
    replayBuffer: ReplayBuffer
        replay buffer instance
    Returns:
    --------
    trajectory
        a trajectory with random actions 
    """

    # Check if current batch in replay buffer is empty. 
    # If yes, then a new time grid needs to be created.
    if replayBuffer._current_count_in_batch == 0:
        env.make_time_grid()

    # Reset environment. Observation is a dictionary 
    obs = env.reset()

    # Keys of the trajectory that specify each transition
    trajectory_keys = [
        "obs",
        "image",
        "action",
        "reward",
        "discount",
        "is_first",
        "is_terminal",
        "time",
    ]

    # Initialize trajectory
    trajectory = {key: [] for key in trajectory_keys}

    # Add initial state to the trajectory. Similar to Dreamer, add a zero 
    # action for the first state of the trajectory. This means action a_{i} always 
    # corresponds to the action applied on the observation o_{i-1}
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    trajectory["obs"].append(obs["obs"])
    trajectory["image"].append(obs["image"])
    trajectory["action"].append(zero_action)
    trajectory["reward"].append(0.0)
    trajectory["discount"].append(1.0)
    trajectory["is_first"].append(True)
    trajectory["is_terminal"].append(False)
    trajectory["time"].append(obs["time"])

    # Episode loop
    done = False
    while not done:
        # Sample random action using the action space from the environment
        action = env.action_space.sample()

        # Perform a step in the environment
        next_obs, reward, done, _ = env.step(action)

        # Add transition to the trajectory
        trajectory["obs"].append(next_obs["obs"])
        trajectory["image"].append(next_obs["image"])
        trajectory["action"].append(action)
        trajectory["reward"].append(reward)
        trajectory["discount"].append(1.0 - float(done))
        trajectory["is_first"].append(False)
        trajectory["is_terminal"].append(done)
        trajectory["time"].append(next_obs["time"])

        obs = next_obs

    # Convert transitions to numpy array
    for k in trajectory_keys:
        trajectory[k] = np.asarray(trajectory[k])

    return trajectory

def parse_args():
    """Parse command line arguments.

    Returns:
    --------
    argparse.Namespace
        Arguments parsed to the command-line that specify configurations of the replay buffer 
        and the environment
    """
    parser = argparse.ArgumentParser(description="Replay Buffer (Dreamer-style)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of trajectories per batch")
    parser.add_argument("--time_limit", type=int, default=200, help="Time limit per trajectories")
    parser.add_argument("--irregular", action="store_true", help="Sample irregular data")
    parser.add_argument("--capacity", type=int, default=100, help="capacity of replay buffer")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to sample")
    parser.add_argument("--physical_step_size", type=float, default=0.05, help="Step size of numerical integration of pendulum environment")
    parser.add_argument("--action_hold_min", type=int, default=1, help="Minimum number of action holds in the environment")
    parser.add_argument("--action_hold_max", type=int, default=2, help="Maximum number of action holds in the environment")
    
    parser.add_argument("--directory")
    
    return parser.parse_args()

def save_metadata(args, filename_path: str="metadata.json"):
    """Save metadata to a JSON file.
    
    Parameters:
    -----------
    argparse.Namespace
        Arguments parsed to the command-line that specify configurations of the replay buffer 
        and the environment
    filename_path: str
        Path to file where metadata are stored
    """

    os.makedirs(os.path.dirname(filename_path), exist_ok=True) if os.path.dirname(filename_path) else None
    with open(filename_path, "w") as f:
        json.dump(vars(args), f, indent=4)

def main():
    # Parse arguments 
    args = parse_args()
    tools.set_seed_everywhere(args.seed)

    # Save meta data to JSON file
    save_metadata(args, os.path.join(args.directory, "metadata.json"))

    # Create environment
    env = GymEnv(name="Pendulum-v1", 
                 irregular=args.irregular, 
                 time_limit=args.time_limit, 
                 action_hold_min=args.action_hold_min,
                 action_hold_max=args.action_hold_max,
                 seed=args.seed)
    env.action_space.seed(args.seed)

    # Set frequency of environment 
    env.set_physical_dt(args.physical_step_size)

    # Create replay buffer
    replayBuffer = ReplayBuffer(args.directory, args.batch_size, args.capacity)

    # Sample trajectories
    # Progress bar
    pbar = tqdm(
        initial=len(replayBuffer),
        total=args.num_batches,
        desc="Fill replay buffer",
        unit="batch",
    )

    # Add trajectories until the desired number of batches is reached
    while len(replayBuffer) < args.num_batches: 
        # Get current size of replay buffer (number of batches) before adding a trajectory
        before = len(replayBuffer)
        
        # Generate random trajectory
        trajectory = sample_trajectory(env, replayBuffer)

        # Add trajectory to replay buffer
        replayBuffer.add_trajectory(trajectory)

        # Get current size of replay buffer (number of batches) after adding a trajectory
        after = len(replayBuffer)

        # Update progress bar
        pbar.update(after - before)

if __name__ == "__main__":
    main()
