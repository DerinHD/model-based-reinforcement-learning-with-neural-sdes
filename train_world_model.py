"""World Model Training for Pendulum Environment
This script trains a world model on the pendulum environment of OpenAI Gymnasium using pre-collected data by create_replay_buffer.py
The world model is trained to reconstruct and predict the environment dynamics without interaction and no policy learning.

To modify the world model architecture and hyperparaemeters for training, update the parameters under the 
config gym_Pendulum in configs.yaml.
To run this script, the arguments 
--configs gym_pendulum 
--task gym_Pendulum-v1

Further Command-line arguments:
    --logdir: str                           Directory where results will be saved.
    --train_steps: int                      Number of training steps for the world model.
    --eval_interval: int                    Interval (in steps) to evaluate the world model during training.
    --replay_buffer_dir_train:  str         Directory where the replay buffer for training is stored.
    --replay_buffer_dir_eval:  str          Directory where the replay buffer for evaluatiion is stored.
    --visualize:   optional                 Flag to show visualizations during evaluation.

Example command:
python .\world_model_training.py --configs gym_pendulum --logdir sde_world_model --train_steps 10000 --eval_interval 500 --replay_buffer_dir_train ... --replay_buffer_dir_eval ... 
"""

import argparse
from models import WorldModel
from ruamel.yaml import YAML
import os
import pathlib
import sys
import gymnasium as gym
import numpy as np
import torch 
import tools
import matplotlib.pyplot as plt
import os
import tools
from replaybuffer import ReplayBuffer
from sklearn.decomposition import PCA

sys.path.append(str(pathlib.Path(__file__).parent))

yaml = YAML(typ="safe", pure=True)
from envs.gymnasium import GymEnv

import csv, pathlib
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Simple Logger class to store results in tensorboard and csv
    """
    def __init__(self, logdir: str, initial_step: int=0):
        """Docstring for __init__

        Parameters:
        -----------
        logdir: str
            Name of the directory where logs will be saved.
        initial_step: int
            Initial step count for logging. 
            Zero if model is used for the first time or non zero if model is loaded from a checkpoint path.
        """
        # Specify log directory and create if it not exists
        self._logdir = pathlib.Path(logdir)
        self._logdir.mkdir(parents=True, exist_ok=True)

        # Create a summary writer for Tensorboard
        self._writer = SummaryWriter(self._logdir.as_posix())
        self._step = initial_step

    def log_scalar_to_tensor_board(self, key: str, value: float, step=None):
        """Log a scalar value to TensorBoard.
        
        Parameters:
        -----------
        key: str
            name of the scalar
        value: float
            scalar value
        step:
            step count 
        """
        step = self._step if step is None else step
        self._writer.add_scalar(key, value, step)

    def log_to_csv(self, key: str, value: float, step=None): 
        """Log a scalar value to a CSV file.
        
        Parameters:
        -----------
        key: str
            name of the scalar
        value: float
            scalar value
        step:
            step count 
        """
        step = self._step if step is None else step
        csv_path = self._logdir / f"{key.replace('/', '_')}.csv"
        
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([step, value])

    def close(self):
        """Close the logger.
        """
        self._writer.close()


def save_model(model: WorldModel, step: int, path: str="world_model_checkpoint.pt"):
    """Save the model in a checkpoint file.
    
    Parameters:
    -----------
    model: WorldModel
        World model instance to save
    step:
        Current step count
    path:
        Path to save the checkpoint
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": model._model_opt._opt.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, path)
    print(f"World model checkpoint was saved to {path}")

def load_model(model: WorldModel, path: str="world_model_checkpoint.pt", device="cpu"):
    """Load the model checkpoint.

    Parameters:
    -----------
    model: WorldModel
        World model instance to load
    path:
        Path to the checkpoint file
    path:
        Device where the tensors are mapped to.
    """

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model._model_opt._opt.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint["step"]
    
    print(f"Model checkpoint was loaded from {path}")
    return step

def PCA_transformation(latent_posterior: torch.Tensor, 
                       latent_prior: torch.Tensor, 
                       rewards: torch.Tensor, 
                       logdir: str ):  
    """
    Perform a PCA transformation on latent trajectories of posterior and prior.
    A PCA is fitted on the posterior and the prior is tranformed onto this PCA space.
    Addtionally, a color bar visualizing the mapping of latent states to predicted rewards are included
    
    Parameters:
    -----------
    latent_posterior:
        latent trajectories of the posterior
    latent_prior:
        latent trajectories of the prior
    rewards:
       ground-truth rewards
    logdir:
        directory to save
    """

    # Detach tensors and convert to numpy
    latent_posterior = latent_posterior.detach().cpu().numpy()
    latent_prior = latent_prior.detach().cpu().numpy()
    reward_np = rewards.detach().cpu().numpy()

    B, T, L = latent_posterior.shape

    # Get flattened version
    latent_posterior_flattened = latent_posterior.reshape(B * T, L)
    latent_prior_flattened = latent_prior.reshape(B * T, L)
    reward_flattened = reward_np.reshape(B * T)

    # Create PCA and fit on posterior. Get the first two principal components
    pca = PCA(n_components=2)
    pca.fit(latent_posterior_flattened)

    # function generate a scatter plot
    def scatter_plot(ax, latent_trajectory, reward_np, title):
        latent_pca = pca.transform(latent_trajectory)
        sc = ax.scatter(
            latent_pca[:, 0],
            latent_pca[:, 1],
            c=reward_np,      
            cmap="viridis",        
            s=20
        )

        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True)

        return sc

    # Create subplots with two plots (Latent space for posterior and prior)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    sc_posterior = scatter_plot(ax[0], latent_posterior_flattened, reward_flattened, "Posterior Latent Space")
    sc_prior = scatter_plot(ax[1], latent_prior_flattened, reward_flattened,  "Prior Latent Space")

    # Add colorbar for reward
    colorbar = fig.colorbar(sc_posterior, ax=axes, label="Reward")

    save_directory = os.path.join(logdir, "visualizations_latent_space_by_PCA")
    os.makedirs(save_directory, exist_ok=True)
    path_to_directory = os.path.join(save_directory, f"latent_spcae_PCA_{step}.png")
    plt.savefig(path_to_directory)
    plt.show()

def parse_args():
    """Parse command line arguments.
    """
    # 0. load configs from configs.yaml. Similar to dreamer.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    # Add argument to name directory where results will be saved. 
    parser.add_argument("--logdir", type=str, required=True)
    # Add argument how many steps to train
    parser.add_argument("--train_steps", type=int, default=100000)
    # Add argument when to evaluate during training
    parser.add_argument("--eval_interval", type=int, default=1000)
    # Add argument if visualizations should be shown during evaluation # default is False if not provided
    parser.add_argument("--visualize", action="store_true") 
    # Add argument to specify folder where training replay buffer is stored
    parser.add_argument("--replay_buffer_dir_train", type=str, required=True)
    # Add argument to specify folder where evaluation replay buffer is stored
    parser.add_argument("--replay_buffer_dir_eval", type=str, required=True)

    return parser.parse_known_args()

if __name__ == '__main__':
    # 0. Load configs from configs.yaml. Similar to dreamer.py
    args, remaining = parse_args()
    
    print("Command Line Args:  ", args, remaining)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    print("Available configs: ", list(configs.keys()))

    # Recursive update function to merge configs. This code snippet was adapted from dreamer.py
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
        
    # 1. Create directory for saving results if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)

    # 2. Copy config parameters to logdir for reference
    with open(os.path.join(args.logdir, "used_configs.yaml"), "w") as f:
        yaml.dump(vars(config), f)
    print("Final configs used for training: ", config)
    tools.set_seed_everywhere(config.seed)
    if getattr(config, "deterministic_run", False):
        tools.enable_deterministic_run()

    # 3. Create environment to extract observation and action space. This environment will not be used further in this script
    env = GymEnv("Pendulum-v1", irregular=False, time_limit=10.0)

    # Define the dimension of actions. This is need to be compatible with the Dreamer code
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = int(env.action_space.n)
    else:
        num_actions = int(np.prod(env.action_space.shape)) if env.action_space.shape else 1
    config.num_actions = num_actions
    
    # 4. Create world model instance
    world_model = WorldModel(env.observation_space, env.action_space, step=0, config=config).to(device=config.device)

    # 5. Load replay buffer for training and evaluation
    replayBuffer_train = ReplayBuffer(args.replay_buffer_dir_train, 
                                batch_size=config.batch_size, 
                                capacity=config.capacity)  
    replayBuffer_train.load_dataset()
    

    replayBuffer_eval = ReplayBuffer(args.replay_buffer_dir_eval, 
                                batch_size=config.batch_size, 
                                capacity=config.capacity)  
    replayBuffer_eval.load_dataset()

    # 6. Search for existing checkpoints in the logdir. If checkpoint exists, load data from from checkpoint path and initialize starting step
    checkpoint_path = os.path.join(args.logdir, "world_model_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        start_step = load_model(world_model, checkpoint_path, device=config.device)
    else:
        start_step = 0

    # Create tensorboard writer that use start step as initial step and loads logs from logdir.
    writer = Logger(logdir=args.logdir, initial_step=start_step)

    # 7. Training loop
    for step in range(start_step, start_step+args.train_steps):    
        # 7.1 sample batch from replay buffer as training data

        # Options 
        # 1. Train on full length of trajectory
        #training_batch = replayBuffer_train.sample_batch_full_episode()

        # 2. Train on a specific sequence time. Enable subsampling if needed
        training_batch = replayBuffer_train._sample_batch_by_sequence_time(
            sequence_time= config.sequence_time,
            #subsampling_enabled=True
        )

        # 7.2 Train world model on batch
        _, _, metrics = world_model._train(training_batch)

        # 7.3 Log training metrics
        model_loss = metrics["model_loss"]
        kl = metrics["kl"]
        dyn_loss = metrics["dyn_loss"].mean()
        rep_loss = metrics["rep_loss"].mean()
        likelihood = metrics["obs_loss"].mean()
        reward_loss = metrics["reward_loss"].mean()

        print(f"Step: {step}: Model Loss: {model_loss}, KL: {kl}, KL Path: {dyn_loss}, Initial KL Loss: {rep_loss} , Likelihood Loss: {likelihood}, Reward Loss: {reward_loss}")

        # 7.4 Write logs to tensorboard
        writer.log_scalar_to_tensor_board("world_model/model_loss", model_loss, step)
        writer.log_scalar_to_tensor_board("world_model/kl", kl, step)
        writer.log_scalar_to_tensor_board("world_model/kl_path", dyn_loss, step)
        writer.log_scalar_to_tensor_board("world_model/initial_kl", rep_loss, step)
        writer.log_scalar_to_tensor_board("world_model/likelihood_loss", likelihood, step)

        # 7.5 Write logs to csv file
        writer.log_to_csv("world_model/model_loss", model_loss, step)
        writer.log_to_csv("world_model/kl", kl, step)
        writer.log_to_csv("world_model/kl_path", dyn_loss, step)
        writer.log_to_csv("world_model/initial_kl", rep_loss, step)
        writer.log_to_csv("world_model/likelihood_loss", likelihood, step)

        # 8. Evaluation loop and save model checkpoint
        if step % args.eval_interval == 0 and step > 0:  # Evaluate world model every eval_interval steps
            # 8.1 Save model checkpoint
            save_model(world_model, step, checkpoint_path)

            world_model.eval()

            with torch.no_grad():
                # Evaluate on validation buffer

                # Options 
                # 1. Evaluate on full length of trajectory
                #evaluation_batch = replayBuffer_eval.sample_batch_full_episode()

                # 2. Evaluate a specific sequence time. Enable subsampling if needed
                evaluation_batch = replayBuffer_eval._sample_batch_by_sequence_time(
                   sequence_time = config.sequence_time,
                    #subsampling_enabled=False
                )
                

                # 8.2 Preprocess evaluation batch and encode observations
                evaluation_batch = world_model.preprocess(evaluation_batch)
                embed = world_model.encoder(evaluation_batch)

                B, T = embed.shape[:2]

                # get data from evaluation batch
                actions = evaluation_batch["action"]
                is_first = evaluation_batch["is_first"]
                times = evaluation_batch["time"]
                truth = evaluation_batch["obs"]
                truth_reward = evaluation_batch["reward"]

                #  8.3 Compute posterior over the entire sequence using observe method from dynamics model
                if config.use_sde:
                    posterior, _ = world_model.dynamics.observe(embed=embed, 
                                                           action=actions, 
                                                           dt=config.dt,
                                                           times=times)
                else:
                    posterior, _ = world_model.dynamics.observe(embed=embed, action=actions, is_first=is_first)

                # 8.3.1 Predict rewards from posterior
                reward_head = world_model.heads["reward"]
                reward_posterior = reward_head(
                    world_model.dynamics.get_feat(posterior)
                ).mode() 

                posterior_latents = world_model.dynamics.get_feat(posterior)

                # 8.3.2 Decode posterior 
                decoder = world_model.heads["decoder"]
                reconstruction_posterior = decoder(world_model.dynamics.get_feat(posterior))["obs"].mode()

                # Use initial state from posterior for prior imagination
                init = {k: v[:, 0] for k, v in posterior.items()} 
                steps = actions #[:, 1:]

                # 8.4 Compute prior over the entire sequence
                if config.use_sde:
                    prior = world_model.dynamics.imagine_with_action(steps, init, times, dt=config.dt)
                else:
                    prior = world_model.dynamics.imagine_with_action(steps, init)

                # 8.4.1 Predict rewards from prior
                reward_prior = reward_head(
                    world_model.dynamics.get_feat(prior)
                ).mode() 

                prior_latents = world_model.dynamics.get_feat(prior)
                
                # Perform PCA transformation on posterior and prior path
                PCA_transformation(posterior_latents, prior_latents, truth_reward, args.logdir)


                # 8.4.2 Decode prior
                reconstruction_prior = decoder(world_model.dynamics.get_feat(prior))["obs"].mode()
            
                # 8.5 Compute MSE between reconstructed posterior/prior and ground truth
                mse_post_true = torch.mean((reconstruction_posterior - truth) ** 2).item()
                mse_prior_true = torch.mean((reconstruction_prior - truth[:, :]) ** 2).item()
                
                print(f"MSE: Posterior vs Truth: {mse_post_true:.4e}")
                print(f"MSE: Prior vs Truth:     {mse_prior_true:.4e}")

                # 8.6 Compute posterior over the entire sequence using obs_step method from dynamics model.
                # This method processes one time step at a time and is used during online interaction with the environment.
                
                posterior_per_step_mse = []
                posterior_per_step_state = None
                reconstruction_posterior_per_step_history = []  
                posterior_per_step_mse_history = []

                time_elapsed = float(times[0, 0]) 
                next_reset_time = time_elapsed + config.sequence_time
                for t in range(0,T):
                    if time_elapsed > next_reset_time:
                        print(f"Reset state")
                        posterior_per_step_state = None
                        next_reset_time += config.sequence_time

                    # get data at step t
                    embed_t   = embed[:, t]        
                    is_first_t= is_first[:, t]       
                    prev_action = actions[:, t]
                    time_t = times[:, t]   
                    
                    if config.use_sde:
                        posterior_per_step_state = world_model.dynamics.obs_step(
                            prev_state = posterior_per_step_state,
                            prev_action = prev_action,
                            embed= embed_t,
                            is_first = is_first_t,
                            current_time = time_t,
                            dt = config.dt
                        )
                    else:
                        posterior_per_step_state, _ = world_model.dynamics.obs_step(
                            posterior_per_step_state, prev_action, embed_t, is_first_t
                        )

                    truth_t = truth[:, t]
                    time_elapsed += float(time_t[0]) - time_elapsed

                    print(f"Time elapsed: {time_elapsed}")
                
                    reconstruction_posterior_per_step = decoder(world_model.dynamics.get_feat(posterior_per_step_state))["obs"].mode()
                    r = reward_head(
                            world_model.dynamics.get_feat(posterior_per_step_state)
                        ).mode() 
                    posterior_per_step_mse_history.append(r[0].detach().cpu())
                    posterior_per_step_mse.append(torch.mean((reconstruction_posterior_per_step - truth_t) ** 2).item())                    
                    reconstruction_posterior_per_step_history.append(reconstruction_posterior_per_step[0].detach().cpu())

                posterior_per_step_mse = float(np.mean(posterior_per_step_mse))

                print(f"MSE: Posterior per step vs Truth: {posterior_per_step_mse:.4e}")

                # 8.7 Compute prior per step over the entire sequence using img_step method from dynamics model.
                # This method processes one time step at a time and is used during planning process in dreamer.
                prior_per_step_state = init
                prior_per_step_mse = []
                reward_prior_per_step_history = []
                reconstruction_prior_per_step_history = []

                for t in range(T - 1):
                    # use action at t+1 to predict state at t+1 since a_{t} corresponds to the action performed at observation o_{t-1}
                    action_t = actions[:, t + 1] 
                    imagination_time = (times[:, t + 1] - times[:, t])[0] 
                    #print(times)
                    if config.use_sde:
                        prior_per_step_state = world_model.dynamics.img_step(prior_per_step_state, action_t, imagination_time=imagination_time, dt=config.dt)
                    else:
                        prior_per_step_state = world_model.dynamics.img_step(prior_per_step_state, action_t)

                    reconstruction_prior_per_step = decoder(world_model.dynamics.get_feat(prior_per_step_state))["obs"].mode()
                    r = reward_head(
                            world_model.dynamics.get_feat(prior_per_step_state)
                        ).mode() 
                    reward_prior_per_step_history.append(r[0].detach().cpu())
                    truth_t = truth[:, t + 1]

                    prior_per_step_mse.append(torch.mean((reconstruction_prior_per_step - truth_t) ** 2).item())
                    reconstruction_prior_per_step_history.append(reconstruction_prior_per_step[0].detach().cpu())

                prior_per_step_mse = float(np.mean(prior_per_step_mse)) 
                print(f"Prior per step vs Truth:     {prior_per_step_mse:.4e}")
                
                # 8.8 Log to Tensorboard
                writer.log_scalar_to_tensor_board("eval/mse_post_true", mse_post_true, step)
                writer.log_scalar_to_tensor_board("eval/obs_step_post_mse", posterior_per_step_mse, step)
                writer.log_scalar_to_tensor_board("eval/img_step_mse", prior_per_step_mse, step)

                # 8.9 Log to CSV
                writer.log_to_csv("eval/mse_post_true", mse_post_true, step)
                writer.log_to_csv("eval/obs_step_post_mse", posterior_per_step_mse, step)
                writer.log_to_csv("eval/img_step_mse", prior_per_step_mse, step)

                # 8.10 Visualization of results
                reconstruction_posterior_per_step_curve_zero = torch.stack(reconstruction_posterior_per_step_history, dim=0)
                reconstruction_prior_per_step_curve_zero = torch.stack(reconstruction_prior_per_step_history, dim=0)

                reconstruction_posterior_per_step_curve_zero  = reconstruction_posterior_per_step_curve_zero.numpy() 
                reconstruction_prior_per_step_curve_zero  = reconstruction_prior_per_step_curve_zero.numpy()      

                # Detach arrays
                truth_np   = truth[0].detach().cpu().numpy()
                reconstruction_posterior_curve_zero = reconstruction_posterior[0].detach().cpu().numpy()
                reconstruction_prior_curve_zero = reconstruction_prior[0].detach().cpu().numpy()
                reward_truth_np = truth_reward[0].detach().cpu().numpy()
                reward_post_np  = reward_posterior[0].detach().cpu().numpy().squeeze(-1) 
                reward_prior_np = reward_prior[0].detach().cpu().numpy().squeeze(-1) 

                # Preprocess reward lists
                reward_post_step_np  = torch.stack(posterior_per_step_mse_history).numpy().squeeze(-1) 
                reward_prior_step_np = torch.stack(reward_prior_per_step_history).numpy().squeeze(-1) 


                obs_dim = truth.shape[-1]
                dims_to_plot = min(3, obs_dim)

                # Detach time array
                t_full = evaluation_batch["time"][0].cpu() 

                # Create figure for plots
                fig, axes = plt.subplots(dims_to_plot, 1, figsize=(14, 18))
                fig.suptitle(
                    "Reconstruction of Posterior and Prior",
                    fontsize=16
                )

                if dims_to_plot == 1:
                    axes = [axes]

                for i in range(dims_to_plot):
                    ax = axes[i]
                    
                    # Plot Ground truth data
                    ax.plot(t_full, truth_np[:, i], label="Truth", color="black", linewidth=2)
                    ax.plot(t_full, reconstruction_posterior_curve_zero[:, i], label="Posterior", color="green")
                    ax.plot(t_full, reconstruction_prior_curve_zero[:, i], label="Prior", color="red")
                    ax.plot(t_full, reconstruction_posterior_per_step_curve_zero[:, i], label="Posterior (per step)", color="green", alpha=0.8)
                    
                    # Set ticks at all positions
                    ax.set_xticks(t_full)
                    labels = [""] * len(t_full)
                    labels[0] = f"{t_full[0]:.2f}"
                    labels[-1] = f"{t_full[-1]:.2f}"
                    ax.set_xticklabels(labels)

                    prior_nan_pad = np.concatenate(([np.nan], reconstruction_prior_per_step_curve_zero[:, i])) 
                    ax.plot(t_full, prior_nan_pad, label="Prior (per step)", color="purple", alpha=0.8)

                    ax.set_title(f"Observation Dimension {i}")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Observation Value")
                    ax.grid(True)
                    ax.legend(fontsize=8)

                fig.tight_layout(rect=[0, 0, 1, 0.98])

                # Visualize 
                if args.visualize:
                    plt.show()
                else: # Save
                    vis_dir = os.path.join(args.logdir, "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f"eval_step_{step}_compare.png")
                    fig.savefig(vis_path)
                    plt.close(fig)

                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                ax.set_title("Reward: Truth vs Posterior vs Post. per step vs Prior per step")

                # Ground truth
                ax.plot(t_full, reward_truth_np, label=f"Reward Truth: Total sum={np.sum(reward_truth_np)}", color="black", linewidth=2)
                # Observe (posterior full sequence)
                ax.plot(t_full, reward_post_np, label=f"Reward Posterior: Total sum={np.sum(reward_post_np)}", color="green")
                # Posterior per step
                ax.plot(t_full, reward_post_step_np, label=f"Reward Posterior per step:  Total sum={np.sum(reward_post_step_np)}", color="blue", alpha=0.8)
                # Imagination per step
                reward_prior_step_nan_pad= np.concatenate(([np.nan], reward_prior_step_np))
                ax.plot(t_full, reward_prior_step_nan_pad, label=f"Reward Prior per step: Total sum={np.sum(reward_prior_step_np)}", color="purple", alpha=0.8)

                ax.set_xlabel("Time")
                ax.set_ylabel("Reward")
                ax.grid(True)
                ax.legend()

                ax.set_xlabel("Time")
                ax.set_ylabel("Reward")
                ax.grid(True)
                ax.legend()

                fig.tight_layout()

                if args.visualize:
                    plt.show()
                else:
                    reward_vis_path = os.path.join(vis_dir, f"eval_step_{step}_reward.png")
                    fig.savefig(reward_vis_path)
                    plt.close(fig)

        
