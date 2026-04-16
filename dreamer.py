"""Dreamer main script

This script is the core script to run the agent. The dreamer agent class and the main function are contained
in this file. 

Code modifications to this file:
- (Optional) Replacement of original data set by new replay buffer for effienct handling of irregular data
- Reset handling of the latent state for interaction of the agent with the environment if latent SDE model ist used
All code modifcations are commented and can be found by searching after "Code modification"
"""
import argparse
import functools
import os
import pathlib
import sys
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import re
import shutil
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

yaml = YAML(typ="safe", pure=True)

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
from replaybuffer import ReplayBuffer

to_np = lambda x: x.detach().cpu().numpy()

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset:ReplayBuffer):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, sequence_count_step, reset_time, state=None, training=False):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                # Code modification: 
                # Check which data set is used
                if self._config.use_replay_buffer:
                    data = self._dataset._sample_batch_by_sequence_time(
                        self._config.sequence_time,
                        subsampling_enabled=self._config.subsampling_enabled,
                        subsampling_time_gap_min=self._config.subsampling_time_gap_min,
                        subsampling_time_gap_max=self._config.subsampling_time_gap_max,
                    )
                else:
                    data = next(self._dataset)

                self._train(data)

                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                #d Code modification: 
                # In the irregular setting. The observation array is inhomegenous (Each element in the
                # array has a different shape). Therefore, a checker needs to be implemented first
                # Apply the check when iterating over each metric
                def values_type(values):
                    if np.isscalar(values):
                        return "scalar"
                    if all(np.isscalar(x) for x in values):
                        return "floats"
                    if all(isinstance(x, np.ndarray) for x in values):
                        return "arrays"
                    return "mixed"
                
                def is_inhomogeneous(list_of_arrays):
                    return len({a.shape for a in list_of_arrays}) > 1
                
                for name, values in self._metrics.items():
                    # Check value type
                    if values_type(values) =="arrays":
                        if is_inhomogeneous(values):
                            values = np.concatenate([e.flatten() for e in values])

                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []

                if self._config.video_pred_log:
                    if self._config.use_replay_buffer:
                        openl = self._wm.video_pred(self._dataset.sample_batch_full_trajectory(), self._config.traindir, self._step)
                    else:
                        openl = self._wm.video_pred(next(self._dataset))
                        self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state, reset_time = self._policy(obs, state, training, sequence_count_step, reset_time)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state, reset_time

    def _policy(self, obs, state, training, sequence_count_step, reset_time):
        # Code modification:
        # Reset handling: Since euler solver for SDE integration is not stable for long horizon. A reset needs
        # to be performed at the end of each sequence window defined by 
        # - batch_length for original dataset
        # - reset_time for time-based rollouts
        # The variable sequence_count_step is either a time or a step counter
        if self._config.use_sde and reset_time is not None and reset_time <= 0.0:
            reset_time = self._config.sequence_time

        if state is None: 
            latent = action = None
        elif self._config.use_sde and reset_time is not None and sequence_count_step >= reset_time:
            # print("Resetting state", "at time: ", sequence_count_step)
            latent = action = None
            reset_time += self._config.sequence_time
        elif self._config.use_sde and sequence_count_step % self._config.batch_length == 0:
            # print("Resetting state", "at step", sequence_count_step)
            latent = action = None
        else:
            latent, action = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)

        # Code modification:
        # obs_step method of latent SDE model includes dt and current time of observation as parameters
        if self._config.use_sde:
            latent = self._wm.dynamics.obs_step(
                latent,
                action,
                embed,
                obs["is_first"],
                dt=self._config.dt_wm,
                current_time=obs["time"],
            )
        else:
            latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]

        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state, reset_time

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

# Code modification:
# Step counter for new replay buffer
def count_steps_new_replay_buffer(folder: str) -> int:
    """Count current environment steps of model

    Computes the current environment step count based on all batch files
    with the format batch_<steps>_xxxxxx.npz. 
    Steps means the sum of all trajectory lengths

    Parameters:
    -----------
    folder: str
        path to directory where batch files are stored

    Returns:
    --------
    steps: int
        Environment step count of model
    """

    # Define pattern for batch file
    pattern = re.compile(r"batch_(\d+)_\d+\.npz")
    steps = 0

    # Iterate over each file in directory
    for file in os.listdir(folder):
        match = pattern.match(file)
        if match:
            # Add steps to the current step count
            steps += int(match.group(1))

    return steps

# Code modification:
# Step counter for old dataset
def count_steps_old_dataset(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    print(f"Make env {id} ({mode}): {suite}_{task}")
    print("suite", suite)
    print("task", task)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        if mode == "train":
            env.set_physical_dt(config.dt_env_train)
            print("Physical dt of the environment: ", env.get_physical_dt())
        elif mode == "eval":
            env.set_physical_dt(config.dt_env_eval)
            print("Physical dt of the environment: ", env.get_physical_dt())
        else:
            raise ValueError(f"Invalid mode: {mode}")
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    # Code modification:
    # Add gym environment to the environment lists
    elif suite == "gym":
        from envs.gymnasium import GymEnv

        if mode == "train":
            time_limit = config.time_limit_train
            action_hold_min = config.action_hold_min_train
            action_hold_max = config.action_hold_max_train
        elif mode == "eval":
            time_limit = config.time_limit_eval
            action_hold_min = config.action_hold_min_eval
            action_hold_max = config.action_hold_max_eval
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
        env = GymEnv(name=task, 
                     action_repeat=config.action_repeat, 
                     time_limit= config.time_limit_train if "train" in mode else config.time_limit_eval,
                     irregular=config.irregular,
                     action_hold_min= action_hold_min,
                     action_hold_max= action_hold_max,
                     seed = config.seed
        )
        if mode == "train":
            env.set_physical_dt(config.dt_env_train)
        elif mode == "eval":
            env.set_physical_dt(config.dt_env_eval)
        else:
            # raise error if mode is not train or eval
            raise ValueError(f"Invalid mode: {mode}")
        
        print("Physical dt of the environment: ", env.get_physical_dt())    

        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)

    # Check that TimeLimit wrapper is not used for the gym environment setting because
    # the time limit wrapper is defined by a time limit in terms 
    # of number of steps but gym environment uses a time limit in terms of seconds
    if not config.use_replay_buffer:
        if suite == "dmc":
            if mode == "train":
                time_limit = int(np.ceil(config.time_limit_train / env.get_physical_dt()))
            elif mode == "eval":
                time_limit = int(np.ceil(config.time_limit_eval / env.get_physical_dt()))
            else:
                raise ValueError(f"Invalid mode: {mode}")
            env = wrappers.TimeLimit(env, time_limit)
        elif suite != "gym":
            env = wrappers.TimeLimit(env, config.time_limit)
        
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env, prefix=f"{mode}-{id}")
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

def main(config):
    print("Main function.--------------------------------")
    suite, _ = config.task.split("_", 1)
    if config.use_replay_buffer and config.envs != 1:
        raise ValueError(
            "The new replay buffer currently only supports envs=1. "
            "Multi-environment support requires aligned per-environment time grids "
            "and reset handling."
        )
    if not config.use_replay_buffer and getattr(config, "irregular", False):
        raise ValueError(
            "The old dataset path currently requires irregular=False. "
            "Use the new replay buffer for irregular time grids."
        )
    if (
        config.use_sde
        and not config.use_replay_buffer
        and suite in ("dmc", "gym")
    ):
        expected_sequence_time = config.batch_length * config.dt_env_train
        if not np.isclose(config.sequence_time, expected_sequence_time):
            raise ValueError(
                "For regular time-based environments without replay buffer, "
                "sequence_time must match batch_length * dt_env_train. "
                f"Got sequence_time={config.sequence_time}, "
                f"batch_length={config.batch_length}, "
                f"dt_env_train={config.dt_env_train}, "
                f"expected {expected_sequence_time}."
            )

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    # Code modification:
    # Copy config file to log directory
    source_file = Path("configs.yaml")
    destination_file = Path(config.logdir) / "configs.yaml"
    shutil.copy(source_file, destination_file)

    # Code modification:
    # Determine number of environment steps 
    if config.use_replay_buffer:
        step = count_steps_new_replay_buffer(config.traindir)
    else:
        step = count_steps_old_dataset(config.traindir)
    
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")

    # Code modification
    # Initialize new replay buffer or old dataset
    if config.use_replay_buffer:
        # Create replay buffers 
        replay_buffer_train = ReplayBuffer(directory=config.traindir,
                                        batch_size=config.batch_size,
                                        capacity= config.capacity)
        replay_buffer_train.load_dataset()
    else:
        if config.offline_traindir:
            directory = config.offline_traindir.format(**vars(config))
        else:
            directory = config.traindir
        train_eps = tools.load_episodes(directory, limit=config.dataset_size)
        if config.offline_evaldir:
            directory = config.offline_evaldir.format(**vars(config))
        else:
            directory = config.evaldir
        eval_eps = tools.load_episodes(directory, limit=1)

    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space

    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(config.num_actions, "actions" if hasattr(acts, "n") else "action dims")
    print("Observation Space", train_envs[0].observation_space)

    print("Action Space", acts.low, acts.high)
    state = None
    if not config.offline_traindir:
        # Code modification:
        # Prefill depends on type of dataset
        if config.use_replay_buffer:
            prefill = max(0, config.prefill - count_steps_new_replay_buffer(config.traindir))
        else:
            prefill = max(0, config.prefill - count_steps_old_dataset(config.traindir))
        
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        # Code modification:
        # Due to reset handling, additional (unsused) parameters need to be included to the method
        def random_agent(obs, done, current_time, reset_time, state, training):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None, 0.0

        # Code modification:
        # Parse different arguments to the simulate function dependend on which dataset is used
        if config.use_replay_buffer:
            state = tools.simulate_with_new_replaybuffer(
                random_agent,
                train_envs,
                replay_buffer_train, # different
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=prefill,
            )
        else:
            state = tools.simulate_with_old_dataset(
                random_agent,
                train_envs,
                train_eps, # different
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=prefill,
            )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")

    # Code modification:
    # Parse dataset to the Dreamer agent
    if config.use_replay_buffer:
        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            replay_buffer_train, # different
        ).to(config.device)
    else:
        train_dataset = make_dataset(train_eps, config)
        eval_dataset = make_dataset(eval_eps, config)

        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            train_dataset, # different
        ).to(config.device)


    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    
    print("Training agent.")
    # make sure eval will be executed once after config.steps
    print("Steps", agent._step, config.steps, config.eval_every)
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)

            # Code modification:
            # Parse different arguments to the simulate function dependend on which dataset is used
            if config.use_replay_buffer:
                tools.simulate_with_new_replaybuffer(
                    eval_policy,
                    eval_envs,
                    None,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
            else:
                tools.simulate_with_old_dataset(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
            if config.video_pred_log and not config.use_replay_buffer:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        print("Start training.")

        # Code modification:
        # Parse different arguments to the simulate function dependend on which dataset is used
        if config.use_replay_buffer:
            state = tools.simulate_with_new_replaybuffer(
                agent,
                train_envs,
                replay_buffer_train,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            ) 
        else:
            state = tools.simulate_with_old_dataset(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    print("Command Line Args:  ", args, remaining)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    print("Available configs: ", list(configs.keys()))

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
    print("Parsed config-------------------------------------------")
    main(parser.parse_args(remaining))
