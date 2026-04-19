import gymnasium as gym
from gymnasium import spaces 
import numpy as np
import matplotlib.pyplot as plt

class GymEnv(gym.Env):
    """
    This class is used to define a custom environment compatible with the OpenAI Gymnasium library.
    The environment is assumed to be continuous meaning that the termination is specified by a time limit.
    Furthermore the environment allows to run an episode on a non-uniform time grid by using action holds which keep
    an action constant for a specific number of steps.

    Attributes:
    -----------
    _name: str
        name of OpenAI environment (e.g. Pendulum-v1
    _env: gym.Env
        original environment    
    _repeat: int
        number of action repeats
    _time_limit: float
        duration of an episode in the environment
    _time: float 
        current time in environment
    _use_action_hold: bool
        states if action holds are enabled or not
    _action_hold_grid: 
        saves the current grid of action holds 
    _action_hold_grid_idx:
        identifies current action hold in the environment 
    _physical_step_size:
        stores frequency of environment
    _action_hold_min:
        minimum number of action holds 
    _action_hold_max:
        maxmimum number of action holds

    Methods:
    --------
    set_physical_dt:
        set frequency of environment
    get_physical_dt:
        get frequency of environment
    get_current_time:
        get current time in environment
    make_time_grid:
        reconfigure action hold grid of environment
    observation_space:
        observation space of environment
    action_space:
        action space of environment that can be used to sample random actions
    _wrap_obs:
        wrap observation with additional data such as the flags is_first and is_terminal
        and the current time of the observation
    configure_action_holds:
        configure the min and max number of action holds
    reset:
        reset environment
    step:
        perform an action in the environment.
    close:
        close the environment
    """

    def __init__(self, 
                 name: str, 
                 action_repeat: int = 1, 
                 time_limit: float = 10.0, 
                 seed: int = None, 
                 use_action_hold: bool = False,
                 action_hold_min: int = 1,
                 action_hold_max: int = 2,
                 observation_gap_min: int = 1,
                 observation_gap_max: int = 1
                 ):
        """Docstring for __init__
    
        Parameters:
        name: str
            name of environment (e.g. Pendulum-v1)
        action_repeat: int
            number of action repeats in the environment (typically 1)
        time_limit: float
            duration of an episode in the environment specified in seconds
        seed: 
            seed id to allow deterministic episodes
        use_action_hold:
            states if action holds are enabled or not
        action_hold_min:
            minimum number of action holds
        action_hold_max:
            maxmimum number of action holds

        Note: The value of action holds specifies how long (:=number of steps in the environment) 
        an action is kept constant  
        """

        self._name = name
        # Initialize original environment 
        self._env = gym.make(name)

        self.seed = seed

        if seed is not None:
            self._env.reset(seed=seed)
        

        self._repeat = action_repeat 
        self._time_limit = time_limit 
        self._time = 0.0  
        self._use_action_hold = use_action_hold

        self._action_hold_grid = None 
        self._action_hold_grid_idx = 0 
        self._action_hold_min = action_hold_min
        self._action_hold_max = action_hold_max
        self._observation_gap_grid = None
        self._observation_gap_grid_idx = 0
        self._observation_gap_min = observation_gap_min
        self._observation_gap_max = observation_gap_max
        self._steps_until_next_observation = 1

        # Check which environment is used since the frequency is defined differently for every environment
        if self._name == "Pendulum-v1":
            self._physical_step_size = self._env.unwrapped.dt 
        # add more environments ...

    def set_physical_dt(self, value: float):
        """Set frequency of environment

        Parameters:
        -----------
        value: float
            frequency 
        """

        # Check which environment is used since the frequency is defined differently for every environment
        if self._name == "Pendulum-v1":
            self._env.unwrapped.dt = value
            self._physical_step_size = self._env.unwrapped.dt
        # add more environments ...
    
    def get_physical_dt(self):
        """Get frequency of environment

        Returns:
        -----------
        physical_dt: float
            frequency 
        """

        return self._physical_step_size
    
    def get_current_time(self):
        """Get current time in environment
        Returns:
        --------
        time: float
        """

        return self._time

    def make_time_grid(self):
        """Reconfigure action and observation gap grids."""
        if self._use_action_hold:
            self._action_hold_grid = self._make_hold_grid(
                self._action_hold_min,
                self._action_hold_max,
            )
        else:
            self._action_hold_grid = None

        if self._observation_gap_min > 1 or self._observation_gap_max > 1:
            self._observation_gap_grid = self._make_hold_grid(
                self._observation_gap_min,
                self._observation_gap_max,
            )
        else:
            self._observation_gap_grid = None

    def _make_hold_grid(self, hold_min: int, hold_max: int):
        hold_min = max(int(hold_min), 1)
        hold_max = max(int(hold_max), hold_min)
        max_steps = int(np.ceil(self._time_limit / (hold_min * self._physical_step_size))) + 1
        holds = np.random.randint(hold_min, hold_max + 1, size=max_steps)
        cumulative_time = np.cumsum(holds * self._physical_step_size)
        cutoff = np.searchsorted(cumulative_time, self._time_limit, side="right")
        return holds[:cutoff + 1]

    def _schedule_next_observation(self):
        if self._observation_gap_grid is None:
            self._steps_until_next_observation = 1
            return

        idx = min(self._observation_gap_grid_idx, len(self._observation_gap_grid) - 1)
        self._steps_until_next_observation = int(self._observation_gap_grid[idx])
        self._observation_gap_grid_idx += 1

    @property
    def observation_space(self):
        """Returns observation space of the environment
        """

        obs_space = self._env.observation_space
        
        # Dreamer expects a dictionary for the observation space
        return spaces.Dict({
            "obs": obs_space, 
            "image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),  # dummy image space to be compatible with Dreamer
            "obs_valid": spaces.Box(0, 1, (), dtype=np.bool_),
        })
    
    @property
    def action_space(self):
        """Returns action space of the environment
        """
        return self._env.action_space

    def _wrap_obs(
        self,
        obs: np.ndarray,
        is_first: bool = False,
        is_terminal: bool = False,
        obs_valid: bool = True,
    ):
        """wrap observation with additional data such as the flags is_first and is_terminal
        and the current time of the observation

        Parameters:
        -----------
        obs: np.ndarray:
            original observation of environment 
        is_first: bool
            flag if observation is the first of the episode 
        is_terminal:
            flag if observation is the end of the episode
        
        Returns:
            observation dictionary containing of all observations in the environment
        """
        
        # If obs is a vector, create a dummy observation for logging
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            image = obs

        # Return wrapped observation
        return {
            "obs": obs.astype(np.float32),
            "image": image,
            "is_first": is_first,
            "is_terminal": is_terminal,
            "time": self._time,
            "obs_valid": np.bool_(obs_valid),
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment. 
        
        Since class inherits from Gym.Env class, seed and options need to be added as parameters but
        are not required.
    
        Parameters:
        -----------
            seed: int
                seed id to allow deterministic runs 
            options: dict
                additional info about the reset
        Returns:
        --------
            wrapped_observation:
                wrapped observation with is_first flag set to True
        """

        # Create fresh hold grids for each new episode.
        self.make_time_grid()

        # If seed is given, parse to reset function
        if self.seed is not None:
            obs, _ = self._env.reset(seed=self.seed)
        else:
            obs, _ = self._env.reset()
        
        # Reset time
        self._time= 0

        # Reset index identifier of dt grid
        self._action_hold_grid_idx = 0
        self._observation_gap_grid_idx = 0
        self._schedule_next_observation()

        # Wrap observation and set is_first flag to True
        wrapped_observation = self._wrap_obs(
            obs,
            is_first=True,
            is_terminal=False,
            obs_valid=True,
        )

        return wrapped_observation

    def step(self, action: np.ndarray):
        """Perform a step in the environment. 

        Since action holds are used, a step does not mean one physical step in the environment
        but a number of physical steps specified by the current action hold in the action hold grid
        
        Parameters:
            action: np.adarray
                Action that is to step in the environment
        """

        # Check if action holds are enabled.
        # If yes, use the action hold grid to determine the current action hold
        if self._use_action_hold:
            n = int(self._action_hold_grid[self._action_hold_grid_idx])
            self._action_hold_grid_idx += 1
        else:
            n = 1
        
        # Store the elapsed time 
        elapsed = 0.0
        steps_taken = 0
        # Store the last reward. This variable is used to skip rewards during the stepping process.
        last_reward = 0.0
        
        obs = None
        done = False
        
        # Perform steps n times 
        for _ in range(n):
            # terminated and truncated are disabled. Instead, time limit defines the end of the episode.
            obs, reward, terminated, truncated, info = self._env.step(action)
            
            last_reward = reward
            elapsed += self._physical_step_size 
            steps_taken += 1

            # End episode if time limit was reached
            if self._time + elapsed >= self._time_limit:
                done = True
                break
    
        # Update current time by the elapsed time 
        self._time += elapsed

        obs_valid = True
        if self._observation_gap_grid is not None:
            self._steps_until_next_observation -= steps_taken
            obs_valid = done or self._steps_until_next_observation <= 0
            if obs_valid and not done:
                self._schedule_next_observation()

        agent_reward = float(last_reward) if obs_valid else 0.0

        # Wrap obswervation
        wrapped_obs = self._wrap_obs(
                            obs,
                            is_first=False,
                            is_terminal=done,
                            obs_valid=obs_valid,
                        )
        

        return wrapped_obs, agent_reward, done, info

    def close(self):
        """Close the environment.
        """
        return self._env.close()
