import gymnasium as gym
from gymnasium import spaces 
import numpy as np
import matplotlib.pyplot as plt

class GymEnv(gym.Env):
    """
    This class is used to define a custom environment compatible with the OpenAI Gymnasium library.
    The environment is assumed to be continuous meaning that the termination is specified by a time limit.
    Furthermore the environment allows to run an episode on a non-uniform time grid by using action holds which keep
    an action constant for a specific number of steps => irregular decision times.

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
    _irregular: bool
        states if environment consists of irregular decision times or not
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
                 irregular: bool = False,
                 action_hold_min: int = 1,
                 action_hold_max: int = 2
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
        irregular:
            states if environment consists of irregular decision times or not
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

        if seed is not None:
            self._env.reset(seed=seed)
        

        self._repeat = action_repeat 
        self._time_limit = time_limit 
        self._time = 0.0  
        self._irregular = irregular

        self._action_hold_grid = None 
        self._action_hold_grid_idx = 0 
        self._action_hold_min = action_hold_min
        self._action_hold_max = action_hold_max

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
        """reconfigure action hold grid of environment

        A fixed grid of random action holds is generated and used when stepping in the environment.
        """
    
        # Determine the maximum size of the action hold grid
        max_steps = int(np.ceil(self._time_limit / (self._action_hold_min * self._physical_step_size ))) + 1

        # Generate a list of random action holds with size max_steps
        n = np.random.randint(self._action_hold_min, self._action_hold_max + 1, size=max_steps)

        # Compute the cumulative sum to determine the current time in the grid 
        t = np.cumsum(n * self._physical_step_size)
        
        # Cut off grid at the point where time limit was reached. Include the last action hold too 
        # which exceeds the time limit
        cutoff = np.searchsorted(t, self._time_limit, side="right")

        # store final grid
        self._action_hold_grid = n[:cutoff + 1]

    @property
    def observation_space(self):
        """Returns observation space of the environment
        """

        obs_space = self._env.observation_space
        
        # Dreamer expects a dictionary for the observation space
        return spaces.Dict({
            "obs": obs_space, 
            "image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)  # dummy image space to be compatible with Dreamer
        })
    
    @property
    def action_space(self):
        """Returns action space of the environment
        """
        return self._env.action_space

    def _wrap_obs(self, obs: np.ndarray, is_first: bool = False, is_terminal:bool=False):
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
            "time": self._time
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

        # If action hold grid is None, create an initial grid. This ensures null exceptions
        if self._action_hold_grid is None:
            self.make_time_grid()

        # If seed is given, parse to reset function
        if seed is not None:
            obs, _ = self._env.reset(seed=seed)
        else:
            obs, _ = self._env.reset()
        
        # Reset time
        self._time= 0

        # Reset index identifier of dt grid
        self._action_hold_grid_idx = 0

        # Wrap observation and set is_first flag to True
        wrapped_observation = self._wrap_obs(obs, is_first=True, is_terminal=False)

        return wrapped_observation

    def step(self, action: np.ndarray):
        """Perform a step in the environment. 

        Since action holds are used, a step does not mean one physical step in the environment
        but a number of physical steps specified by the current action hold in the action hold grid
        
        Parameters:
            action: np.adarray
                Action that is to step in the environment
        """

        # Check if irregularity is enabled.
        # If yes, use the action hold grid to determine the current action hold
        if self._irregular:
            n = int(self._action_hold_grid[self._action_hold_grid_idx])
            self._action_hold_grid_idx += 1
        else:
            n = 1
        
        # Store the elapsed time 
        elapsed = 0.0
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

            # End episode if time limit was reached
            if self._time + elapsed >= self._time_limit:
                done = True
                break
    
        # Update current time by the elapsed time 
        self._time += elapsed

        # Wrap obswervation
        wrapped_obs = self._wrap_obs(
                            obs,
                            is_first=False,
                            is_terminal=done,
                        )
        

        return wrapped_obs, last_reward, done, info

    def close(self):
        """Close the environment.
        """
        return self._env.close()