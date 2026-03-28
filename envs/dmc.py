import gym
import numpy as np


class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        self.time = 0.0
        self.mujoco_dt = float(self._env.physics.model.opt.timestep)  # seconds per mujoco substep
        print(f"mujoco_dt = {self.mujoco_dt}")

        # seconds per dm_control env.step(action)
        if hasattr(self._env, "control_timestep"):
            self.control_dt = float(self._env.control_timestep())
        else:
            # fallback (rare): assume 1 mujoco step per env.step
            self.control_dt = self.mujoco_dt

        print(f"control_dt from env = {self.control_dt}")

        # number of mujoco substeps per env.step
        self.control_steps = int(round(self.control_dt / self.mujoco_dt))
        self.control_steps = max(1, self.control_steps)
        print(f"control_dt = {self.control_dt}, control_steps = {self.control_steps}")
        # dt per dm_control env.step (should equal control_dt)
        self.dt_env = self.mujoco_dt * self.control_steps

        # dt per *your wrapper* step (because you repeat env.step)
        self.dt = self.dt_env * self._action_repeat
        print(f"dt = {self.dt}")

    def set_physical_dt(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"physical dt must be positive, got {value}")

        action_repeat = value / self.dt_env
        rounded_action_repeat = int(round(action_repeat))
        if not np.isclose(action_repeat, rounded_action_repeat, rtol=0.0, atol=1e-8):
            raise ValueError(
                "Requested physical dt "
                f"{value} is not an integer multiple of the native DMC control dt "
                f"{self.dt_env}."
            )

        self._action_repeat = max(1, rounded_action_repeat)
        self.dt = self.dt_env * self._action_repeat
        print(
            "Updated DMC physical dt to "
            f"{self.dt} using action_repeat={self._action_repeat}"
        )

    def get_physical_dt(self):
        return self.dt

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        spaces["time"] = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(), dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break

        self.time += self.dt
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["time"] = self.time
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self.time = 0.0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["time"] = self.time
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
