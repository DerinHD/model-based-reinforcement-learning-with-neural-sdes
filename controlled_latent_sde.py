import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from torch.distributions import Normal


class ControlledLatentSDE(nn.Module):
    """
    This class defines a controlled latent neural stochastic differentical equation (SDE)
    as a continuous reformulation of the discrete RSSM model in Dreamer.

    The model consists of posterior and prior drifts and a diagonal diffusion. It also allows to 
    use a policy as a closed-loop controller.

    Attributes:
    -----------
    latent_dim_deter: int
        dimension of deterministic component
    latent_dim_stoch: int
        dimension of stochastic component
    context_dim: int
        dimension of the encodings
    hidden_dim: int
        dimension of the hidden layers
    act_dim: int
        dimension of the action space
    intitial_deter_state: nn.Parameter
        learnable parameter for the initial state of the determinstic component
    initial_prior_distribution: nn.Linear
        linear network for the computation of the initial prior distribution
    initial_posterior_distribution: nn.Linear 
        linear network for the computation of the initial posterior distribution
    deterministic_transition_function: nn.Sequential
        sequential network representing the transition function of the deterministic component
    prior_drift_for_stochastic: nn.Sequential
        sequential network representing the prior drift 
    posterior_drift_for_stochastic: nn.Sequential
        sequential network representing the posterior drift
    diffusion_for_stochastic: nn.Sequential
        sequential network representing the diagonal diffusion
    actions: torch.Tensor
        batch of actions with the shape [T,B, act_dim]:
        T:=number of time steps,
        B:=number of batch elements (sequences)
        act_dim:= dimension of action space
    embedings: torch.Tensor
        batch of embedings of the observation with the shape [T,B, context_dim]:
        T:=number of time steps,
        B:=number of batch elements (sequences)
        context_dim:= dimension of the embedding
    time_steps: torch.tensor
        one dimensional tensor representing the time grid of the batch elements (sequences)
    controller: 
        closed-loop controller
    energy_preserving: bool
        flag whether to use energy preserving drifts or not

    Methods:
    --------
    contextualize:
        When training the models on sequences, store the time steps, embeddings and actions directly in the class 
        since the functions f,h and g do not allow to parse additional arguments.
    set_controller:
        Set closed-loop controller
    h:
        prior drift
    f:
        posterior drift
    g:
        diffusion
    initial:
        method to initialize posterior and prior
    """

    # SDE specific configuration
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        latent_dim_deter: int,
        latent_dim_stoch: int,
        context_dim: int,
        hidden_dim: int,
        act_dim: int,
        energy_preserving_enabled: bool=False
    ) -> None:
        super().__init__()
        """Docstring for __init__
        
        Parameters:
        -----------
        latent_dim_deter: int
        dimension of deterministic component
        latent_dim_stoch: int
            dimension of stochastic component
        context_dim: int
            dimension of the encodings
        hidden_dim: int
            dimension of the hidden layers
        act_dim: int
            dimension of the action space
        """

        self.latent_dim_deter = latent_dim_deter
        self.latent_dim_stoch = latent_dim_stoch
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        # Initialize learnable parameter of the deterministic component with a zero vector
        self.initial_deter_state = nn.Parameter(torch.zeros(latent_dim_deter))

        # The initial prior distribution (mean, logstd) is parameterized by the deterministic state
        self.initial_prior_distribution = nn.Linear(
            self.latent_dim_deter,
            2 * self.latent_dim_stoch,
        )

        # The initial posterior distribution (mean, logstd) is parameterized by the deterministic state and the first observation
        self.initial_posterior_distribution = nn.Linear(
            self.context_dim + self.latent_dim_deter,
            2 * self.latent_dim_stoch,
        )

        # The deterministic transition function depends on the ...
        # - current deterministic state
        # - current stochastic state
        # - action performed at the current latent state
        # and the output is the next deterministic state
        self.deterministic_transition_function = nn.Sequential(
            nn.Linear(latent_dim_deter + latent_dim_stoch + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim_deter),
        )

        # The prior drift for the stochastic component is parameterized by the deterministic state.
        self.prior_drift_for_stochastic = nn.Sequential(
            nn.Linear(latent_dim_deter, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim_stoch),
        )

        # The posterior drift for the stochastic component is parameterized by the deterministic state and the context (next observation)
        self.posterior_drift_for_stochastic = nn.Sequential(
            nn.Linear(latent_dim_deter + context_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim_stoch),
        )

        # The diagonal diffusion for the stochastic component is parameterized by the deterministic state.
        self.diffusion_for_stochastic = nn.Sequential(
            nn.Linear(latent_dim_deter, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim_stoch),
            nn.Sigmoid(),  
        )

        # initialize the contexts (action, embedding and time steps) with None
        self.actions = None        # [T, B, act_dim]
        self.embeddings = None     # [T, B, context_dim]
        self.time_steps = None     # [T]

        # If closed-loop controller is used for imagination, it needs to be assigned 
        self.controller = None
        self.use_stochastic_controller = True
        self.initial_controller_action = None

        self.energy_preserving = energy_preserving_enabled

    def contextualize(self, ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Store current contexts in the class instance.

        The order must be the following and the shapes must be correct:

        time_steps: torch.tensor
            one dimensional tensor representing the time grid of the batch elements (sequences)
        embedings: torch.Tensor
            batch of embedings of the observation with the shape [T,B, context_dim]:
            T:=number of time steps,
            B:=number of batch elements (sequences)
            context_dim:= dimension of the embedding
        actions: torch.Tensor
            batch of actions with the shape [T,B, act_dim]:
            T:=number of time steps,
            B:=number of batch elements (sequences)
            act_dim:= dimension of action space
        
        Parameters:
        -----------
        ctx: tuple
            tuple consisting of the contexts described above
        """
        self.time_steps, self.embeddings, self.actions = ctx

    def set_controller(
        self,
        controller: nn.Module,
        use_stochastic_policy: bool = True,
        initial_action: torch.Tensor = None,
    ):
        """Set closed-loop controller"""
        self.controller = controller
        self.use_stochastic_controller = use_stochastic_policy
        self.initial_controller_action = initial_action

    def h(self, t, y):
        """Prior drift 
        
        Parameters:
        -----------
        t: float
            current time in the integration step
        y: tensor
            current latent state
        Returns:
        --------
            Prior drift vector
        """

        # Split latent state into deterministic and stochastic components
        deter, stoch = torch.split(
            y, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1
        )

        # Check if closed-loop controller is available.
        # If not, use contexts
        if self.controller is not None:
            if self.initial_controller_action is not None:
                action = self.initial_controller_action
                self.initial_controller_action = None
            else:
                policy = self.controller(y)
                if self.use_stochastic_controller:
                    action = policy.sample()  # [B, act_dim]
                else:
                    action = policy.mode()  # [B, act_dim]
        else:
            # Determine the time step for the action that is applied at the latent state y
            # Right index is used for search sorted since Dreamer stores the actions with a shift:
            # a{t} corresponds to the action performed at observation o{t-1}
            time_index = torch.searchsorted(self.time_steps, t, right=True).clamp(
                max=self.time_steps.size(0) - 1
            )
            action = self.actions[time_index]

        if self.energy_preserving:
            # Interpret deterministic_transition_function as an attractor
            deter_next = self.deterministic_transition_function(torch.cat([deter, stoch, action], dim=-1))
            # Subtract attractor by the current deterministic state
            deter_drift = deter_next - deter

            # Interpret prior_drift_for_stochastic as an attractor
            stoch_next_mean = self.prior_drift_for_stochastic(deter_next)
            # Subtract attractor by the current stochastic state
            stoch_drift = stoch_next_mean - stoch
        else:
            deter_drift = self.deterministic_transition_function(torch.cat([deter, stoch, action], dim=-1))
            stoch_drift = self.prior_drift_for_stochastic(deter)

        # concatenate deterministic and stochastic drifts
        return torch.cat([deter_drift, stoch_drift], dim=-1)

    def f(self, t, y):
        """Posterior drift 
        
        Parameters:
        -----------
        t: float
            current time in the integration step
        y: tensor
            current latent state
        Returns:
        --------
            Posterior drift vector
        """
        # Split latent state into deterministic and stochastic components
        deter, stoch = torch.split(
            y, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1
        )

        # Determine the time step for the action that is applied at the latent state y.
        # Right index is used for search sorted since Dreamer stores the actions with a shift:
        # a{t} corresponds to the action performed at observation o{t-1}
        time_index = torch.searchsorted(self.time_steps, t, right=True).clamp(
            max=self.time_steps.size(0) - 1
        )

        # Get action at current time step
        action = self.actions[time_index]
        # Get future observation at current time step (inference)     
        context = self.embeddings[time_index]  
        
        if self.energy_preserving: 
            # Interpret deterministic_transition_function as an attractor
            deter_next = self.deterministic_transition_function(torch.cat([deter, stoch, action], dim=-1))
            # Subtract attractor by the current deterministic state
            deter_drift = deter_next - deter

            # Interpret posterior_drift_for_stochastic_net as an attractor
            stoch_next_mean = self.posterior_drift_for_stochastic(torch.cat([deter_next, context], dim=-1))
            # Subtract attractor by the current stochastic state
            stoch_drift = stoch_next_mean - stoch
        else:
            deter_drift = self.deterministic_transition_function(torch.cat([deter, stoch, action], dim=-1))
            stoch_drift = self.posterior_drift_for_stochastic(torch.cat([deter, context], dim=-1))

        # concatenate deterministic and stochastic drifts   
        return torch.cat([deter_drift, stoch_drift], dim=-1)

    def g(self, t, y):
        """Diffusion
        
        Parameters:
        -----------
        t: float
            current time in the integration step
        y: tensor
            current latent state

        Returns:
        --------
            Diffusion vector 
        """
        # Split latent state into deterministic and stochastic components
        deter, stoch = torch.split(
            y, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1
        )
        
        # diffusion for the stochastic state is parameterized by the current deterministic state
        diffusion_stoch = self.diffusion_for_stochastic(deter)  
        # Theoretically, the diffusion of deter is obviously zero but practically, a small epsilon needs
        # to be applied to avoid division by zero when computing the KL divergence
        diffusion_deter = torch.full_like(deter, 1e-6)   

        # concatenate deterministic and stochastic components
        diffusion_vec = torch.cat([diffusion_deter, diffusion_stoch], dim=-1)
        return diffusion_vec

    def initial(self, batch_size: int, embed0: torch.Tensor):
        """Initialize Posterior and Prior state
        
        Parameters:
        -----------
        batch_size: int
            Number of initial states to produce
        embed_0: torch.Tensor
            Initial observation
        Returns:
        --------
        y0_prior: torch.Tensor
            initial prior state
        y0_post: torch.Tensor
            initial posterior state
        stats0: tuple
            distribution parameters of the initial states:
            - p_mean: prior mean
            - p_std: prior std
            - q_mean: posterior mean
            - q_std: posterior std
        """
        device = embed0.device

        # Generate #batch_size initial deterministic states
        h0 = self.initial_deter_state.unsqueeze(0).expand(batch_size, -1).to(device)  

        # Compute prior distributions using the initial deterministic states
        p_mean, p_logstd = self.initial_prior_distribution(h0).chunk(2, dim=-1) 
        p_std = F.softplus(p_logstd) + 1e-4
        p_dist = Normal(p_mean, p_std)
        # Sample initial stochastic state from prior distribution
        s0_prior = p_dist.rsample() 

        # Compute posterior distributions using the initial deterministic states and the inital observation
        q_inp = torch.cat([h0, embed0], dim=-1) 
        q_mean, q_logstd = self.initial_posterior_distribution(q_inp).chunk(2, dim=-1)
        q_std = F.softplus(q_logstd) + 1e-4
        q_dist = Normal(q_mean, q_std)
        
        # Sample initial stochastic state from posterior distribution
        s0_post = q_dist.rsample()  

        # Concatenate deterministic and stochastic components
        y0_prior = torch.cat([h0, s0_prior], dim=-1)  
        y0_post = torch.cat([h0, s0_post], dim=-1)    

        # Store and return the distribution parameters to compute KL divergence
        stats0 = (p_mean, p_std, q_mean, q_std)
        
        return y0_prior, y0_post, stats0
