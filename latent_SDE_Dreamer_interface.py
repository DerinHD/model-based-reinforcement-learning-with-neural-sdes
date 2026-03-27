import torch
from torch import nn
import torchsde
from torch import distributions as torchd
import tools
from torch.distributions import Normal, kl_divergence


class LatentSDEDreamerInterface(nn.Module):
    """
    This class implements an interface for the class ControlledLatentSDE such that Dreamer can use 
    the continous dynamics of the latent neural SDE.

    More precisely, the WorldModel class replaces the RSSM by this class that acts as an Wrapper for the
    ControlledLatentSDE class and provides the required functions fo training, planning and interaction.

    Schematic illustration how world model and latent SDE model communicate

         world model      <------------ LatentSDEDreamerInterface  ------------> ControlledLatentSDE
                                                observe 
                                                obs_step
                                                img_step
                                            imagine_with_action

    Attributes:
    -----------
    latent_dim: 
        dimension of latent state
    act_dim:
        dimension of action space
    embed_dim:
        dimension of embedding (encoded observation)
    hidden_dim:
        dimension of hidden layers
    device: 
        device on which model runs    
    """
    def __init__(self, deter_dim:int, stoch_dim:int, act_dim:int, embed_dim:int, hidden_dim:int, device="cuda", solver="euler"):
        super().__init__()
        from controlled_latent_sde import ControlledLatentSDE
        """Docstring for __init__
        Parameters:
        -----------
        deter_dim: int
            dimension of deterministic state
        stoch_dim: int
            dimension of stochastic state
        act_dim: int
            dimension of action space
        embed_dim: int
            dimension of embedding
        hidden_dim: int
            dimension of hidden layers
        device:
            device on which model runs
        """

        # Latent dimension is the concatenation of deterministic and stochastic state
        self.latent_dim = deter_dim + stoch_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device
        print(f"Using {solver} solver for SDE integration.")
        self.solver = solver
        
        self.model = ControlledLatentSDE(
            latent_dim_deter=deter_dim,
            latent_dim_stoch=stoch_dim,
            context_dim=embed_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
        )

    def convert_to_dict(self, z):
        return {"latent": z}

    def get_feat(self, state:dict):
        """Return feature vector
    
        Parameters:
        -----------
        state: dict
            latent state as a dictionary
        Returns:
        --------
            latent state vector
        """
        return state["latent"]

    def observe(
        self, 
        embed: torch.Tensor,            
        action: torch.Tensor,              
        times: torch.Tensor,
        dt: float,
        kl_free: float = 1e-3,
    ):
        """Dreamer Interface: observe function.

        This function is used by the world model to train the dynamics by computing the posterior.
        Additionally it returns the KL loss. 

        Parameters:
        -----------
        embed: torch.Tensor
            batch of embeddings
        action: torch.Tensor
            batch of actions
        times: torch.Tensor
            batch of times
        dt: float
            integration step size for the Euler solver
        kl_free: float
            free bits

        Returns:
        --------
        post: dict
            posterior
        kl_metrics: tuple
            KL metrics:
            - kl_loss: KL loss
            - kl_value: KL loss detached
            - kl_path: KL path loss (without initial KL)
            - initial_kl: Initial KL loss
        """
        device = embed.device

        # Dreamer stores batches in the shape B,T,E:
        # - B: Batch dimension
        # - T: Number of time steps
        # - E: Embedding dimension
        B, T, E = embed.shape

        # Convert shape to Latent SDE format
        embed_transposed = embed.transpose(0, 1).contiguous()   # [T,B,E]
        acts_transposed  = action.transpose(0, 1).contiguous()  # [T,B,A]

        # For regularly sampled data, time grids can be different since the environments are time-invariant. 
        # For irregularly sampled data, we will use the new data set structure by the class Replay Buffer where all 
        # sequences of a batch have the same time grid.
        # Therefore, we can just take the first time grid.
        #ts = times[0] 

        ts = torch.arange(T, device=device) * dt

        # Compute initial posterior by the initial method 
        y0_prior, y0_post, (p_mean0, p_std0, q_mean0, q_std0) = self.model.initial(B, embed_transposed[0])

        # Create normal distirbutions from the distribution parameters
        p_dist0 = Normal(p_mean0, p_std0)
        q_dist0 = Normal(q_mean0, q_std0)

        # Compute KL loss between initial prior and posterior
        kl_init = kl_divergence(q_dist0, p_dist0)  
        kl_init = kl_init.sum(-1, keepdim=True)    
        # Clamp initial KL wby free bits
        kl_init = torch.clamp(kl_init, min=kl_free)

        # Store contexts (time grid, embedding and actions)
        self.model.contextualize((ts, embed_transposed, acts_transposed))

        # Perform SDE integration and use Euler solver. Set logqp to True to receive log density ratio between
        # posterior and prior
        y, log_ratio = torchsde.sdeint(
            self.model,
            y0_post.to(device),
            ts,
            method=self.solver,
            dt=dt,
            logqp=True,
            names={"drift": "f", "diffusion": "g"}
        ) 

        # Clamp Kl path by freebits
        kl_path = torch.clip(log_ratio, min=kl_free)                             
        
        # Convert back to Dreamer shape
        y = y.transpose(0, 1).contiguous()   

        # Convert to dictionary and save times to the states
        post = self.convert_to_dict(y)
        post["time"] = times


        initial_kl = kl_init                                                     # [B,1]
        kl_path = kl_path.transpose(0, 1).contiguous()                           # [B,T-1] 
        kl_loss  = torch.hstack((initial_kl, kl_path)) # # stack initial KL => [B,T] 
        kl_value = kl_loss.detach()

        # Store KL metrics
        kl_metrics = (kl_loss, kl_value, kl_path, initial_kl)
        return post, kl_metrics

    def obs_step(self, 
                 prev_state: dict,
                 prev_action: torch.Tensor, 
                 embed: torch.tensor, 
                 is_first: torch.tensor, 
                 dt:float, 
                 current_time: torch.Tensor):
        """Dreamer Interface: obs_step function.

        This function is used by the world model to interact with the environment in the latent space.
        Since parrallel environments are possible the obs_step emthod can receive a batch of episodes

        Parameters:
        prev_state: dict
            previous latent state. 
        prev_actipn: torch.Tensor
            action that was performed at the latent state
        embed: torch.Tensor
            embedding of the current observation
        is_first: torch.Tensor
            flag to check if the current state is a starting state => latent state needs to be initialized
        dt: float
            integration step size for the Euler solver
        current_time: torch.Tensor
            current times of the observations

        Returns:
        --------
        post: dict
            posterior
        --------

        """
        B = embed.shape[0]

        # If the episode starts for the first time, then the previous state does not exists. In this case,
        # the initial state needs to be computed by the initial method from the latent SDE model
        if prev_state is None:
            _, y0_post, _ = self.model.initial(B, embed)
            post = self.convert_to_dict(y0_post)
            post["time"] = current_time

            return post
        

        y_prev_all = self.get_feat(prev_state)
        t_prev_all = prev_state["time"]

        #  Initialize posterior with zeros
        y_post_all = torch.zeros_like(y_prev_all)

        # Iterate over each batch element
        for b in range(B): 
            # Check if is first flag is set =>  latent state needs to be initialized
            if is_first[b]:
                _, y0_post, _ = self.model.initial(1, embed[b:b+1])
                y_post_all[b] = y0_post[0]
                continue # No integration needed for this batch element since only initial state is required

            # Get previous time 
            t_prev = t_prev_all[b]
            # Get current time
            t_curr = current_time[b]

            # Stack times to define the time grid
            #ts = torch.stack([t_prev, t_curr])

            ts = torch.tensor([0, dt], device=embed.device)

            y_prev = y_prev_all[b:b+1]

            # Since a time interval is given, we also need to specify at least two embeddings and actions. 
            # Therefore we just assign the same values to both time steps
            ctx_pair = torch.stack([embed[b], embed[b]], dim=0).unsqueeze(1)  
            act_pair = torch.stack([prev_action[b], prev_action[b]], dim=0).unsqueeze(1)  

            # Store contexts in latent SDE model
            self.model.contextualize((
                ts,
                ctx_pair,
                act_pair,
            ))

            # Perform SDE integration for the time interval
            y = torchsde.sdeint(
                self.model,
                y_prev,
                ts,
                method=self.solver,
                dt=dt,
                names={"drift": "f", "diffusion": "g"},
                logqp=False,
            )

            # only the last step is needed 
            y_post_all[b] = y[1, 0]

        post = self.convert_to_dict(y_post_all)

        # Save current time in state
        post["time"] = current_time

        return post
    
    def img_step(self, 
                 prev_state: dict, 
                 prev_action: torch.Tensor, 
                 imagination_time: float, 
                 imagination_dt:float):
        """Dreamer Interface: img_step function.
        
        This function is used by the world model to perform imagination steps during the rollout.

        Parameters:
        -----------
        prev_state: dict
            previous latent state
        prev_action: torch.Tensor
            action that is performed during the imagination step
        imagination_time: float
            time duration of the imagination step 
        dt: float
            integration step size for the Euler solver

        Returns:
        --------
        prior: dict
            posterior
        """
        device = prev_state["latent"].device
        B = prev_state["latent"].shape[0]
        
        y_prev = prev_state["latent"]
        t_prev = prev_state["time"][0] # Assuming all elements in batch have same time
        t_curr = t_prev + imagination_time

        # Create time grid
        ts = torch.tensor(
            [0.0, imagination_time],
            device=device
        )
        
        # Dummy context since prior is not conditioned on future observations
        ctx_dummy = torch.zeros(2, B, self.embed_dim, device=device)
        acts_pair = torch.stack([prev_action, prev_action], dim=0)

        # Store contexts in latent SDE model
        self.model.contextualize((ts, ctx_dummy, acts_pair))

        # Perform SDE integration for the time interval
        y = torchsde.sdeint(
            self.model,
            y_prev,
            ts,
            method=self.solver,
            dt=imagination_dt,
            names={"drift": "h", "diffusion": "g"},
            logqp=False,
        )

        y_next = y[1]

        prior = self.convert_to_dict(y_next)
        prior["time"] = t_curr.unsqueeze(0).repeat(B)

        return prior

    def imagine_with_action(self, 
                            actions: torch.Tensor, 
                            init_state: dict, 
                            times: torch.tensor, 
                            dt:float):
        """Dreamer interface:  imagine_with_action function.

        This function is just used for evaluating the performance of the world model in terms 
        reconsturction quality.

        Parameters:
        -----------
        actions: torch.Tensor
            Set of actions that were performed in the episode
        init_state: dict
            Initial state to start rollout
        times: torch.Tensor
            Time steps of the episode
        dt: float
            integration step size for the Euler solver

        Returns:
        --------
        prior: dict
            prior states
        """
        device = actions.device
        B, T, _ = actions.shape

        # times [T]
        ts = times[0]

        y0 = init_state["latent"]  # [B, L]

        # actions in [T,B,A]
        acts_transposed = actions.transpose(0, 1)

        # Summy context since prior is not conditioned on future observations
        ctx_dummy = torch.zeros(T, B, self.embed_dim, device=device)
        # Store contexts in latent SDE model
        self.model.contextualize((ts, ctx_dummy, acts_transposed))

        y = torchsde.sdeint(
            self.model,
            y0,
            ts,
            method=self.solver,
            dt=dt,
            names={"drift": "h", "diffusion": "g"},
            logqp=False,
        )  

        y = y.transpose(0, 1).contiguous()
        
        prior = self.convert_to_dict(y)

        return prior
