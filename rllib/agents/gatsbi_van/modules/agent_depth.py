import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.models import resnet18
from gatsbi_rl.gatsbi.arch import ARCH

class AgentDepth(nn.Module):
    """ 
        Infer the depth latent variable for the agent, with prior of N(0, I).
    """
    def __init__(self):
        nn.Module.__init__(self)
        
        self.embed_agent_latent = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE +
            ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, 2 * ARCH.Z_DEPTH_DIM)
        )

    def forward(self, z_agent, h_agent, action, leverage=False, model_T=0):
        """
            Forward inference of agent depth.
        """
        B, T, *_ = z_agent.size()

        detached_timesteps = T - model_T

        if leverage:
            z_agent_d, z_agent_g = torch.split(z_agent, [detached_timesteps, model_T], dim=1)
            h_agent_d, h_agent_g = torch.split(h_agent, [detached_timesteps, model_T], dim=1)
            action_d, action_g = torch.split(action, [detached_timesteps, model_T], dim=1)
            with torch.no_grad():
                out_d = self.embed_agent_latent(torch.cat([z_agent_d, h_agent_d, action_d], dim=-1))
            out_g = self.embed_agent_latent(torch.cat([z_agent_g, h_agent_g, action_g], dim=-1))
            out = torch.cat([out_d, out_g], dim=1)
        else:
            out = self.embed_agent_latent(torch.cat([z_agent, h_agent, action], dim=-1)) # [B, T, Zm + Hm + A]

        z_depth_loc, z_depth_scale = torch.split(out, [ARCH.Z_DEPTH_DIM] * 2, dim=-1) # [B, T, 1], [B, T, 1]

        return z_depth_loc, z_depth_scale

    def get_agent_depth_prior(self):
        return Normal(0, 1)

    def get_agent_depth_map(self, z_agent_depth, agent_mask):
        
        """ Compute agent depth from the latent variable.
            Refer to render method of object module agent_mask[0]
            (torch.ones_like(agent_mask, device=agent_mask.device) * torch.sigmoid(-z_agent_depth[..., None, None]))[0]
        """
        agent_depth_map = agent_mask * torch.sigmoid(-z_agent_depth[..., None, None])
        return agent_depth_map, torch.sigmoid(-z_agent_depth)



       