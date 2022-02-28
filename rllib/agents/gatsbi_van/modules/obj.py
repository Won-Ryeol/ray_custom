from numpy.lib.function_base import diff
import torch
import random
import numpy as np
import math
from torch.nn import functional as F
from torch import nn, t
from torch.distributions import RelaxedBernoulli, Normal, kl_divergence
from torch.types import Device
from torchvision.models import resnet18

from .module import MLP, Flatten, gaussian_kernel_2d, kl_divergence_bern_bern, \
    BatchApply, transform_tensors, anneal
from .utils import spatial_transform
from .arch import ARCH
from collections import defaultdict
# for debugging
from contextlib import nullcontext

# TODO (chmin): clean up the imports.
# TODO (chmin): vanilla gatsbi has no occlusion-related modules.
from ray.rllib.agents.gatsbi_van.modules.arch import ARCH

from IQA_pytorch import SSIM, MS_SSIM

class ObjModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.Z_DIM = ARCH.Z_PRES_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM
        self.T = ARCH.T[0]
        self.tau = ARCH.TAU_START_VALUE
        self.z_pres_prior_prob = ARCH.Z_PRES_PROB_START_VALUE
        self.z_scale_prior_loc = ARCH.Z_SCALE_MEAN_START_VALUE
        self.z_scale_prior_scale = ARCH.Z_SCALE_STD
        self.z_shift_prior_loc = ARCH.Z_SHIFT_MEAN
        self.z_shift_prior_scale = ARCH.Z_SHIFT_STD

        self.img_encoder = ImgEncoder()
        self.proposal_encoder = ProposalEncoder()
        self.proposal_obj_encoder = ProposalEncoder()

        self.pred_proposal = PredProposal()
        self.glimpse_decoder = GlimpseDecoder()
        self.latent_post_disc = LatentPostDisc()
        self.latent_post_prop = LatentPostProp()
        self.latent_prior_prop = LatentPriorProp()
        self.latent_prior_prop_action = LatentPriorProp()
        self.pres_depth_where_what_prior = PresDepthWhereWhatPrior()
        #TODO: the latent part is not used. I put it here so we can load old checkpoints
        self.pres_depth_where_what_latent_post_disc = PresDepthWhereWhatPostLatentDisc()
        self.bg_attention_encoder = BgAttentionEncoder()

        # For compute propagation map for discovery conditioning
        self.prop_map_mlp = MLP((
            # ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM,
            self.Z_DIM,
            *ARCH.PROP_MAP_MLP_LAYERS,
            ARCH.PROP_MAP_DIM,
        ),
            act=nn.CELU(),
        )
        # Compute propagation conditioning
        self.prop_cond_self_prior = MLP(
            (
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM,
                ARCH.PROP_COND_FEAT_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_relational_prior = MLP(
            (
                # (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                ARCH.PROP_COND_FEAT_DIM * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )

        self.prop_cond_self_post = MLP(
            (
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM,
                ARCH.PROP_COND_FEAT_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_relational_post = MLP(
            (
                # (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                ARCH.PROP_COND_FEAT_DIM * 2,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_weights_prior = MLP(
            (
                # (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                ARCH.PROP_COND_FEAT_DIM * 2,
                *ARCH.PROP_COND_MLP_LAYERS,
                1,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_weights_post = MLP(
            (
                # (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                ARCH.PROP_COND_FEAT_DIM * 2,
                *ARCH.PROP_COND_MLP_LAYERS,
                1,
            ),
            act=nn.CELU(),
        )

        # Propagation RNN initial states
        self.h_init_post = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        self.c_init_post = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))

        self.h_init_prior = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        self.c_init_prior = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))

        # Temporal object state rnn, used to encode history of z
        self.temporal_rnn_post_input = BatchApply(nn.Linear(
            self.Z_DIM + ARCH.PROP_COND_DIM + 64 + 1,
            ARCH.RNN_INPUT_DIM
        ))
        self.temporal_rnn_post = BatchApply(nn.LSTMCell(
            ARCH.RNN_INPUT_DIM,
            ARCH.RNN_HIDDEN_DIM
        ))

        self.temporal_rnn_prior_input = BatchApply(nn.Linear(
            self.Z_DIM + ARCH.PROP_COND_DIM + 64 + 1,
            ARCH.RNN_INPUT_DIM
        ))
        self.temporal_rnn_prior = BatchApply(nn.LSTMCell(
            ARCH.RNN_INPUT_DIM,
            ARCH.RNN_HIDDEN_DIM))

        self.z_positional_enhance = nn.Linear(ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM, ARCH.ACTION_ENHANCE)

        self.agent_object_interaction = MLP(
            (
                ARCH.BG_PROPOSAL_DIM + ARCH.Z_MASK_DIM,
                *ARCH.AO_INTERACTION_MLP_LAYERS,
                ARCH.BG_PROPOSAL_DIM,
            ),
            act=nn.CELU(),
        )

        self.extract_local_agent_feature = MLP(
            (
                ARCH.Z_DIM + ARCH.RNN_INPUT_DIM + ARCH.Z_MASK_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM,
                *ARCH.LOCAL_AGENT_MLP_LAYERS,
                ARCH.Z_MASK_DIM,
            ),
            act=nn.CELU(),
        )

        self.extract_global_agent_feature = MLP(
            (
                ARCH.Z_MASK_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM + 1,
                *ARCH.GLOBAL_AGENT_MLP_LAYERS,
                ARCH.Z_MASK_DIM,
            ),
            act=nn.CELU(),
        )

        self.extract_ao_attention = MLP(
            (
                ARCH.Z_WHERE_DIM + ARCH.RNN_INPUT_DIM +  ARCH.ACTION_ENHANCE + ARCH.RNN_CTX_MASK_HIDDEN_DIM,
                *ARCH.AO_ATTENTION_MLP_LAYERS,
                1
            ),
            act=nn.Tanh(),
        )

        # placeholder
        self.start_id = torch.zeros(1).long()
        
        # list to compute the distribution alignment of z_occ
        self.z_occ_list = []
        self.global_step = 0

    def anneal(self, global_step):
        self.tau = anneal(global_step, ARCH.TAU_START_STEP, ARCH.TAU_END_STEP,
                          ARCH.TAU_START_VALUE, ARCH.TAU_END_VALUE, 'linear')
        self.z_scale_prior_loc = anneal(global_step, ARCH.Z_SCALE_MEAN_START_STEP, ARCH.Z_SCALE_MEAN_END_STEP,
                                        ARCH.Z_SCALE_MEAN_START_VALUE, ARCH.Z_SCALE_MEAN_END_VALUE, 'linear')
        self.z_pres_prior_prob = anneal(global_step, ARCH.Z_PRES_PROB_START_STEP, ARCH.Z_PRES_PROB_END_STEP,
                                        ARCH.Z_PRES_PROB_START_VALUE, ARCH.Z_PRES_PROB_END_VALUE, 'exp')
        self.global_step = global_step

    def get_discovery_priors(self, device):
        """
        Returns:
            z_depth_prior
            z_where_prior
            z_what_prior
            z_dyna_prior
        """
        return (
            Normal(0, 1),
            Normal(torch.tensor([self.z_scale_prior_loc] * 2 + [self.z_shift_prior_loc] * 2, device=device),
                   torch.tensor([self.z_scale_prior_scale] * 2 + [self.z_shift_prior_scale] * 2, device=device)
                   ),
            Normal(0, 1),
            Normal(0, 1)
        )

    def get_init_recur_state(self):
            """
            Return the recurrent states.
            """

            h_obj_t_post = self.h_init_post.expand(1, ARCH.MAX, ARCH.RNN_HIDDEN_DIM)
            c_obj_t_post = self.c_init_post.expand(1, ARCH.MAX, ARCH.RNN_HIDDEN_DIM)

            h_obj_t_prior = self.h_init_prior.expand(1, ARCH.MAX, ARCH.RNN_HIDDEN_DIM)
            c_obj_t_prior = self.c_init_prior.expand(1, ARCH.MAX, ARCH.RNN_HIDDEN_DIM)

            state_dict = {
                'post' : (h_obj_t_post, c_obj_t_post),
                'prior' : (h_obj_t_prior, c_obj_t_prior)
            }
            return state_dict


    def get_state_init(self, B, prior_or_post):
        assert prior_or_post in ['post', 'prior']
        if prior_or_post == 'prior':
            h = self.h_init_prior.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
            c = self.c_init_prior.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
        else:
            h = self.h_init_post.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
            c = self.c_init_post.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)

        return h, c

    def get_dummy_things(self, B, device):
        # Empty previous time step
        h_post = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        c_post = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        h_prior = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        c_prior = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        z_pres = torch.zeros(B, 0, ARCH.Z_PRES_DIM, device=device)
        z_depth = torch.zeros(B, 0, ARCH.Z_DEPTH_DIM, device=device)
        z_where = torch.zeros(B, 0, ARCH.Z_WHERE_DIM, device=device)
        z_what = torch.zeros(B, 0, ARCH.Z_WHAT_DIM, device=device)
        z_dyna = torch.zeros(B, 0, ARCH.Z_DYNA_DIM, device=device)
        ids = torch.zeros(B, 0, device=device).long()
        return (h_post, c_post), (h_prior, c_prior), (z_pres, z_depth, z_where, z_what, z_dyna), ids

    def forward(self, *args, **kargs):
        return self.track(*args, **kargs)


    def infer(self, history, obs, mix, discovery_dropout, 
            z_agent=None, h_agent=None, enhanced_act=None, first=False, fg=None,
            agent_depth=None, agent_kypt=None, episodic_step=None):
        """
        Inference method of object module for the policy.
        length 8 list history contains:
            z_pres, z_depth, z_where, z_what, z_dyna, h_objs, c_objs, ids
        """
        B, C, H, W = obs.size()

        stop_discover = False
        if episodic_step is not None and episodic_step > 100:
            stop_discover = True

        # return h+c states, sto_latents, and obj ides
        z_pres, z_depth, z_where, z_what, z_dyna, h_obj, c_obj, ids = history
        
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        state_post = [h_obj, c_obj]

        #? do we use state_prior here?
        state_prior = [torch.zeros(B, ARCH.MAX, ARCH.RNN_HIDDEN_DIM, device=obs.device), \
                torch.zeros(B, ARCH.MAX, ARCH.RNN_HIDDEN_DIM, device=obs.device)]

        things = defaultdict(list)

        if first:
            self.start_id = torch.zeros(B, device=obs.device).long()

            # TODO (chmin): deprecated.
            # _agent_kypt = agent_kypt.clone()
            # agent_kypt_first = torch.cat([_agent_kypt[..., 0][..., None], 
            #     - _agent_kypt[..., 1][..., None], _agent_kypt[..., -1][..., None]], dim=-1)

            # setattr(self, "agent_kypt_prev", agent_kypt_first.clone().detach())

        x = obs # [B, 3, H, W]
        # Update object states: propagate from prev step to current step
        state_post_prop, state_prior_prop, z_prop, _, proposal = self.propagate( 
            x, fg, state_post, state_prior, z, mix, z_agent, h_agent, enhanced_act,
            first=first)

        ids_prop = ids
        if (first or torch.rand(1) > discovery_dropout) and not stop_discover:
            state_post_disc, state_prior_disc, z_disc, ids_disc, _, _, _ \
            = self.discover(x, z_prop, mix, self.start_id, z_agent, h_agent, enhanced_act)
        else: # Do not conduct discovery
            state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, obs.device)
            z_occ_disc = torch.zeros(B, ARCH.MAX, 1).to(obs.device)
            
        # Combine discovered and propagated things, and sort by p(z_pres)
        state_post, state_prior, z, ids, proposal, z_occ_combined = self.combine(
            state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
            state_post_prop, state_prior_prop, z_prop, ids_prop, proposal,
        )

        fg, alpha_map, _importance_map, y_att, alpha_att_hat = self.render(z)
        # TODO (chmin): verify this.
        self.start_id = ids.max(dim=1)[0] + 1
        
        things = dict(
            z_pres=z[0],  # (B, N, 1)
            z_depth=z[1],  # (B, N, 1)
            z_where=z[2],  # (B, N, 4)
            z_what=z[3],  # (B, N, D)
            z_dyna=z[4],  # (B, N, D)
            ids=ids,  # (B, N)
            fg=fg,  # (B, C, H, W)
            proposal=proposal,  # (B, N, 4)
            alpha_map=alpha_map,  # (B, 1, H, W)
            z_objs=z,
            h_c_objs=state_post # [B, N, H] # RNN hidden state of the object.
        )
        return things

    def track(self, obs, mix, discovery_dropout, z_agent=None, h_agent=None, 
        enhanced_act=None, agent_mask=None, leverage=False, backup=False, model_T=0,
        agent_depth=None, agent_kypt=None):
    
        """
        Doing tracking
        Args:
            obs: (B, T, C, H, W)
            bg: (B, T, C, H, W)
            discovery_dropout: (0, 1)
            agent_mask: (B, T, 1, G, G)
        Returns:
            A dictionary. Everything will be (B, T, ...). Refer to things_t.
        """

        B, T, C, H, W = obs.size()
        #! Empty, N=0, clean states obs.reshape(-1, 3, 64, 64)

        state_post, state_prior, z, ids = self.get_dummy_things(B, obs.device) # all params are zeros.
        first = True

        occ_refine_loss = torch.zeros([]).to(obs.device)

        if first:
            _agent_kypt_first = agent_kypt[:, 0].clone()
            agent_kypt_first = torch.cat([_agent_kypt_first[..., 0][..., None], 
                - _agent_kypt_first[..., 1][..., None], _agent_kypt_first[..., -1][..., None]], dim=-1)
            setattr(self, "agent_kypt_prev", agent_kypt_first.clone().detach())

        start_id = torch.zeros(B, device=obs.device).long()

        things = defaultdict(list)

        detached_timesteps = T - model_T
        # obs[0, :, :3].reshape(-1, 3, 64, 64) obs[0, t-1, 3:]
        # TODO (chmin): check the shape of z_agent and h_agent for modelling the interaction
        fg = torch.zeros(B, 3, H, W, device=obs.device)
        for t in range(T): # track over time seq., #! It seems like there's no dependency along the timestep.
            
            noisy_train_context = torch.no_grad() if leverage and t < detached_timesteps \
                else nullcontext()
            
            with noisy_train_context:
                # (B, 3, H, W) obs[:, 0, :3] obs[:, t-2, :3]  fg
                x = obs[:, t] # seq is concat of image of fg masks
                # Update object states: propagate from prev step to current step
                z_agent_t = z_agent[:, t]
                h_agent_t = h_agent[:, t]
                enhanced_act_t = enhanced_act[:, t]
                # TODO (chmin): check the 'kl_prop' and 'kl_disc' self.render(z)[0] 
                state_post_prop, state_prior_prop, z_prop, kl_prop, proposal, \
                    z_occ, z_occ_post_prob, z_pres_post_prob, \
                        z_where_offset_norm, uncertain_pos_norm, z_depth_offset, \
                        proposal_scale_update = self.propagate(
                    x, fg.clone().detach(), state_post, state_prior, z, mix[:, t], z_agent_t, 
                    h_agent_t, enhanced_act_t, first=first, agent_depth=agent_depth[:, t], 
                    agent_kypt=agent_kypt[:, t])
                if t == 0:
                    kl_occ = torch.zeros(B, device=x.device)
                    kl_uncertain = torch.zeros(B, device=x.device)
                if t >= 1:
                    kl_occ = kl_prop[5]
                    kl_uncertain = kl_prop[-1]
                    kl_prop = kl_prop[:5]

                ids_prop = ids
                # if first or (t < 15 and torch.rand(1) > discovery_dropout):
                if first or torch.rand(1) > discovery_dropout and t < 100:
                    state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc, prop_map, \
                    x_enc, z_occ_disc = self.discover(
                        x, z_prop, mix[:, t], start_id, z_agent_t, h_agent_t, enhanced_act_t, leverage, backup, 
                        agent_depth=agent_depth[:, t], agent_kypt=agent_kypt[:, t], z_occ_prop=z_occ,
                        proposal_scale_update=proposal_scale_update)
                    first = False
                else: # Do not conduct discovery
                    state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, obs.device)
                    kl_disc = (0.0, 0.0, 0.0, 0.0, 0.0)
                    z_occ_disc = torch.zeros(B, ARCH.MAX, 1).to(obs.device)
                # Combine discovered and propagated things, and sort by p(z_pres)
                # TODO (chmin): thresholding is necessary; should be no grad
                # reject the agent being discovered.
                if (not z_disc[0].size(1) == 0 and self.global_step >= ARCH.REJECT_ALPHA_START 
                    and self.global_step < ARCH.REJECT_ALPHA_UNTIL):
                    # do discovery
                    z_disc = self.reject_by_render(agent_mask[:, t], z_disc)

                state_post, state_prior, z, ids, proposal, z_occ_combined = self.combine(
                    state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2], z_occ_disc,
                    state_post_prop, state_prior_prop, z_prop, ids_prop, proposal, z_occ
                )

                # TODO (chmin): we should track z_occ.
                setattr(self, "z_occ_prev", z_occ_combined.clone().detach())

                kl = [x + y for (x, y) in zip(kl_prop, kl_disc)] # make a paired list of kl-divergences
                # kl.extend([kl_occ, kl_uncertain])
                kl.extend([kl_occ])
                # self.render(z_disc)[0]  self.render(z_disc)[1]  self.render(z)[0] self.render(z_prop)[1]
                fg, alpha_map, _importance_map, y_att, alpha_att_hat = self.render(z)
                start_id = ids.max(dim=1)[0] + 1

                things_t = dict(
                    z_pres=z[0],  # (B, N, 1)
                    z_depth=z[1],  # (B, N, 1)
                    z_where=z[2],  # (B, N, 4)
                    z_what=z[3],  # (B, N, D)
                    z_dyna=z[4],  # (B, N, D)
                    # z_objs=torch.cat(z + [ids[..., None].float()], dim=-1), # [B, N, 6 + 2D + 1],
                    z_objs=torch.cat(z, dim=-1), # [B, N, 6 + 2D + 1],
                    h_objs=state_post[0], # [B, N, H] -> RNN hidden state
                    c_objs=state_post[1], # [B, N, H] -> RNN cell state
                    kl_pres=kl[0],  # (B,)
                    kl_depth=kl[1],  # (B,)
                    kl_where=kl[2],  # (B,)
                    kl_what=kl[3],  # (B,)
                    kl_dyna=kl[4],  # (B,)
                    kl_occ=kl[5],  # (B,)
                    kl_fg=kl[0] + kl[1] + kl[2] + kl[3] + kl[4] + kl[5],  # (B,)
                    ids=ids,  # (B, N)
                    fg=fg,  # (B, C, H, W)
                    proposal=proposal,  # (B, N, 4)
                    alpha_map=alpha_map,  # (B, 1, H, W)
                    _importance_map=_importance_map,
                    alpha_att_hat=alpha_att_hat,
                    z_depth_offset=z_depth_offset,
                    y_att=y_att,
                    disc_prop_map=prop_map, # (B, D, G, G) TODO (chmin): regularize this.
                    disc_x_enc=x_enc,
                    z_occ=z_occ_combined,
                    z_occ_post_prob=z_occ_post_prob,
                    z_pres_post_prob=z_pres_post_prob,
                    z_where_offset_norm=z_where_offset_norm,
                    uncertain_pos_norm=uncertain_pos_norm
                )
                for key in things_t:
                    things[key].append(things_t[key])
                    
        things = {k: torch.stack(v, dim=1) for k, v in things.items()}
        return things

    def reject_by_render(self, agent_mask=None, z_disc=None):
        
        z_pres_disc, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc = z_disc
        if z_pres_disc.size(1) == 0:
            raise ValueError("This should not happen!")
        #! Pg.3 of supl.
        B, N, _ = z_pres_disc.size()
        # Reshape to make things easier
        z_pres = z_pres_disc.view(B * N, -1)
        z_where = z_where_disc.view(B * N, -1)
        z_what = z_what_disc.reshape(B * N, -1)
        agent_mask[0].detach().cpu().numpy().transpose(1, 2, 0)
        # Decoder z_what
        # (B*N, 3, H, W), (B*N, 1, H, W)
        o_att, alpha_att = self.glimpse_decoder(z_what) #! Eq.(42) of supl.

        # (B*N, 1, H, W)
        alpha_att_hat = alpha_att * z_pres[..., None, None] #! Eq.(43) of supl.
        # (B*N, 3, H, W)
        y_att = alpha_att_hat * o_att #! Eq.(44) of supl.
        # y_att_(hat) and alpha_att_hat are of small glimpse size (H_g * W_g)
        # To full resolution, apply inverse spatial transform (ST) (B*N, 1, H, W)
        y_att = spatial_transform(y_att, z_where, (B * N, 3, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(45) of supl.

        # To full resolution, (B*N, 1, H, W)
        alpha_att_hat = spatial_transform(alpha_att_hat, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(46) of supl.
        # Reshape back to original shape
        y_att = y_att.view(B, N, 3, *ARCH.IMG_SHAPE)
        alpha_att_hat = alpha_att_hat.view(B, N, 1, *ARCH.IMG_SHAPE)
        #! why logit is "-z_depth" ...?: depth increase -> less weight, thus -z_depth
        # (B, N, 1, H, W). H, W are glimpse size. #? why logit is "-z_depth" ...?
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth_disc[..., None, None])
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + 1e-5) #! Eq.(47) of supl.
        importance_map.sum(1).detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        # Final fg (B, N, 3, H, W)
        fg = (y_att * importance_map).sum(dim=1) #! Eq.(48) of supl.+
        # Fg mask (B, N, 1, H, W)
        alpha_map = (importance_map * alpha_att_hat).sum(dim=1) #! Eq.(49) of supl.
        fg[0].detach().cpu().numpy().transpose(1, 2, 0)
        z_pres_no_agent = z_pres.reshape(B, N, -1).clone()

        batch_idx, obj_idx = torch.where((agent_mask[:, None] * importance_map).reshape(B, N, -1).amax(dim=-1) > ARCH.AGENT_DETECT_THRESHOLD)
        z_pres_no_agent[batch_idx.long(), obj_idx.long()] = 0.0

        return (z_pres_no_agent, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc)



    def _choose_sample(self, sampled_latent, sampled_keypoints, sample_losses):
        """Returns the first or lowest-loss sample, depending on learning phase.
        During training, the sample with the lowest loss is returned.
        During inference, the first sample is returned without regard to the loss.
        Args:
            sampled_latent: [num_samples, batch_size, latent_code_size] tensor.
            sampled_keypoints: [num_samples, batch_size, 3 * num_keypoints] tensor.
            sample_losses: [num_samples, batch_size] tensor.
        Returns:
            Two tensors: latent and keypoint representation of the best sample.
        """
        # Find the indices of the samples with the lowest loss:
        best_sample_ind = torch.argmin(sample_losses, dim=0).int()  # [B,]
        batch_ind = torch.arange(0, sample_losses.size(1), dtype=torch.int32, device=sampled_latent.device)  # [B,]
        # first rank: the best sample index; second rank: index for batch [0, 1, .., B-1]

        # https://discuss.pytorch.org/t/batched-index-select-tf-gather-nd/27402
        best_latent = sampled_latent[best_sample_ind.long(), batch_ind.long()]  # [B, Zk]
        best_keypoints = sampled_keypoints[best_sample_ind.long(), batch_ind.long()]  # [B, 3 * K]

        # During training, return the best sample. During inference, return the
        # first sample:
        if self.training:  # if traiing mode
            return [best_latent, best_keypoints]
        else:
            return [sampled_latent[0], sampled_keypoints[0]]



    def generate(self, obs, mix, cond_steps, sample=True, z_agent=None, h_agent=None, enhanced_act=None, agent_mask=None,
        agent_depth=None, agent_kypt=None):
        """
        Generate new frames, given a set of input frames
        Args:
            seq: (B, T, 3, H, W)
            bg: (B, T, 3, H, W), generated bg images
            cond_steps: number of input steps
            sample: bool, sample or take mean

        Returns:
            log
        """
        B, T, C, H, W = obs.size()

        start_id = torch.zeros(B, device=obs.device).long()
        state_post, state_prior, z, ids = self.get_dummy_things(B, obs.device)

        things = defaultdict(list)
        first = False

        _agent_kypt_first = agent_kypt[:, 0].clone()
        agent_kypt_first = torch.cat([_agent_kypt_first[..., 0][..., None], 
            - _agent_kypt_first[..., 1][..., None], _agent_kypt_first[..., -1][..., None]], dim=-1)
        setattr(self, "agent_kypt_prev", agent_kypt_first.clone().detach())

        fg = torch.zeros(B, 3, H, W, device=obs.device)
        for t in range(T):
            z_agent_t = z_agent[:, t]
            h_agent_t = h_agent[:, t]
            enhanced_act_t = enhanced_act[:, t]

            if t < cond_steps: # inference
                
                if t == 0:
                    first = True
                # Input, use posterior
                x = obs[:, t]
                # state_post and state_prior are the RNN states.
                state_post_prop, state_prior_prop, z_prop, kl_prop, proposal, z_occ, *_, proposal_scale_update \
                = self.propagate(x, fg, state_post,
                    state_prior, z, mix[:, t], z_agent_t, h_agent_t, enhanced_act_t, 
                    first=first, agent_depth=agent_depth[:, t], agent_kypt=agent_kypt[:, t])
                ids_prop = ids
                if t < 5:
                    state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc, _, _, z_occ_disc \
                     = self.discover(x, z_prop, mix[:, t], 
                        start_id, z_agent_t, h_agent_t, enhanced_act_t, agent_depth=agent_depth[:, t], 
                        agent_kypt=agent_kypt[:, t], z_occ_prop=z_occ, proposal_scale_update=proposal_scale_update)
                else:
                    state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, obs.device)
            
                if (not z_disc[0].size(1) == 0 and self.global_step >= ARCH.REJECT_ALPHA_START 
                    and self.global_step < ARCH.REJECT_ALPHA_UNTIL):
                    # do discovery
                    z_disc = self.reject_by_render(agent_mask[:, t], z_disc)
            else:
                # Generation, use prior
                state_prior_prop, z_prop, z_occ = self.propagate_gen(state_prior, z, mix[:, t], sample, z_agent_t, 
                                    h_agent_t, enhanced_act_t, agent_depth=agent_depth[:, t], agent_kypt=agent_kypt[:, t])
                state_post_prop = state_prior_prop
                ids_prop = ids
                state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, obs.device)

            state_post, state_prior, z, ids, proposal, z_occ_combined = self.combine(
                state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2], z_occ_disc,
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal, z_occ
            )

            # TODO (chmin): we should track z_occ.
            setattr(self, "z_occ_prev", z_occ_combined.clone().detach())

            with torch.no_grad():
                fg, alpha_map, _importance_map, y_att, alpha_att_hat = self.render(z)
                start_id = ids.max(dim=1)[0] + 1 #! increase the starting ID if new objects are found. 
                things_t = dict(
                    z_pres=z[0],  # (B, N, 1)
                    z_depth=z[1],  # (B, N, 1)
                    z_where=z[2],  # (B, N, 4)
                    z_what=z[3],  # (B, N, D)
                    z_dyna=z[4],  # (B, N, D)
                    ids=ids,  # (B, N)
                    fg=fg,  # (B, C, H, W)
                    alpha_map=alpha_map,  # (B, 1, H, W)
                    _importance_map=_importance_map,
                    alpha_att_hat=alpha_att_hat,
                    y_att=y_att,
                )
                for key in things_t:
                    things[key].append(things_t[key])
        things = {k: torch.stack(v, dim=1) for k, v in things.items()}
        return things

    def imagine(self, mix, history, z_prop, z_agent=None, h_agent=None,
            enhanced_act=None, sample=True, agent_depth=None, agent_kypt=None):
        """
        Generate new frames given a set of input frames. NOTE that this is 
        prior samples, but the gradient should flow to train actor-critic.
        It requires conditional frame (posterior from the dataset.)
        NOTE: refer to generate() method of GATSBI, after the cond_steps
        """
        state_prior = history
        B = state_prior[0].size(0) # B -> batch * time_seq
        z, ids, proposal = z_prop

        things = defaultdict(list)
        # we only use prior temporal states here.
        # TODO (chmin): proposal should also be propagated
        # TODO (chmin): verify this
        state_prior_prop, z_prop, z_occ = self.propagate_gen( # sample determines deter/stoch samplingu
            state_prev=state_prior, z_prev=z, bg=mix, sample=sample,
            z_agent=z_agent, h_agent=h_agent, enhanced_act=enhanced_act, agent_depth=agent_depth,
            agent_kypt=agent_kypt)
        
        ids_prop = ids # make sure it's long tensor
        state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, mix.device)
        state_post_prop = state_prior_prop

        z_occ_disc = torch.zeros(B, ARCH.MAX, 1).to(mix.device)

        state_post, state_prior, z, ids, proposal, z_occ_combined = self.combine(
            state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2], z_occ_disc,
            state_post_prop, state_prior_prop, z_prop, ids_prop, proposal, z_occ
        )
        # with torch.no_grad(): # TODO (chmin): this return should be the same as track().
        #     fg, alpha_map, *_ = self.render(z) # [B*T, 3, 64, 64], [B*T, 1, 64, 64]
        start_id = ids.max(dim=1)[0] + 1 #! increase the starting ID if new objects are found.
        things = dict(
            z_pres=z[0], # [B*T, N, 1]
            z_depth=z[1], # [B*T, N, 1]
            z_where=z[2], 
            z_what=z[3], 
            z_dyna=z[4],
            z_objs=torch.cat(z, dim=-1), # [B*T, N, 6 + 2D + 1]
            ids=ids, # [B*T, N]
            proposal=proposal, # [B*T, N, 4]
            h_c_objs=state_prior, # tuple of LSTM states
            # fg=fg,
            # alpha_map=alpha_map,
            z_occ=z_occ_combined
        )
        return things

    def discover(self, x, z_prop, bg, start_id=0, z_agent=None, h_agent=None, enhanced_act=None):
        """ # Sec A.3. of supl.
        Given current image and propagated objects, discover new objects
        Args:
            x: (B, D, H, W), current input image
            bg: (B, 3, H, W), bg w/o agent
            agent_mask: (B, 1, G, G), if agent exist, a mask for agent-existing area
            z_prop:
                z_pres_prop: (B, N, 1)
                z_depth_prop: (B, N, 1)
                z_where_prop: (B, N, 4)
                z_dyna_prop: (B, N, D)
            start_id: the id to start indexing. The id to start indexing...
        Returns:
            (h_post, c_post): (B, N, D)
            (h_prior, c_prior): (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            ids: (B, N)
            kl:
                kl_pres: (B,)
                kl_depth: (B,)
                kl_where: (B,)
                kl_what: (B,)
                kl_dyna (B,) x[:, :3] x[:, 3:]
        )
        """
        B, *_ = x.size() 
        # (B, D, G, G) x[:, 3:]
        x_enc = self.img_encoder(x)
        # (B, D, G, G) # TODO (chmin): check the shape of 'z_prop'
        prop_map = self.compute_prop_map(z_prop) # construct 2D Gaussian kernel #! Eq.(32) of supl?
        # (B, D, G, G) prop_map.mean(1)[:, None] x_enc.mean(1)[:, None] enc.mean(1)[:, None]
        enc = torch.cat([x_enc, prop_map], dim=1) # condition discovery on propagation.
        #! get posteriors of latents from discovery
        (z_pres_post_prob, z_depth_post_loc, z_depth_post_scale, z_where_post_loc,
         z_where_post_scale, z_what_post_loc, z_what_post_scale, z_dyna_loc,
         z_dyna_scale) = self.pres_depth_where_what_latent_post_disc(enc) #! Eq.(33) of supl

        z_dyna_loc, z_dyna_scale = self.latent_post_disc(enc) #! Eq.(34) of supl
        # z_dyna_loc, z_dyna_scale = self.latent_post_disc_action.forward_act(enc, ee) #! Eq.(34) of supl
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample()

        # Compute posteriors. All (B, G*G, D)
        z_pres_post = RelaxedBernoulli(temperature=self.tau, probs=z_pres_post_prob)
        z_pres = z_pres_post.rsample() #! Eq.(35) of supl

        z_depth_post = Normal(z_depth_post_loc, z_depth_post_scale)
        z_depth = z_depth_post.rsample() #! Eq.(36) of supl

        z_where_post = Normal(z_where_post_loc, z_where_post_scale)
        z_where = z_where_post.rsample() #! Eq.(37) of supl
        z_where = self.z_where_relative_to_absolute(z_where) #! Eq.(39), (40)

        z_what_post = Normal(z_what_post_loc, z_what_post_scale)
        z_what = z_what_post.rsample() #! Eq.(38) of supl

        # Combine the posterior samples z_{***}
        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        # reject the agent before the real? rejection self.render(z)[0]             
        # TODO (chmin): reject the gripper being captured as the agent. -> use ssim?
        # self.reject_agent_by_render(z, agent_obs, ARCH.MAX_PIX_THRESHOLD, x)
        # just normalize alpha_map * agent_obs
        # Rejection
        if ARCH.REJECTION: # Rejection adopted from SCALOR
            z = self.rejection(z, z_prop, ARCH.REJECTION_THRESHOLD)
            z = self.self_rejection(z, 0.6)

        #! these are original codes
        ids = torch.arange(ARCH.G ** 2, device=x_enc.device).expand(B, ARCH.G ** 2) + start_id[:, None]

        # Update temporal states     self.render(z)[0]   self.render(z_prop)[0]
        state_post_prev = self.get_state_init(B, 'post')
        state_prior_prev = self.get_state_init(B, 'prior')

        z, state_post_prev, state_prior_prev, ids = self.select(
            z_pres, z, state_post_prev, state_prior_prev, ids)
        #! these are original codes
        # propagated objects should be weighted by z_occ.

        # TODO (chmin): add keypoint related features for temporal encoding.
        _agent_kypt = agent_kypt.clone()

        agent_kypt = torch.cat([_agent_kypt[..., 0][..., None], 
            - _agent_kypt[..., 1][..., None], _agent_kypt[..., -1][..., None]], dim=-1)

        kl_pres = kl_divergence_bern_bern(z_pres_post_prob, torch.full_like(z_pres_post_prob, self.z_pres_prior_prob))

        z_depth_prior, z_where_prior, z_what_prior, z_dyna_prior = self.get_discovery_priors(x.device)
        # where prior, (B, G*G, 4)
        kl_where = kl_divergence(z_where_post, z_where_prior)
        kl_where = kl_where * z_pres

        # what prior (B, G*G, D)
        kl_what = kl_divergence(z_what_post, z_what_prior)
        kl_what = kl_what * z_pres

        # what prior (B, G*G, D)
        kl_depth = kl_divergence(z_depth_post, z_depth_prior)
        kl_depth = kl_depth * z_pres

        # latent prior (B, G*G, D)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior)
        kl_dyna = kl_dyna

        # TODO (chmin): crop out top N objects. to reduce computational cost.
        # kl_pres, kl_depth, kl_where, kl_what, kl_dyna = self.select_kldiv(
        #     z_pres, (kl_pres, kl_depth, kl_where, kl_what, kl_dyna))
        z_occ_disc = torch.zeros_like(z[0]).to(x.device)

        state_post = self.temporal_encode(state_post_prev, z, bg, z_agent, h_agent, agent_depth, 
            enhanced_act, prior_or_post='post', agent_kypt=agent_kypt, z_occ=z_occ_disc) # get random hidden & cell state
        state_prior = self.temporal_encode(state_prior_prev, z, bg, z_agent, h_agent, agent_depth,
            enhanced_act, prior_or_post='prior', agent_kypt=agent_kypt, z_occ=z_occ_disc)


        #! Sum over non-batch dimensions
        kl_pres = kl_pres.flatten(start_dim=1).sum(1)
        kl_where = kl_where.flatten(start_dim=1).sum(1)
        kl_what = kl_what.flatten(start_dim=1).sum(1)
        kl_depth = kl_depth.flatten(start_dim=1).sum(1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(1)
        # kl_dyna = torch.zeros_like(kl_dyna)
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)

        return state_post, state_prior, z, ids, kl, prop_map, x_enc

    def propagate_gen(self, state_prev, z_prev, bg, sample=False, z_agent=None, h_agent=None, enhanced_act=None,
        agent_depth=None, agent_kypt=None):
        """
        One step of propagation generation
        Args:
            h_prev, c_prev: (B, N, D)
            z_prev:
                z_pres_prev: (B, N, 1)
                z_depth_prev: (B, N, 1)
                z_where_prev: (B, N, 4)
                z_what_prev: (B, N, D)
        Returns:
            h, c: (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
        """
        h_prev, c_prev = state_prev # [B*T, H]
        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev

        # (B, N, D)
        #! latent size of z_dyna is 128 (default)
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop(state_prev[0])
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_prior.rsample() if sample else z_dyna_loc

        # All (B, N, D)

        _agent_kypt = agent_kypt.clone()

        agent_kypt = torch.cat([_agent_kypt[..., 0][..., None], 
            - _agent_kypt[..., 1][..., None], _agent_kypt[..., -1][..., None]], dim=-1)

        kypt_pos = agent_kypt[..., :2] # [B, K, 2]
        # agent_kypt_mean = torch.mean(kypt_pos, dim=1, keepdim=True) # [B, 1, 2] mean position of agent keypoints.
        kypt_weight = agent_kypt[...,-1][..., None] # [B, K, 1]
        agent_kypt_mean = (kypt_pos * kypt_weight).sum(1, keepdim=True) / kypt_weight.sum(1, keepdim=True)
        
        obj_pos = z_where_prev[..., 2:] # [B, N, 2] 

        z_agent_depth = agent_depth[:, None] # [B, 1, 1]

        # TODO (chmin): make sure that z_depth be normalized.
        obj_to_agent_depth = z_agent_depth - torch.sigmoid(-z_depth_prev) # [B, N, 1]
        obj_to_agent_where = agent_kypt_mean - obj_pos # [B, N, 2] positional vector from agent to each object.

        kypt_embeds = self.occlu_policy.uncertain_attention.embed(kypt_pos)
        obj_embeds = self.occlu_policy.uncertain_attention.embed(obj_pos)

        z_occ, _ = self.occlu_policy.z_occ_prior(
            agent_kypt_embedding=kypt_embeds,
            obj_embedding=obj_embeds,
            obj_to_agent_depth=obj_to_agent_depth,
            obj_to_agent_where=obj_to_agent_where,
            enhanced_act=enhanced_act
            )

        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, z_where_offset_scale,
         z_what_offset_loc,
         z_what_offset_scale, z_depth_gate, z_where_gate, z_what_gate, z_where_gate_raw) = self.pres_depth_where_what_prior(
             z_dyna, z_occ=z_occ)

        # Always set them to one during generation
        z_pres_prior = RelaxedBernoulli(temperature=self.tau, probs=z_pres_prob)
        z_pres = z_pres_prior.rsample()
        z_pres = (z_pres > 0.5).float()
        z_pres = torch.ones_like(z_pres)
        z_pres = z_pres_prev #* z_pres
        # z_pres = z_pres_prev * z_pres * (1. - z_occ) + z_pres_prev * z_occ   # -> refer to Sec 2.5 of PRML

        z_where_prior = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_prior.rsample() if sample else z_where_offset_loc
        z_where = torch.zeros_like(z_where_prev)

        uncertain_pos, uncertain_gate_raw, _ = self.occlu_policy.sample_object(
            agent_kypt=agent_kypt, proposal=z_where_prev, enhanced_act=enhanced_act, z_occ=z_occ,
            z_agent=z_agent, h_agent=h_agent, agent_kypt_prev=self.agent_kypt_prev)
        uncertain_pos = ARCH.Z_SHIFT_UPDATE_SCALE * torch.tanh(uncertain_pos) 
        uncertain_gate = torch.sigmoid(uncertain_gate_raw)

        setattr(self, "agent_kypt_prev", agent_kypt.clone().detach())

        if self.global_step >= ARCH.REJECT_ALPHA_UNTIL:
            z_where[..., :2] = z_where_prev[..., :2] + (1. - z_occ) * ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
                z_where_offset[..., :2])

            # shift
            z_where[..., 2:] = z_where_prev[..., 2:] + (1. - z_occ) * ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
                z_where_offset[..., 2:]) + z_occ * uncertain_gate * uncertain_pos

            z_depth_prior = Normal(z_depth_offset_loc, z_depth_offset_scale)
            z_depth_offset = z_depth_prior.rsample() if sample else z_depth_offset_loc
            z_depth = z_depth_prev + (1. - z_occ) * ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset

            z_what_prior = Normal(z_what_offset_loc, z_what_offset_scale)
            z_what_offset = z_what_prior.rsample() if sample else z_what_offset_loc
            z_what = z_what_prev + (1. - z_occ) * ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset)
        else:
            z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2]) #! Eq.(27) of supl.
            # Shift
            z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
                z_where_offset[..., 2:])

            z_depth_prior = Normal(z_depth_offset_loc, z_depth_offset_scale)
            z_depth_offset = z_depth_prior.rsample()  #! Eq.(15) of supl.
            z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset # [B, D, 1]

            z_what_prior = Normal(z_what_offset_loc, z_what_offset_scale)
            z_what_offset = z_what_prior.rsample()  #! Eq.(17) of supl.
            z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset) # [B, D, 64]

        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        state = self.temporal_encode(state_prev, z, bg, z_agent, h_agent, agent_depth, 
            enhanced_act, prior_or_post='prior', agent_kypt=agent_kypt, z_occ=z_occ)

        return state, z, z_occ


    def propagate(self, x, fg, state_post_prev, state_prior_prev, z_prev, bg, z_agent, h_agent, 
        enhanced_act, first=False):
        """
        # One step posterior propagation
        Args:
            x: (B, 3, H, W), img
            fg: (B, 3, H, W), image from previous steps.
            (h, c), (h, c): each (B, N, D)
            z_prev:
                z_pres: (B, N, 1) -> Bernoulli distrib.
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)

        Returns:
            h_post, c_post: (B, N, D)
            h_prior, c_prior: (B, N, D)
            z:
                z_pres: (B, N, 1) 
                z_depth: (B, N, 1) 
                z_where: (B, N, 4) 
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            kl:
                kl_pres: (B,)
                kl_what: (B,)
                kl_where: (B,)
                kl_depth: (B,)
                kl_dyna: (B,)
            proposal_region: (B, N, 4) x[:, :3]
            x[:, :3] fg[1]
        """

        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev
        if len(z_pres_prev.size()) == 4: # exception handling
            z_pres_prev = torch.squeeze(z_pres_prev, 0)
        B, N, _ = z_pres_prev.size() # [B, N (max), 1]
        if first: # num of entities. zero if nothing has been discovered before
            # No object is propagated -> then do discovery self.render(z_prev)[0]
            return state_post_prev, state_prior_prev, z_prev, (0.0, 0.0, 0.0, 0.0, 0.0)

        # self.render(z_prev)[0]
        h_post, c_post = state_post_prev # [B, N, 128], [B, N, 128] # NOTE that h_post is inferred by Eq.(18)
        h_prior, c_prior = state_prior_prev # [B, N, 128], [B, N, 128]

        # Predict proposal locations, (B, N, 2)
        proposal_offset = self.pred_proposal(h_post) #! Eq.(19) of supl., by posterior OS-RNN; h_{hat}_{t-1}^k
        proposal = torch.zeros_like(z_where_prev) # [B, N, 4] -> size of proposal area

        # Update size only; extract proposal region of the image, centered @ the prev. obj. location o_{t-1}^{xy,k}
        proposal[..., 2:] = z_where_prev[..., 2:] # o^{xy} - (x,y)
        proposal[..., :2] = z_where_prev[..., :2] + ARCH.PROPOSAL_UPDATE_MIN + ( #! o^{hw}; Eq.(19) of supl., (h,w)
            ARCH.PROPOSAL_UPDATE_MAX - ARCH.PROPOSAL_UPDATE_MIN) * torch.sigmoid(proposal_offset)

        # Get proposal glimpses S
        # x_repeat: (B*N, 3, H, W) -> observation for each object N.
        x_repeat = torch.repeat_interleave(x[:, :3], N, dim=0) # crop-out the first three channel.
        # if N = 5 & batch is a, b, ..., : aaaaabbbbbccccc ... # if torch.repeat ab...ab...ab...ab...ab...

        # (B*N, 3, H, W) #! Eq.(4) of paper
        proposal_glimpses = spatial_transform(image=x_repeat, z_where=proposal.view(B * N, 4), #! Eq.(20) of supl.
                    out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))

        # (B, N, 3, H, W) proposal_glimpses.reshape(-1, 3, 16, 16)
        proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)

        # (B, N, D) occlusion_glimpses.reshape(-1, 3, 16, 16)
        proposal_enc = self.proposal_encoder(proposal_glimpses) #! Eq.(21) of supl.

        # (B, N, D)
        # This will be used to condition everything:
        enc = torch.cat([proposal_enc, h_post], dim=-1) # [B, N, D] + [B, N, D] = [B, N, 2D]

        z_dyna_loc, z_dyna_scale = self.latent_post_prop(enc) #! Eq.(22) of supl.
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample() #! Eq.(23) of supl.
        # given dynamics latent (z_{dyna}) from posterior (z_dyna_post), the attribute latents are computed as follows
        #! {pres, depth, where, what} are inferred from z_dyna  (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, #! Eq.(13) of supl.
         z_where_offset_scale, z_what_offset_loc, z_what_offset_scale,
         z_depth_gate, z_where_gate, z_what_gate, z_where_gate_raw) = self.pres_depth_where_what_prior(z_dyna)

        # Sampling
        z_pres_post = RelaxedBernoulli(torch.tensor(self.tau, device=x.device), probs=z_pres_prob)  #! Eq.(14) of supl.
        z_pres = z_pres_post.rsample()
        z_pres = z_pres_prev * z_pres

        z_where_post = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_post.rsample()  #! Eq.(16) of supl.
        z_where = torch.zeros_like(z_where_prev)
        # Scale; [B, N, 4] -> [B, N, 2] (mean), [B, N, 2] (shift).
        # if z_occ: update by this policy       fg

        # TODO (chmin): z_occ ~ 1 (occlusion); z_occ ~ 0 (no occlusion)
        # it is not a sufficient reward signal when z_occ is ... z_occ ~ 1 
        # Scale
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])

        z_depth_prior = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_prior.rsample()
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset

        z_what_post = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_post.rsample()  #! Eq.(17) of supl.
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset)

        # TODO (chmin): experimental
        # z_what[occ_ind] = z_what_prev[occ_ind].clone().detach()
        
        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        # TODO (chmin): agent-object interaction.
        # Update states via RNN; In: state_*_prev - hidden & cell state of RNN, z - tuple of z_att and z_dyna

        state_post = self.temporal_encode(state_post_prev, z, bg, z_agent, h_agent,  
            enhanced_act, prior_or_post='post')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, z_agent, h_agent,
            enhanced_act, prior_or_post='prior')

        z_dyna_loc, z_dyna_scale = self.latent_prior_prop(h_prior) #! Eq(23) of supl.
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior) #! we get post & prior z_{dyna} from each Eq.(23) and Eq.(12)
        #! This is not kl divergence. This is an auxialiary loss q(z_pres|) is fit to fixed prior
        kl_pres = kl_divergence_bern_bern(z_pres_prob, torch.full_like(z_pres_prob, self.z_pres_prior_prob))
        # If we don't want this auxiliary loss
        if not ARCH.AUX_PRES_KL:
            kl_pres = torch.zeros_like(kl_pres)

        # Reduced to (B,)
        #! Sec. 3.4.2 of paper; Again, this is not really kl
        kl_pres = kl_pres.flatten(start_dim=1).sum(-1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(-1)

        # We are not using q, so these will be zero: ->  Only prior exist?
        kl_where = torch.zeros_like(kl_pres)
        kl_what = torch.zeros_like(kl_pres)
        kl_depth = torch.zeros_like(kl_pres)
        assert kl_pres.size(0) == B
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)

        # return the updated RNN states and the stochastic latent.
        return state_post, state_prior, z, kl, proposal
        
    def compute_prop_map(self, z_prop):
        """ #! please refer to paragraph below Eq.(31) of supl.
        Compute a feature volume to condition discovery. The purpose is not to rediscover objects
        Args:
            z_prop:
                z_pres_prop: (B, N, D)
                z_depth_prop: (B, N, D)
                z_where_prop: (B, N, 4)
                z_what_prop: (B, N, D)

        Returns:
            map: (B, D, G, G). This will be concatenated with the image feature
        """
        # get the latents (pres, depth, where, what, dyna) from propagation
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop # N is the number of propagated objects
        z_pres_prop = z_pres_prop.detach()
        # if len(z_prop) == 4:
        #     z_pre
        B, N, _ = z_pres_prop.size()
        # TODO (chmin): figure out how to return
        if z_pres_prop.size(1) == 0: # No object is propagated
            # First frame, empty prop map
            return torch.zeros(B, ARCH.PROP_MAP_DIM, ARCH.G, ARCH.G, device=z_pres_prop.device)

        assert N == ARCH.MAX # check the max # of objects in the scene
        # Use only z_what and z_depth here
        # (B, N, D)
        # TODO: I could have used relative z_where as SILOT here. But that will induce many computations. So I won't do it here.
        z_prev = torch.cat(z_prop, dim=-1) # concat (pres, depth, where, what, dyna)
        #! consider propagated objects to prevent the rediscovery.
        # (B, N, D) -> (B, N, D)
        z_prev_enc = self.prop_map_mlp(z_prev) # MLP^{cond} of Eq.(32)?
        
        # (B, N, D), masked out objects with z_pres == 0
        z_prev_enc = z_prev_enc * z_pres_prop # Update the Bernoulli probability

        # Compute a weight matrix of size (B, G*G, N)

        # (2, G, G) -> (G, G, 2) -> (G*G, 2)
        offset = self.get_offset_grid(z_prev.device) # grid that represents the center of z_shift of each cell
        offset = offset.permute(1, 2, 0).view(-1, 2)

        # crop out (B, N, 2) from (B, N, 4)
        z_shift_prop = z_where_prop[..., 2:]

        # Distance matrix; distance btwn the prop. object and corresponding cell center.
        # (1, G*G, 1, 2)
        offset = offset[None, :, None, :]
        # (B, 1, N, 2)
        z_shift_prop = z_shift_prop[:, None, :, :]
        # (B, G*G, N, 2)
        matrix = offset - z_shift_prop
        # (B, G*G, N)
        weights = gaussian_kernel_2d(matrix, ARCH.PROP_MAP_SIGMA, dim=-1)
        # (B, G*G, N) -> (B, G*G, N, 1) weights.permute(0, 3, 2, 1)
        weights = weights[..., None]
        # (B, N, D) -> (B, 1, N, D)
        z_prev_enc = z_prev_enc[:, None, ]

        # (B, G*G, N, 1) * (B, 1, N, D) -> (B, G*G, N, D) -> sum -> (B, G*G, D=ARCH.PROP_MAP_DIM)
        prop_map = torch.sum(weights * z_prev_enc, dim=-2) #! Eq.(32) of supl.
        assert prop_map.size() == (B, ARCH.G ** 2, ARCH.PROP_MAP_DIM)
        # (B, G, G, D)
        prop_map = prop_map.view(B, ARCH.G, ARCH.G, ARCH.PROP_MAP_DIM)
        # (B, D, G, G)
        prop_map = prop_map.permute(0, 3, 1, 2)

        return prop_map

    def compute_prop_cond(self, z, state_prev, prior_or_post):
        """
        Object interaction in vprop
        Args:
            z: z_t
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
            state_prev: h_t
                h: (B, N, D)
                c: (B, N, D)
        Returns:
            cond: (B, N, D)
        """
        assert prior_or_post in ['prior', 'post']
        if prior_or_post == 'prior':
            prop_cond_self = self.prop_cond_self_prior
            prop_cond_relational = self.prop_cond_relational_prior
            prop_cond_weights = self.prop_cond_weights_prior
        else:
            prop_cond_self = self.prop_cond_self_post
            prop_cond_relational = self.prop_cond_relational_post
            prop_cond_weights = self.prop_cond_weights_post

        z_pres, z_depth, z_where, z_what, z_dyna = z
        B, N, _ = z_pres.size()
        h_post_prev, c_post_prev = state_prev  # (capoo) though the name is _post_, it can be _prior_ for the prior case.

        # The feature of one object include the following
        # (B, N, D)

        feat = torch.cat(z + (h_post_prev,), dim=-1)
        # (B, N, D)
        enc_self = prop_cond_self(feat)

        # Compute weights based on gaussian
        # (B, N, 2)
        z_shift_prop = z_where[:, :, 2:]
        # (B, N, 1, 2)
        z_shift_self = z_shift_prop[:, :, None]
        # (B, 1, N, 2)
        z_shift_other = z_shift_prop[:, None]
        # (B, N, N, 2) #* NOTE that this is Manhattan distance.
        # (capoo) dist_matrix[b,k] = (N,2) matrix for distance between k-th object and others
        dist_matrix = z_shift_self - z_shift_other

        z_depth_self = z_depth[:, :, None]
        z_depth_other = z_depth[:, None]
        depth_dist_matrix = z_depth_self - z_depth_other
        # depth_dist_matrix.permute(0, 3, 1, 2)
        # (B, N, 1, D) -> (B, N, N, D)
        # feat_self = enc_self[:, :, None] feat_matrix_self.mean(-1) feat_matrix_other.mean(-1)
        feat_self = feat[:, :, None] # [B, N, 1, D]
        feat_matrix_self = feat_self.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        # (B, 1, N, D) -> (B, N, N, D)
        feat_other = feat[:, None] # [B, 1, N, D]
        feat_matrix_other = feat_other.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        # TODO: Detail here, replace absolute positions with relative ones
        offset = ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_SCALE_DIM # pre_depth 
        offset_depth = ARCH.Z_PRES_DIM
        # Must clone. Otherwise there will be multiple write
        feat_matrix_other = feat_matrix_other.clone() # feat_matrix_other = feat_matrix_self up to here.
        feat_matrix_other[..., offset:offset + ARCH.Z_SHIFT_DIM] = dist_matrix # pres-depth-where-what-dyna
        feat_matrix_other[..., offset_depth:offset_depth + ARCH.Z_DEPTH_DIM] = depth_dist_matrix
        # (B, N, N, 2D)
        feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)

        # COMPUTE WEIGHTS
        # (B, N, N, 1)
        weight_matrix = prop_cond_weights(feat_matrix)
        # (B, N, >N, 1) weight_matrix.permute(0, 3, 1, 2)
        # (capoo) sum of interaction probability between object k and others must be 1
        weight_matrix = weight_matrix.softmax(dim=2) # axis 1 is object n and axis 2 is the other.
        # Times z_pres (B, N, 1)-> (B, 1, N, 1)
        weight_matrix = weight_matrix * z_pres[:, None] # [B, N, N, 1] * [B, 1, N, 1]
        # Self mask, set diagonal elements to zero. (B, >N, >N, 1)
        # weights.diagonal: (B, 1, N)
        diag = weight_matrix.diagonal(dim1=1, dim2=2)  # (capoo) meaning: find dim1 == dim2
        diag *= 0.0 # pointer to diagonal elements.
        # Renormalize (B, N, >N, 1)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=2, keepdim=True) + 1e-4)

        relational_matrix = prop_cond_relational(feat_matrix)

        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)

        # (B, N, D)
        prop_cond = enc_self + enc_relational # self-historical + relational

        # (B, N, D)
        return prop_cond

    def get_offset_grid(self, device):
        """
        Get a grid that represents the center of z_shift of each cell
        Args:
            device: device
        #! called by Gaussian kernel construction and shifting o^{where}
        Returns:
            (2, G, G), where 2 is (x, y)
        """

        # (G, G), (G, G)
        offset_y, offset_x = torch.meshgrid(
            [torch.arange(ARCH.G), torch.arange(ARCH.G)])
        # (2, G, G)
        offset = torch.stack((offset_x, offset_y), dim=0).float().to(device)
        #! Scale: (0, G-1) -> (0.5, G-0.5) -> (0, 2) -> (-1, 1)
        offset = (2.0 / ARCH.G) * (offset + 0.5) - 1.0
        return offset

    def z_where_relative_to_absolute(self, z_where):
        """
        Convert z_where relative value to absolute value. The result is usable
        in spatial transform
        Args:
            z_where: (B, G, G, 4)

        Returns:
            z_where: (B, G, G, 4)
        """
        B, GG, D = z_where.size()
        # (B, G*G, 2) * 2
        z_scale, z_shift = z_where.chunk(2, dim=2)
        # (2, G, G), in range (-1, 1)
        offset = self.get_offset_grid(z_where.device)
        # scale: (-1, 1) -> (-2 / G, 2 / G). As opposed to the full range (-1, 1),
        # The maximum shift if (2 / G) / 2 = 1 / G, which is one cell
        # (2, G, G) -> (G, G, 2) -> (G*G, 2)
        offset = offset.permute(1, 2, 0).view(GG, 2)
        z_shift = (2.0 / ARCH.G) * torch.tanh(z_shift) + offset #! Eq.(38) of supl.
        z_scale = torch.sigmoid(z_scale) #! Eq.(39) of supl.

        # (B, G*G, 4)
        z_where = torch.cat([z_scale, z_shift], dim=2)

        return z_where

    def temporal_encode(self, state, z, bg, z_agent, h_agent, enhanced_act, prior_or_post):
        """
        Encode history into rnn states
        Args:
            state: t-1
                h: (B, N, D)
                c: (B, N, D)
            z: t-1
                z_pres: (B, N, D)
                z_depth: (B, N, D)
                z_where: (B, N, D)
                z_what: (B, N, D)
            prop_cond: (B, N, D), t-1
            bg: [B, 3, dim, dim], t
            prior_or_post: either 'prior' or 'post
        Returns:
            state:
                h: (B, N, D)
                c: (B, N, D)
        """
        # TODO (chmin): for some steps, the shapes become inconsistent

        assert prior_or_post in ['prior', 'post']
        B, N, _ = state[0].size()
        prop_cond = self.compute_prop_cond(z, state, prior_or_post) # Sec. 3.3.1 - Interaction & Occlusion
        bg_enc = self.bg_attention(bg, z) # Sec. 3.3.2 - Situation Awareness #(B, OBJECT_MAX_NUM, D)
        # 1. agent feature and obj features
        if ARCH.AGENT_INTERACTION_MODE >= 2:
            h_obj, c_obj = state # [B, G*G, H_o]
            z_obj = torch.cat(z, dim=-1)
            if len(h_agent.size()) == 1:
                h_agent = h_agent[None]
            # TODO (Chmin): concat z_depth here.
            u_agent = torch.cat([z_agent, h_agent], dim=-1)
            u_agent = torch.stack([u_agent] * N, dim=1)
            u_obj = torch.cat([z_obj, h_obj], dim=-1)
            if ARCH.AGENT_INTERACTION_MODE == 2: # GATSBI.
                glob_feat_agent = self.extract_global_agent_feature(u_agent)
                ambient_enc = self.agent_object_interaction(torch.cat((bg_enc, glob_feat_agent), dim=-1))
            elif ARCH.AGENT_INTERACTION_MODE == 3:
                loc_feat_agent = self.extract_local_agent_feature(torch.cat([u_obj, u_agent], dim=-1))
                u_pos_agent = torch.stack([torch.cat([enhanced_act, h_agent], dim=-1)] * N, dim=1)
                u_pos_obj = torch.cat([z[2], h_obj], dim=-1)
                ao_attention = self.extract_ao_attention(torch.cat((u_pos_obj, u_pos_agent), dim=-1))
                ao_attention = (ao_attention + 0.5) / 2
                ambient_enc = self.agent_object_interaction(torch.cat((bg_enc, loc_feat_agent * ao_attention), dim=-1))
        else:
            ambient_enc = bg_enc
        # bg_enc - (B, N, D)
        # Also encode interaction here
        z = torch.cat(z + (prop_cond, ambient_enc), dim=-1)

        # BatchApply is a cool thing and it works here.
        if prior_or_post == 'post':
            inpt = self.temporal_rnn_post_input(z)
            state = self.temporal_rnn_post(inpt, state)
        else:
            inpt = self.temporal_rnn_prior_input(z)
            state = self.temporal_rnn_prior(inpt, state)

        return state

    def render(self, z):
        """
        Render z into an image
        Args:
            z:
                z_pres: (B, N, D)
                z_depth: (B, N, D)
                z_where: (B, N, D)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
        Returns:
            fg: (B, 3, H, W)
            alpha_map: (B, 1, H, W)
        """
        z_pres, z_depth, z_where, z_what, z_dyna = z
        #! Pg.3 of supl.
        B, N, _ = z_pres.size()
        # Reshape to make things easier
        z_pres = z_pres.view(B * N, -1)
        z_where = z_where.view(B * N, -1)
        z_what = z_what.reshape(B * N, -1)

        # Decoder z_what
        # (B*N, 3, H, W), (B*N, 1, H, W)  o_att * alpha_att
        o_att, alpha_att = self.glimpse_decoder(z_what) #! Eq.(42) of supl.

        # (B*N, 1, H, W)
        alpha_att_hat = alpha_att * z_pres[..., None, None] #! Eq.(43) of supl.
        # (B*N, 3, H, W)
        y_att = alpha_att_hat * o_att #! Eq.(44) of supl.
        # y_att_(hat) and alpha_att_hat are of small glimpse size (H_g * W_g)
        # To full resolution, apply inverse spatial transform (ST) (B*N, 1, H, W)
        y_att = spatial_transform(y_att, z_where, (B * N, 3, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(45) of supl.
        # spatial_transform(alpha_att, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True).reshape(B, N, 1, 64, 64).sum(1)
        # To full resolution, _importance_map.sum(1)
        alpha_att_hat = spatial_transform(alpha_att_hat, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(46) of supl.
        # Reshape back to original shape
        y_att = y_att.view(B, N, 3, *ARCH.IMG_SHAPE)
        alpha_att_hat = alpha_att_hat.view(B, N, 1, *ARCH.IMG_SHAPE)
        #! why logit is "-z_depth" ...?: depth increase -> less weight, thus -z_depth
        # (B, N, 1, H, W). H, W are glimpse size. #? alpha_att_hat.sum(1) importance_map.sum(1)
        _importance_map = alpha_att_hat * torch.sigmoid(-z_depth[..., None, None])
        importance_map = _importance_map / (torch.sum(_importance_map, dim=1, keepdim=True) + 1e-5) #! Eq.(47) of supl.

        # Final fg (B, N, 3, H, W) alpha_att_hat.sum(1) * torch.sigmoid(-z_depth[..., None, None])
        fg = (y_att * importance_map).sum(dim=1) #! Eq.(48) of supl.+ y_att.sum(1)
        # Fg mask (B, N, 1, H, W)
        alpha_map = (importance_map * alpha_att_hat).sum(dim=1) #! Eq.(49) of supl.
        # TODO (chmin): return alpha_att_hat to create depth map with the agent.
        return fg, alpha_map, _importance_map, y_att, alpha_att_hat


    def render_fake(self, z):
        """
        Render z into an image
        Args:
            z:
                z_pres: (B, N, D)
                z_depth: (B, N, D)
                z_where: (B, N, D)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
        Returns:
            fg: (B, 3, H, W)
            alpha_map: (B, 1, H, W)
        """
        z_pres, z_depth, z_where, z_what, z_dyna = z
        #! Pg.3 of supl.
        B, N, _ = z_pres.size()
        # Reshape to make things easier
        z_pres = z_pres.view(B * N, -1)
        z_where = z_where.view(B * N, -1)
        z_what = z_what.reshape(B * N, -1)

        # Decoder z_what
        # (B*N, 3, H, W), (B*N, 1, H, W)  o_att * alpha_att
        o_att, alpha_att = self.glimpse_decoder(z_what) #! Eq.(42) of supl.

        # (B*N, 1, H, W)
        alpha_att_hat = alpha_att * z_pres[..., None, None] #! Eq.(43) of supl.
        # (B*N, 3, H, W)
        y_att = alpha_att_hat * o_att #! Eq.(44) of supl.
        # y_att_(hat) and alpha_att_hat are of small glimpse size (H_g * W_g)
        # To full resolution, apply inverse spatial transform (ST) (B*N, 1, H, W)
        y_att = spatial_transform(y_att, z_where, (B * N, 3, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(45) of supl.
        # spatial_transform(alpha_att, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True).reshape(B, N, 1, 64, 64).sum(1)
        # To full resolution, _importance_map.sum(1)
        alpha_att_hat = spatial_transform(alpha_att_hat, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(46) of supl.
        # Reshape back to original shape
        y_att = y_att.view(B, N, 3, *ARCH.IMG_SHAPE)
        alpha_att_hat = alpha_att_hat.view(B, N, 1, *ARCH.IMG_SHAPE)
        #! why logit is "-z_depth" ...?: depth increase -> less weight, thus -z_depth
        # (B, N, 1, H, W). H, W are glimpse size. #? alpha_att_hat.sum(1) importance_map.sum(1)
        _importance_map = alpha_att_hat * torch.sigmoid(-z_depth[..., None, None])
        importance_map = _importance_map / (torch.sum(_importance_map, dim=1, keepdim=True) + 1e-5) #! Eq.(47) of supl.

        # Final fg (B, N, 3, H, W) alpha_att_hat.sum(1) * torch.sigmoid(-z_depth[..., None, None])
        fg = (y_att * importance_map).sum(dim=1) #! Eq.(48) of supl.+ y_att.sum(1)
        # Fg mask (B, N, 1, H, W)
        alpha_map = (importance_map * alpha_att_hat).sum(dim=1) #! Eq.(49) of supl.
        # TODO (chmin): return alpha_att_hat to create depth map with the agent.
        return fg, alpha_map, _importance_map, y_att, alpha_att_hat





    def select(self, z_pres, *args):
        """

        Args:
            z_pres: (B, N, 1)
            *kargs: each (B, N, *) -> state_post, state_prior, z, ids, proposal

        Returns:ㅡ
            each (B, N, *)
        """
        # Take index
        # 1. sort by z_pres
        # 2. truncate
        # (B, N)
        indices = torch.argsort(z_pres, dim=1, descending=True)[..., 0] # sort along G*G dim
        # Truncate
        indices = indices[:, :ARCH.MAX]

        # Now use this thing to index every other thing
        def gather(x, indices):
            if len(x.size()) > len(indices.size()):
                indices = indices[..., None].expand(*indices.size()[:2], x.size(-1))
            return torch.gather(x, dim=1, index=indices)

        args = transform_tensors(args, func=lambda x: gather(x, indices))
        # return the sorted latents w.r.t. z_{pres}
        return args

    def select_kldiv(self, z_pres, kl_divs, slice_dim=16):
        """

        Args:
            z_pres: (B, N, 1)
            *kargs: each (B, N, *) -> state_post, state_prior, z, ids, proposal

        Returns:ㅡ
            each (B, N, *)
        """
        # Take index
        # 1. sort by z_pres
        # 2. truncate
        # (B, N)
        indices = torch.argsort(z_pres, dim=1, descending=True)[..., 0] # sort along G*G dim
        # Truncate
        indices = indices[:, :slice_dim]

        # Now use this thing to index every other thing
        def gather(x, indices):
            if len(x.size()) > len(indices.size()):
                indices = indices[..., None].expand(*indices.size()[:2], x.size(-1))
            return torch.gather(x, dim=1, index=indices)

        kl_divs = transform_tensors(kl_divs, func=lambda x: gather(x, indices))
        # return the sorted latents w.r.t. z_{pres}
        return kl_divs



    def combine(self,
                state_post_disc, state_prior_disc, z_disc, ids_disc, proposal_disc,
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal_prop):
        """
        Args: # TODO (chmin): debug the shape
            state_post_disc:
                h, c: (B, N, D)
            state_prior_disc:
                h, c: (B, N, D)
            z_disc: all (B, N, D)
                z_pres
                z_depth
                z_where
                z_what
            ids_prop: (B, N)

        Returns:
            state_post:
                h, c: (B, N, D)
            state_prior:
                h, c: (B, N, D)
            z:
                z_pres
                z_depth
                z_where
                z_what
        """

        def _combine(x, y):
            if isinstance(x, torch.Tensor):
                return torch.cat([x, y], dim=1)
            else: # combine list of torch.Tensors
                return [_combine(*pq) for pq in zip(x, y)]

        state_post, state_prior, z, ids, proposal, z_occ = _combine(
            [state_post_disc, state_prior_disc, z_disc, ids_disc, proposal_disc, z_occ_disc],
            [state_post_prop, state_prior_prop, z_prop, ids_prop, proposal_prop, z_occ_prop],
        )
        z_pres = z[0] # z is list of concatenated Tensors of z_disc and z_prop
        state_post, state_prior, z, ids, proposal, z_occ = self.select(z_pres, state_post, state_prior,
            z, ids, proposal, z_occ)

        state_post, state_prior, z, ids, proposal, z_occ = self.select(ids[..., None].clone().float(), state_post, state_prior,
            z, ids, proposal, z_occ)

        # sort by ids.
        return state_post, state_prior, z, ids, proposal, z_occ

    def bg_attention(self, bg, z):
        """
        AOE (Attention On Environment)

        Args:
            bg: (B, C, H, W)
            z:

        Returns:
            (B, N, D)

        """

        # (B, N, D)
        z_pres, z_depth, z_where, z_what, z_dyna = z
        B, N, _ = z_pres.size()

        if not ARCH.BG_CONDITIONED:
            return torch.zeros((B, N, ARCH.PROPOSAL_ENC_DIM), device=z_pres.device)

        if ARCH.BG_ATTENTION:
            # (G, G), (G, G)
            proposal = z_where.clone() # [B, T, 4 ]
            proposal[..., :2] += ARCH.BG_PROPOSAL_SIZE # add 0.25 for each [B, T, 2]

            # Get proposal glimpses
            # (B*N, 3, H, W)
            x_repeat = torch.repeat_interleave(bg, N, dim=0)

            # (B*N, 3, H, W) #! Eq.(4) & (5)
            proposal_glimpses = spatial_transform(x_repeat, proposal.view(B * N, 4),
                                                  out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))
            # (B, N, 3, H, W)
            proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)
            # (B, N, D)
            proposal_enc = self.bg_attention_encoder(proposal_glimpses)
        else:
            # (B, 3, H, W) -> (B, 1, 3, H, W)
            bg_tmp = bg[:, None]
            # (B, 1, D)
            proposal_enc = self.bg_attention_encoder(bg_tmp)
            proposal_enc = proposal_enc.expand(B, N, ARCH.PROPOSAL_ENC_DIM)

        return proposal_enc

    def rejection(self, z_disc, z_prop, threshold):
        """ # please refer to SCALOR paper
        If the bbox of an object overlaps too much with a propagated object, we remove it (z_disc)
        Args:
            z_disc: discovery; [B, N1 = G*G, D]
            z_prop: propagation; [B,N2, D]; where N2 is the # of propagated objects
            threshold: iou threshold

        Returns:
            z_disc
        """
        z_pres_disc, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc = z_disc
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
        # (B, N1, N2, 1) #? (self.iou(z_where_disc, z_where_disc) - torch.eye(16, device=z_pres_disc.device)[None, ..., None]).permute(0, 3, 1, 2)
        iou = self.iou(z_where_disc, z_where_prop) # In: [B, G*G, 4] self.render(z_disc)[0]
        assert torch.all((iou >= 0) & (iou <= 1))
        iou_too_high = iou > threshold # torch.bool Tensors
        # Only for those that exist (B, N1, N2, 1) (B, 1, N2, 1)
        iou_too_high = iou_too_high & (z_pres_prop[:, None, :] > 0.5) # update z_pres
        # (B, N1, 1)
        iou_too_high = torch.any(iou_too_high, dim=-2) # check if there's high IOU along N1 axis
        z_pres_disc_new = z_pres_disc.clone() # rejection mask
        z_pres_disc_new[iou_too_high] = 0.0 # if IOU > 0.8 and p(z_pres) > 0.5, reject z_disc

        return (z_pres_disc_new, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc)


    def self_rejection(self, z_disc, threshold):
            """ # please refer to SCALOR paper
            If the bbox of an object overlaps too much with a propagated object, we remove it (z_disc)
            Args:
                z_disc: discovery; [B, N1 = G*G, D]
                z_prop: propagation; [B,N2, D]; where N2 is the # of propagated objects
                threshold: iou threshold

            Returns:
                z_disc
            """
            z_pres_disc, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc = z_disc
            # z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
            # (B, G*G, G*G, 1) #? what is N1 and is N2 each?
            B, GG, *_ = z_where_disc.size()
            iou = self.iou(z_where_disc, z_where_disc) # In: [B, G*G, 4] iou.permute(0, 3, 1, 2)

            iou = torch.clamp(iou - torch.eye(GG, device=z_pres_disc.device)[None, ..., None], min=0.0)

            assert torch.all((iou >= 0) & (iou <= 1))
            iou_too_high = iou > threshold # torch.bool Tensors (iou > threshold).float().permute(0, 3, 1, 2)
            # Only for those that exist (B, N1, N2, 1) (B, 1, N2, 1) S.float().permute(0, 3, 1, 2)
            iou_too_high = iou_too_high & (z_pres_disc[:, None, :] > 1e-3) # update z_pres
            # (B, N1, 1)
            iou_too_high = torch.any(iou_too_high, dim=-2) # check if there's high IOU along N1 axis
            z_pres_disc_new = z_pres_disc.clone() # rejection mask iou_too_high[:, None].float().permute(0, 3, 1, 2)
            z_pres_disc_new[iou_too_high] = 0.0 # if IOU > 0.8 and p(z_pres) > 0.5, reject z_disc

            return (z_pres_disc_new, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc)



    def iou(self, z_where_disc, z_where_prop):
        """
        #! refer to SCALOR paper
        Args:
            z_where_disc: (B, N1, 4) -> N1 : G*G
            z_where_prop: (B, N2, 4)
        #! NOTE that z_{where} is split into z^{hw} and z^{xy}
        Returns:
            (B, N1, N2)
        """
        B, N1, _ = z_where_disc.size() #? N1: What does this mean?
        B, N2, _ = z_where_prop.size() #? N2:
        def _get_edges(z_where): # z_where_disc: [B, N1 (G*G), 1, 4] , z_where_prop: [B, 1, N2 (#obj), 4]
            z_where = z_where.detach().clone() # we dont' change the original variable
            z_where[..., 2:] = (z_where[..., 2:] + 1) / 2 # relocate X & Y [-1, 1] -> [0, 1]
            # (B, N, 1), (B, N, 1), (B, N, 1), (B, N, 1)
            sx, sy, cx, cy = torch.split(z_where, [1, 1, 1, 1], dim=-1)# h, w, x, y
            left = cx - sx / 2 # x - 2/w
            right = cx + sx / 2 # x + 2/w
            top = cy - sy / 2 # x - 2/h
            bottom = cy + sy / 2 # x + 2/h
            return left, right, top, bottom # z_w_disc - 4 * [B, N1, 1, 1] : z_w_prop - 4 * [B, N1, 1, 1]

        def _area(left, right, top, bottom):
            valid = (bottom >= top) & (right >= left)
            area = (bottom - top) * (right - left)
            area *= valid
            return area
        def _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2):
            #! make 1-1 comparison; 1st_input: where_disc [B, N1, 1, 1] / 2nd_input: where_prop [B, 1, N2, 1] || out:
            left = torch.max(left1, left2)
            right = torch.min(right1, right2)
            top = torch.max(top1, top2)
            bottom = torch.min(bottom1, bottom2)
            area1 = _area(left1, right1, top1, bottom1)
            area2 = _area(left2, right2, top2, bottom2)
            area_intersect = _area(left, right, top, bottom)
            iou = area_intersect / (area1 + area2 - area_intersect + 1e-5) #! Intersection / Union
            # If any of these areas are zero, we ignore this iou
            iou = iou * (area_intersect != 0) * (area1 != 0) * (area2 != 0) # If any of area is False, IoU == False
            return iou
        # In: (B, N1 (G*G), 1, D) - Out: (B, N1, 1, 1)
        left1, right1, top1, bottom1 = _get_edges(z_where_disc[:, :, None]) #? why expand rank 3?
        # (B, 1, N2, 1)
        left2, right2, top2, bottom2 = _get_edges(z_where_prop[:, None]) #? why expand rank 2?
        iou = _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2)
        assert iou.size() == (B, N1, N2, 1)
        return iou # (B, N1, N2, 1)


class ImgEncoder(nn.Module):
    """
    Used in discovery. Input is image plus image - bg
    """

    def __init__(self):
        super(ImgEncoder, self).__init__()

        assert ARCH.G in [8, 4]
        assert ARCH.IMG_SIZE in [64, 128]
        last_stride = ARCH.IMG_SIZE // (8 * ARCH.G)
        # last_stride = 1 if ARCH.G == 8 else 2


        self.last = nn.Conv2d(256, ARCH.IMG_ENC_DIM, 3, last_stride, 1)
        # 3 + 3 = 6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet18()
        self.enc = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4,
        )

    def forward(self, x):
        """
        Get image feature
        Args:
            x: (B, 3, H, W) -> shouldn't this be [B, 6, H , W]?

        Returns:
            enc: (B, 128, G, G)

        """
        B = x.size(0)
        x = self.enc(x)
        x = self.last(x)
        return x

class GlimpseEncoder(nn.Module):
    """
    Used in discovery. Input is image plus image - bg
    """

    def __init__(self):
        super(GlimpseEncoder, self).__init__()

        # last_stride = 1 if ARCH.G == 8 else 2

        # 3 + 3 = 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.last = nn.Conv2d(256, ARCH.IMG_ENC_DIM, 3, 2, 1)
        self.mlp = nn.Linear(64, 64)


        resnet = resnet18(pretrained=True)
        self.enc = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def forward(self, x):
        """
        Get image feature
        Args:
            x: (B, 3, H, W) -> shouldn't this be [B, 6, H , W]?

        Returns:
            enc: (B, 128, G, G)

        """
        B = x.size(0)
        x = self.enc(x)
        x = self.last(x)
        x = x.reshape(B, -1)
        x = self.mlp(x)
        return x



class LatentPostDisc(nn.Module):
    """q(z_dyna|x) in discovery"""

    def __init__(self):
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(ARCH.IMG_ENC_DIM + ARCH.PROP_MAP_DIM, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, ARCH.Z_DYNA_DIM * 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B = x.size(0)
        # (B, D, G, G)
        params = self.enc(x)
        # (B, G, G, D) -> (B, G*G, D)
        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4

        return z_dyna_loc, z_dyna_scale

    def forward_act(self, x, ee):
        """
        Args:
            x: (B, D, G, G) #! NOTE that discovery is based on glimpse
        # TODO (cheolhui) : how can I model condition action here?
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()

        # (B, D, G, G)
        params = self.enc(x)
        # (B, G*G, D)

        _params = []
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([params[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(params[:, n, ...]) # [B, 2D], Out: [B, D]
            _params.append(param)
        params = torch.stack(_params, dim=1)# [B, N, D]

        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4

        return z_dyna_loc, z_dyna_scale


class LatentPostProp(nn.Module):
    """q(z_dyna|x, z[:t]) in propagation"""

    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP(
            [ARCH.PROPOSAL_ENC_DIM + ARCH.RNN_HIDDEN_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())
        self.act_enc = MLP(
                [ARCH.PROPOSAL_ENC_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.ACTION_DIM, 128, 128,
                ARCH.Z_DYNA_DIM * 2
                ], act=nn.CELU())


    def forward(self, x):
        """
        Args:
            x: (B, N, 2D)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = self.enc(x) # [B, 2D], Out: [B, D]
        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1) # [B, N, D]
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4

        return z_dyna_loc, z_dyna_scale

    def forward_action(self, x, ee):
        """
        Args:
            x: (B, N, 2D)
            ee: (B, P)
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = []
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([x[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(x[:, n, ...]) # [B, 2D], Out: [B, D]
            params.append(param)
        params = torch.stack(params, dim=1)# [B, N, D]

        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1) # [B, N, D]
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4

        return z_dyna_loc, z_dyna_scale


class LatentPriorProp(nn.Module):
    """p(z_dyna|z[:t]) in propagation"""

    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP(
            [ARCH.RNN_HIDDEN_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())

        self.act_enc = MLP(
            [ARCH.RNN_HIDDEN_DIM + ARCH.ACTION_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())

    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B = x.size(0)
        params = self.enc(x)
        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4 #!Eq.(12) of supl

        return z_dyna_loc, z_dyna_scale

    def forward_action(self, x, ee):
        """
        Args:
            x: (B, N, D)
            ee: (B, P)
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = []
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([x[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(x[:, n, ...]) # [B, 2D], Out: [B, D]
            params.append(param)        # (B, G*G, D)
        params = torch.stack(params, dim=1) # [B, N, D]

        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4 #!Eq.(12) of supl

        return z_dyna_loc, z_dyna_scale


class PresDepthWhereWhatPrior(nn.Module):
    """p(z_att|z_dyna), where z_att = [z_pres, z_depth, z_where, z_what]"""

    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP( #! encodes z_dyna into z_pres, z_depth, z_what, z_where
            [ARCH.Z_DYNA_DIM, 128, 128,
             ARCH.Z_PRES_DIM + (ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM) * 2 + (
                     ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM
             )
             ], act=nn.CELU())

    def forward(self, enc):
        """
        Args:
            enc: (B, N, D)

        Returns:
            z_pres_prob: (B, N, 1)
            z_depth_loc, z_depth_scale: (B, N, 1)
            z_where_loc, z_where_scale: (B, N, 4)
            z_what_loc, z_what_scale: (B, N, D)
            z_depth_gate: (B, N, 1)
            z_where_gate: (B, N, 4)
            z_what_gate: (B, N, D)
        """
        # (B, N, D)
        params = self.enc(enc) #
        # split into [Z_PRES(1), Z_DEPTH(1), Z_DEPTH, Z_WHERE(4), Z_WHERE, Z_WHAT(64), Z_WHAT, Z_DEPTH, Z_WHERE, Z_WHAT] = 208
        (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale,
         z_what_loc, z_what_scale,
         z_depth_gate, z_where_gate_raw, z_what_gate) = torch.split(params,
                                [ARCH.Z_PRES_DIM] + [ARCH.Z_DEPTH_DIM] * 2 + [
                                    ARCH.Z_WHERE_DIM] * 2 + [
                                    ARCH.Z_WHAT_DIM] * 2 + [ARCH.Z_DEPTH_DIM,
                                                            ARCH.Z_WHERE_DIM,
                                                            ARCH.Z_WHAT_DIM]
                                , dim=-1) # len 10 list of dim [1, 1, 1, 4, 4, 64, 64, 1, 4, 64]
        z_pres_prob = torch.sigmoid(z_pres_prob) # [B, N, 1]
        z_depth_scale = F.softplus(z_depth_scale) + 1e-4 # [B, N, 1]; stddev should be positive
        z_where_scale = F.softplus(z_where_scale) + 1e-4 # [B, N, 4]; stddev should be positive
        z_what_scale = F.softplus(z_what_scale) + 1e-4 # [B, N, 64]; stddev should be positive
        # TODO (cheolhui): check the effect of change in gating range.
        z_depth_gate = torch.sigmoid(z_depth_gate) # [B, N, 1]
        z_where_gate = torch.sigmoid(z_where_gate_raw) # [B, N, 4]
        z_what_gate = torch.sigmoid(z_what_gate) # [B, N, 64]

        return (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale, z_what_loc, 
            z_what_scale, z_depth_gate, z_where_gate, z_what_gate, z_where_gate_raw)


class PredProposal(nn.Module):
    """Given states encoding previous history, compute a proposal location."""

    def __init__(self):
        nn.Module.__init__(self)
        self.net = MLP(
            sizes=[ARCH.RNN_HIDDEN_DIM, 128, 128, 2],
            act=nn.CELU(),
        )
        self.net = BatchApply(self.net)

    def forward(self, enc):
        """
        Args:
            enc: (B, N, D)

        Returns:
            proposal: (B, N, 2). This is offset from previous where.
        """
        return self.net(enc)


class ProposalEncoder(nn.Module):
    """Same as glimpse encoder, but for encoding the proposal area"""

    def __init__(self):
        nn.Module.__init__(self)
        embed_size = ARCH.GLIMPSE_SIZE // 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
        )
        # TODO (cheolhui): robot action should be conditioned independently
        self.enc_what = nn.Linear(128 * embed_size ** 2, ARCH.PROPOSAL_ENC_DIM)

    # def forward(self, x):
    #     """
    #     Args:
    #         x: (B, N, 3, H, W)

    #     Returns:
    #         enc: (B, N, D)
    #     """
    #     B, N, C, H, W = x.size()
    #     x = x.reshape(B * N, 3, H, W)
    #     x = self.enc(x) # [B*N, D, 1, 1]
    #     x = x.flatten(start_dim=1) # [B*N, D]
    #     return self.enc_what(x).view(B, N, ARCH.PROPOSAL_ENC_DIM)
    def forward(self, x):
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            enc: (B, N, D)
        """
        x_size = x.size()
        H, W = x_size[-2], x_size[-1]
        x = x.reshape(-1, 3, H, W)
        x = self.enc(x) # [B*N, D, 1, 1]
        x = x.flatten(start_dim=1) # [B*N, D]

        new_size = x_size[:-3] + (ARCH.PROPOSAL_ENC_DIM, )
        return self.enc_what(x).view(*new_size)




    def forward_action(self, x, ee):
        """
        Infer action-conditioned proposal
        Args:
            x: (B, N, 3, H, W)
            ee: (B, P), P: end-effector dim
        Returns:
            enc: (B, N, D)
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, 3, H, W)
        x = self.enc(x) # [B*N, D, 1, 1] # TODO: seperate first entity to be conditioned on action
        x = x.flatten(start_dim=1) # [B*N, D]
        return self.enc_what(x).view(B, N, ARCH.PROPOSAL_ENC_DIM)


class GlimpseDecoder(nn.Module):
    """Decode z_what into glimpse."""

    def __init__(self):
        nn.Module.__init__(self)
        # Everything here is symmetric to encoder, but with subpixel upsampling

        self.embed_size = ARCH.GLIMPSE_SIZE // 16 # 1
        self.fc = nn.Linear(ARCH.Z_WHAT_DIM, self.embed_size ** 2 * 128)
        self.net = nn.Sequential(
            nn.Conv2d(128, 64 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 64),

            nn.Conv2d(64, 32 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(2, 32),

            nn.Conv2d(32, 16 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(1, 16),

            # Mask and appearance
            nn.Conv2d(16, 4 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, z_what):
        """

        Args:
            z_what: (B, D), where B = batch * (#objects)

        Returns:
            glimpse: (B, 3, H, W)
        """
        B, D = z_what.size()
        x = F.celu(self.fc(z_what))
        #  -> (B, 128, E, E)
        x = x.view(B, 128, self.embed_size, self.embed_size)
        x = self.net(x)
        x = torch.sigmoid(x)
        # (B, 3, H, W), (B, 1, H, W), where H = W = GLIMPSE_SIZE
        o_att, alpha_att = x.split([3, 1], dim=1)

        return o_att, alpha_att


class PresDepthWhereWhatPostLatentDisc(nn.Module):
    """Predict attributes posterior given image encoding and propagation map in discovery"""

    def __init__(self):
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(ARCH.IMG_ENC_DIM + ARCH.PROP_MAP_DIM, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128,
                      ARCH.Z_PRES_DIM + (ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM) * 2,
                      1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_pres_prob: (B, G*G, 1)
            z_depth_loc, z_depth_scale: (B, G*G, 1)
            z_where_loc, z_where_scale: (B, G*G, D)
            z_what_loc, z_what_scale: (B, G*G, D)
            z_dyna_loc, z_dyna_scale: (B, G*G, D)
        """
        B = x.size(0)
        # (B, D, G, G)
        params = self.enc(x)
        # (B, G*G, D)
        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale,
         z_what_loc, z_what_scale, z_dyna_loc, z_dyna_scale) = torch.split(params,
                                            [ARCH.Z_PRES_DIM] + [
                                                ARCH.Z_DEPTH_DIM] * 2 + [
                                                ARCH.Z_WHERE_DIM] * 2 + [
                                                ARCH.Z_WHAT_DIM] * 2 + [
                                                ARCH.Z_DYNA_DIM] * 2, dim=-1)
        z_pres_prob = torch.sigmoid(z_pres_prob)
        z_where_scale = F.softplus(z_where_scale) + 1e-4
        z_depth_scale = F.softplus(z_depth_scale) + 1e-4
        z_what_scale = F.softplus(z_what_scale) + 1e-4
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4

        return z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale, z_what_loc, z_what_scale, z_dyna_loc, z_dyna_scale


class BgAttentionEncoder(nn.Module):
    """Encoding (attended) background region"""

    def __init__(self):
        nn.Module.__init__(self)
        if ARCH.BG_ATTENTION:
            embed_size = ARCH.GLIMPSE_SIZE // 16
        else:
            embed_size = ARCH.IMG_SIZE // 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
        )

        self.enc_what = nn.Linear(128 * embed_size ** 2, ARCH.BG_PROPOSAL_DIM)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            enc: (B, N, D)
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, 3, H, W)
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        return self.enc_what(x).view(B, N, ARCH.BG_PROPOSAL_DIM)

