import torch
from torch import nn
from torch.autograd.grad_mode import no_grad
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .module import Flatten, MLP
from torchvision.models import resnet18
from gatsbi_rl.gatsbi.arch import ARCH
from gatsbi_rl.gatsbi.utils import bcolors
from contextlib import nullcontext

class MixtureModule(nn.Module):
    """ 
    Mixture module of GATSBI.
    """
    def __init__(self, action_dim=7):
        nn.Module.__init__(self)
        self.embed_size = ARCH.IMG_SIZE // 16
        # Embeds sequential images
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3),
            nn.CELU(),
            nn.GroupNorm(num_groups=4, num_channels=64),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, 512),
        )
        #! experimental
        self.mode_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3),
            nn.CELU(),
            nn.GroupNorm(num_groups=4, num_channels=64),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, 512),
        )
        self.enc_fc = nn.Linear(self.embed_size ** 2 * 512, ARCH.IMG_ENC_DIM)
        #! experimental
        self.mode_enc_fc = nn.Linear(self.embed_size ** 2 * 512, ARCH.IMG_ENC_DIM)

        # simultaneously update the RNN states of all the K entities.
        self.rnn_mask_t_post = nn.LSTMCell(ARCH.Z_MASK_DIM, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        self.rnn_mask_t_prior = nn.LSTMCell(ARCH.Z_MASK_DIM, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        # temporal encoding of comp latents
        self.rnn_comp_t_post = nn.LSTMCell(ARCH.Z_COMP_DIM, ARCH.RNN_CTX_COMP_HIDDEN_DIM)
        self.rnn_comp_t_prior = nn.LSTMCell(ARCH.Z_COMP_DIM, ARCH.RNN_CTX_COMP_HIDDEN_DIM)

        # nn Params for scene entities' detr. transition of mask via RNN
        self.h_mask_RNN_t_post = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.h_mask_RNN_t_prior = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_RNN_t_post = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_RNN_t_prior = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))

        # nn Params for scene entities' detr. transition of comp via RNN
        self.h_comp_RNN_t_post = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.h_comp_RNN_t_prior = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.c_comp_RNN_t_post = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.c_comp_RNN_t_prior = nn.Parameter(torch.rand(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))

        # register the initialized RNN states after detached rollouts.
        # define placeholders for them.
        # encodes the posterior input and the action of the agent for k=0,mask inference.
        self.post_net_first = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.IMG_ENC_DIM + ARCH.ACTION_ENHANCE,
                               128, 128, ARCH.MASK_COND_HIDDEN_DIM], act=nn.CELU())

        # encodes the posterior input for the other entities (k=[1:K-1])
        self.post_net_t = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.IMG_ENC_DIM, 128, 128, ARCH.MASK_COND_HIDDEN_DIM],
                              act=nn.CELU())

        # NN that encodes the posterior input and the action of the agent for k=0
        self.prior_net_first = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM
            + ARCH.ACTION_ENHANCE + ARCH.Z_MASK_DIM,
            128, 128, ARCH.MASK_COND_HIDDEN_DIM], act=nn.CELU())

        self.prior_net_t = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM
            + ARCH.Z_MASK_DIM, 128, 128, ARCH.MASK_COND_HIDDEN_DIM], act=nn.CELU())

        self.rnn_mask_post_k = nn.LSTMCell(ARCH.Z_MASK_DIM + ARCH.MASK_COND_HIDDEN_DIM, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        self.h_mask_k_post = nn.Parameter(torch.rand(ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_k_post = nn.Parameter(torch.rand(ARCH.RNN_CTX_MASK_HIDDEN_DIM))

        # Dummy z_mask for first step of rnn_mask
        self.z_mask_0 = nn.Parameter(torch.rand(ARCH.Z_MASK_DIM))
        self.z_comp_0 = nn.Parameter(torch.rand(ARCH.Z_COMP_DIM))
        # Predict mask latent given h
        self.predict_mask = PredictMask()

        # Compute masks given mask latents
        self.mask_cond_dec = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, ARCH.Z_MASK_DIM)
        )
        self.mask_decoder = MaskDecoder()
        # Encode mask and image into component latents
        self.comp_cond_encoder = CompCondEncoder()
        self.comp_cond_decoder = CompCondDecoder()
        # Component decoder
        self.rnn_mask_prior_k = nn.LSTMCell(ARCH.Z_MASK_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        # Initial h and c
        self.h_mask_k_prior = nn.Parameter(torch.rand(ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_k_prior = nn.Parameter(torch.rand(ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        # Compute mask latents
        self.predict_mask_prior = PredictMaskPrior()
        # Compute component latents
        self.predict_comp_prior = PredictComp()
        # ==== Prior related ====
        self.bg_sigma = ARCH.BG_SIGMA # TODO (chmin): deprecated. Use ARCH.SIGMA instead.
        self.agent_sigma = ARCH.BG_SIGMA # It will be udpated as bg_sigma

        # encode the concat of conditioning observations
        self.cond_obs_cat_encoder = nn.Sequential(
            nn.Linear(5 * ARCH.IMG_ENC_DIM, ARCH.IMG_ENC_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.IMG_ENC_HIDDEN_DIM, ARCH.IMG_ENC_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.IMG_ENC_HIDDEN_DIM, ARCH.IMG_ENC_DIM)
        )

        # embed the (recon.) obs for RNN update of mask latent
        self.obs_mask_enc_fn = nn.Sequential(
                 nn.Linear(ARCH.IMG_ENC_DIM + ARCH.Z_MASK_DIM, ARCH.IMG_ENC_HIDDEN_DIM),
                 nn.CELU(),
                 nn.Linear(ARCH.IMG_ENC_HIDDEN_DIM, ARCH.Z_MASK_DIM)
        )

        # embed the (recon.) obs for RNN update of comp latent
        self.obs_comp_enc_fn = nn.Sequential(
                 nn.Linear(ARCH.IMG_ENC_DIM + ARCH.Z_COMP_DIM, ARCH.IMG_ENC_HIDDEN_DIM),
                 nn.CELU(),
                 nn.Linear(ARCH.IMG_ENC_HIDDEN_DIM, ARCH.Z_COMP_DIM)
        )

        # TODO (chmin): vanilla gatsbi has no scale.
        # residual update network of z^m_{k,t}
        self.mask_residual_update = nn.Sequential(
            nn.Linear(2 * ARCH.Z_MASK_DIM, ARCH.RES_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.RES_HIDDEN_DIM, ARCH.RES_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.RES_HIDDEN_DIM ARCH.Z_MASK_DIM)
        )
        
        # TODO (chmin): vanilla gatsbi has no scale.
        # residual update network of z^c_{k,t}
        self.comp_residual_update = nn.Sequential(
            nn.Linear(2 * ARCH.Z_COMP_DIM, ARCH.RES_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.RES_HIDDEN_DIM, ARCH.RES_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.RES_HIDDEN_DIM ARCH.Z_COMP_DIM)
        )

        # enhance the dimension of the action  of z^c_{k,t}
        self.action_enhance = nn.Sequential(
            nn.Linear(ARCH.ACTION_DIM, ARCH.ACTION_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.ACTION_HIDDEN_DIM, ARCH.ACTION_ENHANCE)
        )

        # inv_dyna_model: given z^{m}_{t+1}, h^{m}_{t+1}, z^{m}_{t}, infer a_t
        self.inv_dyna_pred = nn.Sequential(
            nn.Linear(2 * ARCH.Z_MASK_DIM + ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.INV_DYNA_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.INV_DYNA_HIDDEN_DIM, ARCH.INV_DYNA_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.INV_DYNA_HIDDEN_DIM, ARCH.ACTION_DIM)
        )

        self.get_agent_depth_init = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE +
                ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, 2 * ARCH.Z_DEPTH_DIM)
        )

        self.get_agent_depth = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE +
                ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, 3 * ARCH.Z_DEPTH_DIM)
        )

    def anneal(self, global_step):
        self.global_step = global_step

    def forward(self, seq):
        return self.encode(seq)

    def get_initial_state(self):
        """
            Return the initial state of recurrent states of the 
            GATSBI model.
        """
        return (2 * self.h_mask_RNN_t_post.expand(B * ARCH.K,
                ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 , 
            2 * self.c_mask_RNN_t_post.expand(B * ARCH.K,
                ARCH.RNN_CTX_MASK_HIDDEN_DIM)-1,
            2 * self.h_comp_RNN_t_post.expand(B * ARCH.K,
                ARCH.RNN_CTX_COMP_HIDDEN_DIM)-1, 
            2 * self.c_comp_RNN_t_post.expand(B * ARCH.K,
                ARCH.RNN_CTX_COMP_HIDDEN_DIM)-1,
            2 * self.h_mask_RNN_t_prior.expand(B * ARCH.K,
                ARCH.RNN_CTX_MASK_HIDDEN_DIM)-1,
            2 * self.c_mask_RNN_t_prior.expand(B * ARCH.K,
                ARCH.RNN_CTX_MASK_HIDDEN_DIM)-1,
            2 * self.h_comp_RNN_t_prior.expand(B * ARCH.K,
                ARCH.RNN_CTX_COMP_HIDDEN_DIM)-1,
            2 * self.c_comp_RNN_t_prior.expand(B * ARCH.K,
                ARCH.RNN_CTX_COMP_HIDDEN_DIM)-1
                ) 


    def infer(self, history, obs, action, episodic_step=0, first=False, agent_slot=0, agent_depth_prev=None):
        """
            Single step inference for the policy of the agent.
            args:
                history: history containing temporal states for the agent. 
                    (h_mask_t_post, c_mask_t_post, h_comp_t_post, c_comp_t_post)
                obs: observation @ t. [B, C, H, W]
                action: action @ t-1.
                episodic_step: the step of current epsiode.
        """
        B, C, H, W = obs.size() #! 
        A = action.size(-1)
        action = torch.squeeze(action, 1)
        action = self.action_enhance(action) # [B, T, A]

        history = [torch.squeeze(t, 1) for t in history] # reduce the temporal axis
        z_masks_prev, z_comps_prev, \
                h_mask_t_post, c_mask_t_post, h_comp_t_post, c_comp_t_post = history

        # encode the observation of shape [1, 3, 64, 64].
        enc = self.enc(obs)
        obs = obs.view(B, 3, H, W)
        enc = enc.flatten(start_dim=1) # [B, D]

        enc = self.enc_fc(enc)
        enc = enc.view(B, ARCH.IMG_ENC_DIM) # [B, 128]

        if first:
            setattr(self, 'agent_depth_raw_prev', torch.zeros(B, 1).to(obs.device))

        # TODO (Chmin)
        if episodic_step == 0:
            enc = self.cond_obs_cat_encoder(enc.clone().repeat(1, 5))

        # spatial update
        z_mask_k_prev = 2 * self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) -1 # first mask latent for autoreg. update.

        # states for autoreg. spatial LSTM
        h_mask_k_post = 2 * self.h_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # spatial
        c_mask_k_post = 2 * self.c_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # spatial

        masks = [] # list to append mask recons
        z_masks = [] # list to append mask latents
        z_comps = [] # list to append comp latents

        # ! 1) Compute posteriors
        # iterate over K entities, infer autoregressive mask latents
        for k in range(ARCH.K):
            if k == 0: # agent entity.
                post_input = torch.cat([h_mask_t_post[:, k], enc, action], dim=-1)
                post_input = self.post_net_first(post_input)
            else: # other entities
                post_input = torch.cat([h_mask_t_post[:, k], enc], dim=-1)
                post_input = self.post_net_t(post_input)  # k = {1:K}
            # concat posterior info with spatial latent. -> for autoreg. update
            rnn_input = torch.cat([post_input, z_mask_k_prev], dim=-1)
            # autoregressive update of mask hidden state h^m_{t,k-1} -> h^m_{t,k} 
            (h_mask_k_post, c_mask_k_post) = self.rnn_mask_post_k(rnn_input, (
                h_mask_k_post, c_mask_k_post))
            # predict mask latent variable z^m_{t,k} from h^m_{t,k}.
            z_mask_loc, z_mask_scale = self.predict_mask(h_mask_k_post)
            z_mask_post = Normal(z_mask_loc, z_mask_scale) # [B, Zm]
            z_mask = z_mask_post.rsample()  # z^m_t
            # for t >= 1; residual update of the mask variable: z^m_{t+1,k} <- z^m_{t,k} + f(z^m_{t+1,k}, z^m_{t,k}) 
            if episodic_step > 0 and z_masks_prev is not None: # z_masks_prev is updated after t > 1.
                res_m_latent = self.mask_residual_update(torch.cat([z_masks_prev[:, k], z_mask], dim=-1))
                z_mask = torch.add(z_masks_prev[:, k], ARCH.RESIDUAL_SCALE * res_m_latent)
            z_masks.append(z_mask)
            z_mask_k_prev = z_mask  # for autoregressive update.
            # condition action for decoding (cVAE scheme) 
            z_mask = self.mask_cond_dec(torch.cat([z_mask, action], dim=-1))
            mask = self.mask_decoder(z_mask)  # In - [B, Zm], Out - [B, 1, H, W]
            # append the mask recons.
            masks.append(mask)

        # concat z_mask along the spatial dim (K) (add new dim by stacking)
        z_masks_cat = torch.stack(z_masks, dim=1) # [B, K, Zm] 
        masks = torch.stack(masks, dim=1) # [B, K, 1, H, W] in range [0, 1]
        # SBP to ensure to be summed up to 1.
        if ARCH.BG_DECOMP == 'sbp':
            masks = self.SBP(masks)
        # empirically, inverse order SBP also works well.
        elif ARCH.BG_DECOMP == 'inv_sbp':
            masks = self.SBP_inv(masks)

        B, K, _, H, W = masks.size() # get the shape 

        # reshape (B, K, 1, H, W) -> (B*K, 1, H, W)
        masks = masks.view(B * K, 1, H, W)
        # concatenate images along channel axis (B*K, 4, H, W)
        comp_vae_input = torch.cat([(masks + 1e-5).log(),
            obs[:, None].repeat(1, K, 1, 1, 1).view(B * K, 3, H, W)], dim=1)
        # component latents, each [B*K, L]. action [B, A], h_comp_t_post [B*K, Hc]
        z_comp_loc, z_comp_scale = self.comp_cond_encoder(comp_vae_input, 
            action, h_comp_t_post)  # conditional comp encoding
        # get posterior distribution of shape [B, K, Zc]
        z_comp_loc, z_comp_scale = z_comp_loc.reshape(B, K, -1), z_comp_scale.reshape(B, K, -1)
        z_comp_post = Normal(z_comp_loc, z_comp_scale)
        z_comps_raw = z_comp_post.rsample() # [B, k, Zc]
        # iterate over spatial component latents
        if episodic_step > 0 and z_comps_prev is not None: # residual exists for t > 0.
            for k in range(ARCH.K):
                res_c_latent = self.comp_residual_update(torch.cat([z_comps_prev[:, k], z_comps_raw[:, k]], dim=-1))
                z_comp = torch.add(z_comps_prev[:, k], ARCH.RESIDUAL_SCALE * res_c_latent)
                z_comps.append(z_comp)
            # concatenate the component latents
            z_comps_cat = torch.stack(z_comps, dim=1) # [B, K, Zc]
        else:
            z_comps_cat = z_comps_raw
        # condition action on decoding component latent; 
        comps = self.comp_cond_decoder(z_comps_cat.view(B * K, -1), action)

        # Decode into component images, [B*K, 3, H, W]
        comps = comps.view(B, K, 3, H, W)
        masks = masks.view(B, K, 1, H, W)
        modes = comps * masks # [B, K, 3, H, W]
        bg = modes.sum(dim=1)

        modes = modes.reshape(B * ARCH.K, 3, H, W)
        mode_enc = self.mode_enc(modes)
        mode_enc = mode_enc.flatten(start_dim=1)  # [B * K, D]
        bg_enc = self.mode_enc_fc(mode_enc).reshape(B, ARCH.K, -1)
        # # generate background recons.

        # update the temporal history state.
        # ! 3-1) temporal encode of z_mask & z_comp into h_post & h_prior -> update temporal latents along T-axis
        # the RNN update should be 
        h_mask_t_post = h_mask_t_post.reshape(B * ARCH.K, -1)
        c_mask_t_post = c_mask_t_post.reshape(B * ARCH.K, -1)

        # expand the dimension 
        temp_mask_inpt = torch.cat([z_masks_cat, bg_enc], dim=-1) # [B, K, D + Zm]
        temp_mask_inpt = temp_mask_inpt.view(B * ARCH.K, -1)
        # encode the two inputs into one: [B, K, D + Zm] -> [B, K, Zm]
        temp_mask_inpt = self.obs_mask_enc_fn(temp_mask_inpt)

        # temporal update of rnn hidden states of mask prior and posterior
        h_mask_t_post, c_mask_t_post = self.rnn_mask_t_post(temp_mask_inpt, (h_mask_t_post, c_mask_t_post))

        # initialize the RNN hidden states.
        h_comp_t_post = h_comp_t_post.reshape(B * ARCH.K, -1)
        c_comp_t_post = c_comp_t_post.reshape(B * ARCH.K, -1)

        # expand the dimension of background encoding for RNN update of masks of K entities.
        temp_comp_inpt = torch.cat([z_comps_cat, bg_enc], dim=-1)
        temp_comp_inpt = temp_comp_inpt.view(B * ARCH.K, -1)
        # encode the two inputs into one: [B, K, D + Zc] -> [B, K, Zc]
        temp_comp_inpt = self.obs_comp_enc_fn(temp_comp_inpt)

        # temporal update of rnn hiddeen states 
        # h_comp_t_post, c_comp_t_post = self.rnn_comp_t_post(temp_comp_inpt, (h_comp_t_post, c_comp_t_post))
        # TODO (chmin): check if we can use prior network for posterior inference.
        h_comp_t_post, c_comp_t_post = self.rnn_comp_t_prior(temp_comp_inpt, (h_comp_t_post, c_comp_t_post))

        # get agent_depth here.
        depth_inpt = torch.cat([z_masks_cat[:, agent_slot], h_mask_t_post[None][:, agent_slot], action], dim=-1) # [1, D]
        if first:
            agent_depth_out = self.get_agent_depth_init(depth_inpt)
            agent_depth_first_loc, agent_depth_first_scale = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 2, dim=-1)
            agent_depth_raw_dist_first = Normal(agent_depth_first_loc, agent_depth_first_scale)
            agent_depth_raw = agent_depth_raw_dist_first.rsample()
        else:
            agent_depth_out = self.get_agent_depth(depth_inpt)
            agent_depth_loc, agent_depth_scale, agent_depth_gate = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 3, dim=-1)
            agent_depth_raw_dist = Normal(agent_depth_loc, agent_depth_scale)
            agent_depth_raw = agent_depth_raw_dist.rsample()
            agent_depth_gate = torch.sigmoid(agent_depth_gate)
            agent_depth_raw = self.agent_depth_raw_prev + ARCH.Z_DEPTH_UPDATE_SCALE * agent_depth_gate * agent_depth_raw # [B, D, 1]

        setattr(self, 'agent_depth_raw_prev', agent_depth_raw)

        things = {
            'z_comps': z_comps_cat,
            'z_masks': z_masks_cat,
            'comps': comps,
            'masks': masks,
            'agent_depth_raw': agent_depth_raw,
            'h_mask_post': h_mask_t_post,
            'c_mask_post': c_mask_t_post,
            'h_comp_post': h_comp_t_post,
            'c_comp_post': c_comp_t_post,
            'bg' : bg,
            'enhanced_act': action
        }
        return things

    def get_init_recur_state(self):
        """
        Return the recurrent states.
        """
        h_mask_t_post = 2 * self.h_mask_RNN_t_post.expand(1, ARCH.K,
            ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1   # posterior temporal mask hidden state - [B, Hm]
        c_mask_t_post = 2 * self.c_mask_RNN_t_post.expand(1, ARCH.K,
            ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # posterior temporal mask cell state: shape - [B, Hm]
        h_comp_t_post = 2 * self.h_comp_RNN_t_post.expand(1, ARCH.K,
            ARCH.RNN_CTX_COMP_HIDDEN_DIM) -1 # posterior temporal comp hidden state: shape - [B, Hc]
        c_comp_t_post = 2 * self.c_comp_RNN_t_post.expand(1, ARCH.K,
            ARCH.RNN_CTX_COMP_HIDDEN_DIM) -1 # posterior temporal comp cell state: shape - [B, Hc]
        h_mask_t_prior = 2 * self.h_mask_RNN_t_prior.expand(1, ARCH.K,
            ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # prior temporal mask hidden state: shape - [B, Hm]
        c_mask_t_prior = 2 * self.c_mask_RNN_t_prior.expand(1, ARCH.K,
            ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # prior temporal mask cell state: shape - [B, Hm]
        h_comp_t_prior = 2 * self.h_comp_RNN_t_prior.expand(1, ARCH.K,
            ARCH.RNN_CTX_COMP_HIDDEN_DIM) -1 # prior temporal comp hidden state: shape - [B, Hc]
        c_comp_t_prior = 2 * self.c_comp_RNN_t_prior.expand(1, ARCH.K,
            ARCH.RNN_CTX_COMP_HIDDEN_DIM) -1 # prior temporal comp cell state: shape - [B, Hc]

        state_dict = {
            'post' : (h_mask_t_post, c_mask_t_post, h_comp_t_post, c_comp_t_post),
            'prior' : (h_mask_t_prior, c_mask_t_prior, h_comp_t_prior, c_comp_t_prior),

        }
        return state_dict

    def update_agent_sigma(self, agent_idx):
        """
        Update the stddev of the agent slot after the index is found.
        """
        # TODO (chmin): print the log!
        self.agent_sigma = ARCH.AGENT_SIGMA
        print(bcolors.OKCYAN + "The stddev of agent slot {0} has been set as {1}".format(agent_idx, self.agent_sigma))

    def encode(self, seq, action, global_step=0, agent_idx=0, leverage=False, model_T=0, agent_slot=0):
        """
        Background inference backward pass
        :param seq: shape (B, T, C, H, W) it's a crop of o_{t}:t+{t+T-1}
        :param action: shape (B, T + 1, P) it's a crop of a_{t-1}:t+{t+T-1}
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization.
        """
        B, T, C, H, W = seq.size()
        A = action.size(-1)
        if A != ARCH.ACTION_ENHANCE:
            action = self.action_enhance(action)  # [B, T, A]
        seq = seq.reshape(B * T, 3, H, W)
        # encode the whole sequence.
        enc = self.enc(seq)
        seq = seq.view(B, T, 3, H, W)
        enc = enc.flatten(start_dim=1)  # (B*T, D)
        enc = self.enc_fc(enc)  # [B, T, 128]

        enc = enc.view(B, T, ARCH.IMG_ENC_DIM)  # [B, T, 128]

        # TODO (chmin): first step inference should be 
        obs_cat_enc = self.cond_obs_cat_encoder(enc[:, :5].view(B, 5 * ARCH.IMG_ENC_DIM))

        # TODO (chmin): the input should be compatiable with infer() method.
        # concat of observation over first 5 steps: input for z^m_0 and z^c_0.

        # mask and component latents over the K slots
        # spatial initial RNN paramters

        h_mask_t_post = 2 * self.h_mask_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1  # posterior temporal mask hidden state - [B, Hm]
        c_mask_t_post = 2 * self.c_mask_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # posterior temporal mask cell state: shape - [B, Hm]
        h_comp_t_post = 2 * self.h_comp_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM) - 1 # posterior temporal comp hidden state: shape - [B, Hc]
        c_comp_t_post = 2 * self.c_comp_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM) - 1  # posterior temporal comp cell state: shape - [B, Hc]
        h_mask_t_prior = 2 * self.h_mask_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1  # prior temporal mask hidden state: shape - [B, Hm]
        c_mask_t_prior = 2 * self.c_mask_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1  # prior temporal mask cell state: shape - [B, Hm]
        h_comp_t_prior = 2 * self.h_comp_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM) - 1  # prior temporal comp hidden state: shape - [B, Hc]
        c_comp_t_prior = 2 * self.c_comp_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM) - 1 # prior temporal comp cell state: shape - [B, Hc]

        # initialize lists to append temporal info.
        z_masks_t = [] # list to append the mask latent over T.
        z_comps_t = [] # list to append the component latent over T.
        res_m_latents_t = [] # list the append the residual tensor. Used for regularlization.
        res_m_scales_t = [] # list the append the residual tensor. Used for regularlization.
        res_c_latents_t = [] # list the append the residual tensor. Used for regularlization.
        res_c_scales_t = [] # list the append the residual tensor. Used for regularlization.
        masks_t = [] # list to append mask reconstruction. Used for visualization.
        comps_t = [] # list to append comp reconstruction. Used for visualization.
        bgs = []  # list of background reconstruction Tensors
        z_mask_total_kl_k_t = []  # list of mask kls over K and T axes
        z_comp_total_kl_k_t = []  # list of comp kls over K and T axes

        mask_inputs_prior_t = []
        comp_inputs_prior_t = []
        h_masks_t = [] # list of rnn hidden state of mask latents. Used to form agent node on GNN.
        h_comps_t = [] # list of rnn hidden state of comp latents. Used for policy feature.
        #! cell states are used for imagination trajectory
        c_masks_t = []
        c_comps_t = []
        # temporal update
        z_masks_prev = None # placeholder for the temporal concat of mask latent.
        z_comps_prev = None # placeholder for the temporal concat of comp latent.

        agent_depth_raw_list = []
        agent_depth_raw_list_kl = []
        # Iterate over prediction horizon T.

        detached_timesteps = T - model_T

        for t in range(T): # T
            noisy_train_context = torch.no_grad() if leverage and t < detached_timesteps \
                else nullcontext()
            
            if t == detached_timesteps:
                rest_here = 0 # TODO (chmin): remove this after debuging.

            with noisy_train_context:
                # reshape the temporal RNN hidden states.
                h_mask_t_post = h_mask_t_post.view(B, ARCH.K, -1)
                c_mask_t_post = c_mask_t_post.view(B, ARCH.K, -1)
                h_comp_t_post = h_comp_t_post.view(B, ARCH.K, -1)
                c_comp_t_post = c_comp_t_post.view(B, ARCH.K, -1)

                h_mask_t_prior = h_mask_t_prior.view(B, ARCH.K, -1)
                c_mask_t_prior = c_mask_t_prior.view(B, ARCH.K, -1)
                h_comp_t_prior = h_comp_t_prior.view(B, ARCH.K, -1)
                c_comp_t_prior = c_comp_t_prior.view(B, ARCH.K, -1)

                # spatial update
                z_mask_k_prev = 2 * self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) - 1 # first mask latent for autoreg. update.

                # states for autoreg. spatial LSTM
                h_mask_k_post = 2 * self.h_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # spatial
                c_mask_k_post = 2 * self.c_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # spatial


                masks = [] # list to append mask recons
                z_masks = [] # list to append mask latents
                z_comps = [] # list to append comp latents
                res_m_latents = []  # for regularization losses of mask residual latent.
                res_m_scales = []  # for regularization losses of mask residual latent.
                res_c_latents = []  # for regularization losses of comp residual latent.
                res_c_scales = []  # for regularization losses of comp residual latent.

                z_mask_posteriors = [] # list to append post. mask distribs. Used for KL div. computation
                z_comp_posteriors = [] # list to append post. comp distribs. Used for KL div. computation
                z_comp_posteriors_mat = [] # list to append post. comp distribs. Used for KL div. computation

                # TODO (chmin): experimental. reguralize prior network scales.
                mask_inputs_prior = []
                comp_inputs_prior = []

                if t == 0:
                    post_obs = obs_cat_enc 
                else:
                    post_obs = enc[:, t]

                # ! 1) Compute posteriors
                # iterate over K entities, infer autoregressive mask latents
                for k in range(ARCH.K):
                    if k == 0: # autoregressively embed action. a_{t-1}
                        post_input = torch.cat([h_mask_t_post[:, k], post_obs, action[:, t]], dim=-1)
                        post_input = self.post_net_first(post_input)
                    else: # other entities
                        post_input = torch.cat([h_mask_t_post[:, k], post_obs], dim=-1)
                        post_input = self.post_net_t(post_input)  # k = {1:K}
                    # concat posterior info with spatial latent. -> for autoreg. update
                    rnn_input = torch.cat([post_input, z_mask_k_prev], dim=-1)
                    # autoregressive update of mask hidden state h^m_{t,k-1} -> h^m_{t,k} 
                    (h_mask_k_post, c_mask_k_post) = self.rnn_mask_post_k(rnn_input, (
                        h_mask_k_post, c_mask_k_post))
                    # predict mask latent variable z^m_{t,k} from h^m_{t,k}.
                    z_mask_loc, z_mask_scale = self.predict_mask(h_mask_k_post)
                    z_mask_post = Normal(z_mask_loc, z_mask_scale) # [B, Zm]
                    z_mask = z_mask_post.rsample()  # z^m_t

                    # TODO (chmin): residual update seems broken here!
                    # for t >= 1; residual update of the mask variable: z^m_{t+1,k} <- z^m_{t,k} + f(z^m_{t+1,k}, z^m_{t,k}) 
                    if t > 0 and z_masks_prev is not None: # z_masks_prev is updated after t > 1.
                        res_m_latent, res_m_scale = torch.split(self.mask_residual_update(torch.cat([z_masks_prev[:, k], z_mask], dim=-1)),
                            2 * [ARCH.Z_MASK_DIM], dim=-1) 
                        z_mask = torch.add(z_masks_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_m_scale) * res_m_latent)
                        # append the mask residual tensor for regularization.
                        # res_m_latents.append(res_m_latent)
                        res_m_latents.append(torch.cat([res_m_latent, res_m_scale], dim=-1))
                        res_m_scales.append(torch.nn.Hardsigmoid()(res_m_scale))
                    z_mask_k_prev = z_mask # for autoregressive update.
                    # condition action for decoding (cVAE scheme) 
                    z_masks.append(z_mask) # append the individual latent for prior sampling.
                    z_mask = self.mask_cond_dec(torch.cat([z_mask, action[:, t]], dim=-1))
                    mask = self.mask_decoder(z_mask)  # In - [B, Zm], Out - [B, 1, H, W]
                    # append the sampling distributions to compute Kl divergence
                    z_mask_posteriors.append(z_mask_post)
                    # append the mask recons
                    masks.append(mask) # masks[0]
                    # self.mask_decoder(self.mask_cond_dec(torch.cat([torch.ones_like(z_mask, device=enc.device), action[:, t]], dim=-1)))
                # concat z_mask along the spatial dim (K) (add new dim by stacking)
                z_masks_cat = torch.stack(z_masks, dim=1) # [B, K, Zm]
                if t > 0: 
                    res_m_latents = torch.stack(res_m_latents, dim=1)  # [B, K, Zm]
                    res_m_scales = torch.stack(res_m_scales, dim=1)  # [B, K, Zm]

                masks = torch.stack(masks, dim=1) # (B, K, 1, H, W) in range (0, 1)
                # SBP to ensure to be summed up to 1. masks.reshape(-1, 1, 64, 64)
                if ARCH.BG_DECOMP == 'sbp':
                    masks = self.SBP(masks) # (B, K, 1, H, W) self.SBP_inv(masks).reshape(-1, 1, 64 ,64)
                # empirically, inverse order SBP also works well.
                elif ARCH.BG_DECOMP == 'inv_sbp':
                    masks = self.SBP_inv(masks)

                B, K, _, H, W = masks.size() # get the shape 

                # reshape (B, K, 1, H, W) -> (B*K, 1, H, W)
                masks = masks.view(B * K, 1, H, W)
                # add 1e-5 to mask to avoid infinity
                # concatenate images along channel axis (B*K, 4, H, W)
                comp_vae_input = torch.cat([(masks + 1e-5).log(),
                    seq[:, t, None].repeat(1, K, 1, 1, 1).view(B * K, 3, H, W)], dim=1)
                # component latents, each [B*K, L]. action [B, A], h_comp_t_post [B*K, Hc]
                z_comp_loc, z_comp_scale = self.comp_cond_encoder(comp_vae_input, 
                    action[:, t], h_comp_t_post)  # conditional comp encoding
                # get posterior distribution of shape [B, K, Zc]
                z_comp_loc, z_comp_scale = z_comp_loc.reshape(B, K, -1), z_comp_scale.reshape(B, K, -1)
                z_comp_post = Normal(z_comp_loc, z_comp_scale)
                z_comps_raw = z_comp_post.rsample() # [B, k, Zc]
                # iterate over spatial component latents
                if t > 0 and z_comps_prev is not None: # z_comps_prev is not None for t > 0.c
                    for k in range(ARCH.K):
                        res_c_latent, res_c_scale = torch.split(self.comp_residual_update(torch.cat([z_comps_prev[:, k], z_comps_raw[:, k]], dim=-1)),
                            2 * [ARCH.Z_COMP_DIM], dim=-1)
                        z_comp = torch.add(z_comps_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_c_scale) * res_c_latent)
                        # append the comp residual tensor for regularization.
                        res_c_latents.append(torch.cat([res_c_latent, res_c_scale], dim=-1))
                        # res_c_latents.append(res_c_latent)
                        res_c_scales.append(torch.nn.Hardsigmoid()(res_c_scale))
                        z_comps.append(z_comp)
                    # stack the residual tensors 
                    res_c_latents = torch.stack(res_c_latents, dim=1)
                    res_c_scales = torch.stack(res_c_scales, dim=1)
                    # concatenate the component latents 
                    z_comps_cat = torch.stack(z_comps, dim=1) # [B, K, Zc]
                else:
                    z_comps_cat = z_comps_raw

                # condition action on decoding component latent; In - [B * K, Zc], [B, A] Out - [B * K, ]
                comps = self.comp_cond_decoder(z_comps_cat.view(B * K, -1), action[:, t])

                # ! 1-1) acquire component posteriors here, for computing KL divergences
                # reshape the loc & scale of distributions to compute KL divergence
                for k in range(ARCH.K):
                    z_comp_post_this = Normal(z_comp_loc[:, k], z_comp_scale[:, k])
                    # TODO (Chmin): create matrix-shaped [B, B, Zc] posterior distributions.
                    z_comp_posteriors_mat.append
                    z_comp_posteriors.append(z_comp_post_this)
                # Decode into component images, [B*K, 3, H, W] masks.reshape(-1, 1, 64, 64)
                comps = comps.view(B, K, 3, H, W)
                masks = masks.view(B, K, 1, H, W)

                # experimental. fix comp learning during fix alpha steps.
                # facilitates the fg learning
                modes = comps * masks # [B, K, 3, H, W]
                bg = modes.sum(dim=1)
                bgs.append(bg)

                # ! 2) Compute prior distributions and KL divergences, for both mask z_m and component z_c
                z_mask_total_kl_k = 0.0  # will be tensor of shape [B, ]
                z_comp_total_kl_k = 0.0  # will be tensor of shape [B, ]
                # prior of the mask is also autoregressive.
                h_mask_k_prior = 2 * self.h_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # spatial recurrent state
                c_mask_k_prior = 2 * self.c_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # spatial recurrent state
                # TODO (chmin)
                z_comp_priors = []
                for k in range(ARCH.K): # iterate over spatial 
                    # ! 2-1) Compute prior distribution over z_masks
                    # prior distribution of agent mask requires the autoreg. info, history, action, and the previous mask (for residual).
                    if k == 0:
                        if t == 0: # there is no previous latent for t=0. Instead we use the one from posterior of t=0.
                            mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], action[:, t],
                                z_masks_cat[:, k]], dim=-1)
                        else:
                            mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], action[:, t],
                                z_masks_prev[:, k]], dim=-1)
                        mask_input = self.prior_net_first(mask_input)
                    else: # it does not require action.
                        if t == 0: # there is no previous latent for t=0. Instead we use the one from  
                            mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], z_masks_cat[:, k]], dim=-1)
                        else:
                            mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], z_masks_prev[:, k]], dim=-1)
                        mask_input = self.prior_net_t(mask_input)
                    
                    mask_inputs_prior.append(mask_input)
                    # TODO (chmin): prior network training requires posterior samples as inputs
                    # predict the mask latent from prior distribution.
                    z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(mask_input)  # In - [B, Hm], Out - [B, Zm]
                    z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)  # [B, Zm]
                    # ! 2-2) Compute component prior, using posterior samples
                    # prior distribution of agent component requires history, corresponding mask latent, 
                    # and the previous component latent (for residual).
                    if t == 0:
                        z_comp_loc_prior, z_comp_scale_prior, comp_prior_out = self.predict_comp_prior(z_masks[k], z_comps_cat[:, k],
                        h_comp_t_prior[:, k])
                    else:
                        z_comp_loc_prior, z_comp_scale_prior, comp_prior_out = self.predict_comp_prior(z_masks[k], z_comps_prev[:, k],
                        h_comp_t_prior[:, k])
                
                    # y0_comp_loc_prior, y0_comp_scale_prior = self.predict_comp_prior(y0_latent_k_list[k], y_comp[:, k],
                    #                                                                  h_comp_t_prior[:, k])               
                    
                    z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
                    comp_inputs_prior.append(comp_prior_out)
                    # z_comp_priors.append(z_comp_loc_prior, z_comp_scale_prior)
                    # Compute KL divergence of both mask and component, for each entity k

                    z_mask_kl = kl_divergence(z_mask_posteriors[k], z_mask_prior).sum(dim=1)  # [B, Zm] sum:-> [B, ]
                    z_comp_kl = kl_divergence(z_comp_posteriors[k], z_comp_prior).sum(dim=1)  # [B, Zc] sum:-> [B, ]
                    
                    # add agent kl divergence
                    
                    # accumulate the kl tensors.
                    z_mask_total_kl_k = z_mask_total_kl_k + z_mask_kl # [B,]
                    z_comp_total_kl_k = z_comp_total_kl_k + z_comp_kl # [B,]
                    # update the hidden state of the autoreg. RNN.

                    # input for the autoregressive update. the z_mask should be sampled from *posterior*
                    rnn_input = torch.cat([z_masks[k], h_mask_t_prior[:, k]], dim=-1)
                    (h_mask_k_prior, c_mask_k_prior) = self.rnn_mask_prior_k(rnn_input, (h_mask_k_prior, c_mask_k_prior))

                # TODO (chmin):
                mask_inputs_prior_cat = torch.stack(mask_inputs_prior, dim=1) # [B, K, Zm]
                comp_inputs_prior_cat = torch.stack(comp_inputs_prior, dim=1) # [B, K, Zc]

                mask_inputs_prior_t.append(mask_inputs_prior_cat)
                comp_inputs_prior_t.append(comp_inputs_prior_cat)

                modes = modes.reshape(B * ARCH.K, 3, H, W)
                mode_enc = self.mode_enc(modes)
                mode_enc = mode_enc.flatten(start_dim=1)  # [B * K, D]
                bg_encs = self.mode_enc_fc(mode_enc).reshape(B, ARCH.K, -1)

                # ! 3-1) temporal encode of z_mask & z_comp into h_post & h_prior -> update temporal latents along T-axis
                h_mask_t_post = h_mask_t_post.reshape(B * ARCH.K, -1)
                c_mask_t_post = c_mask_t_post.reshape(B * ARCH.K, -1)
                h_mask_t_prior = h_mask_t_prior.reshape(B * ARCH.K, -1)
                c_mask_t_prior = c_mask_t_prior.reshape(B * ARCH.K, -1)

                # expand the dimension of background encoding for RNN update of masks of K entities.
                # bg_encs_mask = torch.stack([bg_enc] * ARCH.K, dim=1)  # [B, K, D]
                temp_mask_inpt = torch.cat([z_masks_cat, bg_encs], dim=-1) # # [B, K, D + Zm] 
                temp_mask_inpt = temp_mask_inpt.view(B * ARCH.K, -1)
                # encode the two inputs into one : [B, K, D + Zm] -> [B, K, Zm]
                temp_mask_inpt = self.obs_mask_enc_fn(temp_mask_inpt)

                # temporal update of rnn hidden states of mask prior and posterior
                h_mask_t_post, c_mask_t_post = self.rnn_mask_t_post(temp_mask_inpt, (h_mask_t_post, c_mask_t_post))
                h_mask_t_prior, c_mask_t_prior = self.rnn_mask_t_prior(temp_mask_inpt, (h_mask_t_prior, c_mask_t_prior))

                # initialize the RNN hidden states.
                h_comp_t_post = h_comp_t_post.reshape(B * ARCH.K, -1)
                c_comp_t_post = c_comp_t_post.reshape(B * ARCH.K, -1)
                h_comp_t_prior = h_comp_t_prior.reshape(B * ARCH.K, -1)
                c_comp_t_prior = c_comp_t_prior.reshape(B * ARCH.K, -1)


                temp_comp_inpt = torch.cat([z_comps_cat, bg_encs], dim=-1)
                temp_comp_inpt = temp_comp_inpt.view(B * ARCH.K, -1)
                # encode the two inputs into one: [B, K, D + Zc] -> [B, K, Zc]
                temp_comp_inpt = self.obs_comp_enc_fn(temp_comp_inpt)

                # temporal update of rnn hiddeen states 
                h_comp_t_post, c_comp_t_post = self.rnn_comp_t_post(temp_comp_inpt, (h_comp_t_post, c_comp_t_post))
                h_comp_t_prior, c_comp_t_prior = self.rnn_comp_t_prior(temp_comp_inpt, (h_comp_t_prior, c_comp_t_prior))

                # ! 4) accumulate losses (of all Ks) for the entire horizon T
                z_mask_total_kl_k_t.append(z_mask_total_kl_k)
                z_comp_total_kl_k_t.append(z_comp_total_kl_k)

                # accumulate mask and comp images temporally for visualization
                masks_t.append(masks) # list of [B, K, 1, H, W]
                comps_t.append(comps) # list of [B, K, 3, H, W]

                # accumulate mask and comp latents temporally
                z_masks_t.append(z_masks_cat)  
                z_comps_t.append(z_comps_cat)

                # accumulate residual tensors
                if t > 0:
                    res_m_latents_t.append(res_m_latents)
                    res_m_scales_t.append(res_m_scales)
                    res_c_latents_t.append(res_c_latents)
                    res_c_scales_t.append(res_c_scales)

                # accumulate the hidden state of mask prior RNN. The agent slot of this represents the trajectory of the agent.
                h_masks_t.append(h_mask_t_prior.reshape(B, ARCH.K, -1)) # [B, K, Hm]
                h_comps_t.append(h_comp_t_prior.reshape(B, ARCH.K, -1)) # [B, K, Hc]

                c_masks_t.append(c_mask_t_prior.reshape(B, ARCH.K, -1)) # [B, K, Hm]
                c_comps_t.append(c_comp_t_prior.reshape(B, ARCH.K, -1)) # [B, K, Hc]

                # update the concatdef of latents.
                z_masks_prev = z_masks_cat  
                z_comps_prev = z_comps_cat # update the concat of latents.

                depth_inpt = torch.cat([z_masks_cat[:, agent_idx], 
                    h_mask_t_post.reshape(B, ARCH.K, -1)[:, agent_idx], action[:, t]], dim=-1) # [B, D]
                if t == 0:
                    agent_depth_out = self.get_agent_depth_init(depth_inpt)
                    agent_depth_first_loc, agent_depth_first_scale = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 2, dim=-1)
                    agent_depth_raw_dist_first = Normal(agent_depth_first_loc, agent_depth_first_scale)
                    agent_depth_raw = agent_depth_raw_dist_first.rsample()
                    agent_depth_raw_first_prior = Normal(0, 1)
                    agent_dpeth_raw_first_kl = kl_divergence(agent_depth_raw_dist_first, agent_depth_raw_first_prior) # [B, 1]
                    agent_depth_raw_list_kl.append(agent_dpeth_raw_first_kl)
                else:
                    agent_depth_out = self.get_agent_depth(depth_inpt)
                    agent_depth_loc, agent_depth_scale, agent_depth_gate = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 3, dim=-1)
                    agent_depth_raw_dist = Normal(agent_depth_loc, agent_depth_scale)
                    agent_depth_raw = agent_depth_raw_dist.rsample()
                    agent_depth_gate = torch.sigmoid(agent_depth_gate)
                    agent_depth_raw = agent_depth_raw_prev + ARCH.Z_DEPTH_UPDATE_SCALE * agent_depth_gate * agent_depth_raw # [B, D, 1]

                agent_depth_raw_list.append(agent_depth_raw)
                agent_depth_raw_prev = agent_depth_raw
                # End of temporal loop

        if leverage:
            kl_bg = [ m_kl + c_kl for (m_kl, c_kl) in zip(z_mask_total_kl_k_t[detached_timesteps:], 
                z_comp_total_kl_k_t[detached_timesteps:])]
        else:
            kl_bg = [ m_kl + c_kl for (m_kl, c_kl) in zip(z_mask_total_kl_k_t, z_comp_total_kl_k_t)]
        things = {
            'comps': torch.stack(comps_t, dim=1), # (B, T, K, 3, H, W)
            'masks': torch.stack(masks_t, dim=1), # (B, T, K, 1, H, W)
            'z_masks': torch.stack(z_masks_t, dim=1),  # (B, T, K, Zm)
            'z_comps': torch.stack(z_comps_t, dim=1),  # (B, T, K, Zc)
            'bg': torch.stack(bgs, dim=1),  # (B, T, 3, H, W)
            'kl_bg': torch.stack(kl_bg, dim=1),  # (B, T),
            'agent_depth_raw': torch.stack(agent_depth_raw_list, dim=1), # [B, T, 1]
            'agent_depth_raw_kl': torch.stack(agent_depth_raw_list_kl, dim=1), # [B, T, 1]
            # 'neg_kl_bg_agent': torch.stack(bg_agent_neg_kls_t, dim=1),  # (B, T)
            # TODO (chmin): check if which of posterior or prior should be given
            'h_masks_post': torch.stack(h_masks_t, dim=1), # [B, T, K, Hm]
            'c_masks_post': torch.stack(c_masks_t, dim=1), # [B, T, K, Hm]
            'h_comps_post': torch.stack(h_comps_t, dim=1), # [B, T, K, Hc]
            'c_comps_post': torch.stack(c_comps_t, dim=1), # [B, T, K, Hc]
            'enhanced_act': action,
            'mask_residuals': torch.stack(res_m_latents_t, dim=1),
            'mask_scales': torch.stack(res_m_scales_t, dim=1),
            'comp_residuals': torch.stack(res_c_latents_t, dim=1),
            'comp_scales': torch.stack(res_c_scales_t, dim=1),
            'mask_inputs_prior': torch.stack(mask_inputs_prior_t, dim=1), # [B, T, K, Zm]
            'comp_inputs_prior': torch.stack(comp_inputs_prior_t, dim=1) # [B, T, K, Zc]
        }
        return things

    def generate(self, seq, action, cond_steps, agent_slot):
        """
            Generate new frames given a set of input frames
            Args:
                seq: (B, T, 3, H, W)
                action: (B, T, A); where P is the end-effector pose dimension
            Returns:
                things:
                    bg: (B, T, 3, H, W)
                    kl: (B, T)
        """
        B, T, C, H, W = seq.size()
        A = action.size(-1)
        if A != ARCH.ACTION_ENHANCE:
            action = self.action_enhance(action)
        seq = seq.reshape(B * T, 3, H, W)
        # encode the whole sequence.
        enc = self.enc(seq)
        seq = seq.view(B, T, 3, H, W)
        enc = enc.flatten(start_dim=1)  # (B*T, D)

        enc = self.enc_fc(enc)  # [B, T, 128]
        enc = enc.view(B, T, ARCH.IMG_ENC_DIM)  # [B, T, 128]
        
        # TODO (chmin): first step inference should be 
        obs_cat_enc = self.cond_obs_cat_encoder(enc[:, :5].view(B, 5 * ARCH.IMG_ENC_DIM))

        # mask and component latents over the K slots.
        # spatial initial RNN paramters
        h_mask_t_post = 2 * self.h_mask_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # random init nn_params: shape - [B, Hm]
        c_mask_t_post = 2 * self.c_mask_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM)  - 1 # random init nn_params: shape - [B, Hm]
        h_comp_t_post = 2 * self.h_comp_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM)  - 1  # random init nn_params: shape - [B, Hc]
        c_comp_t_post = 2 * self.c_comp_RNN_t_post.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM)  - 1  # random init nn_params: shape - [B, Hc]
        h_mask_t_prior = 2 * self.h_mask_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM)  - 1  # random init nn_params: shape - [B, Hm]
        c_mask_t_prior = 2 * self.c_mask_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_MASK_HIDDEN_DIM)  - 1 # random init nn_params: shape - [B, Hm]
        h_comp_t_prior = 2 * self.h_comp_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM)  - 1  # random init nn_params: shape - [B, Hc]
        c_comp_t_prior = 2 * self.c_comp_RNN_t_prior.expand(B * ARCH.K,
                                            ARCH.RNN_CTX_COMP_HIDDEN_DIM)  - 1  # random init nn_params: shape - [B, Hc]
        # initialize lists to append temporal info.
        z_masks_t = [] # list to append the mask latent over T.
        z_comps_t = [] # list to append the component latent over T.
        masks_t = [] # list to apped mask recon. Used for visualization.
        comps_t = [] # list to append comp recon. Used for visualization.
        bgs = []  # list of background reconstruction Tensors
        h_masks_t = [] # list of rnn hidden state of mask latents. Used to form agent node on GNN.
        agent_depth_raw_list = []

        # iterate over the entire horizon
        # upto 'cond_steps' -> posterior
        # after 'cond_steps' -> prior

        # temporally update concat of latents 
        z_masks_prev = None # placeholder for the temporal concat of mask latent.
        z_comps_prev = None # placeholder for the temporal concat of comp latent.

        for t in range(T):
            h_mask_t_post = h_mask_t_post.view(B, ARCH.K, -1)
            c_mask_t_post = c_mask_t_post.view(B, ARCH.K, -1)
            h_comp_t_post = h_comp_t_post.view(B, ARCH.K, -1)
            c_comp_t_post = c_comp_t_post.view(B, ARCH.K, -1)

            h_mask_t_prior = h_mask_t_prior.view(B, ARCH.K, -1)
            c_mask_t_prior = c_mask_t_prior.view(B, ARCH.K, -1)
            h_comp_t_prior = h_comp_t_prior.view(B, ARCH.K, -1)
            c_comp_t_prior = c_comp_t_prior.view(B, ARCH.K, -1)

            masks = [] # list to append mask recons
            z_masks = [] # list to append mask latents
            z_comps = [] # list to append comp latents

            if t == 0:
                post_obs = obs_cat_enc 
            else:
                post_obs = enc[:, t]

            if t < cond_steps: # make posterior inference
                #! initialize spatial latents of dim K for every time T
                z_mask_k_prev = 2 * self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) -1 # autoregressive latents
                h_mask_k_post = 2 * self.h_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # hidden states of autoreg. RNN
                c_mask_k_post = 2 * self.c_mask_k_post.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1# hidden states of autoreg. RNN
                #! 1) Compute posteriors
                # iterate over K entities, infer autoregressive mask latents
                for k in range(ARCH.K):
                    if k == 0: # autoregressively embed action.
                        post_input = torch.cat([h_mask_t_post[:, k], post_obs, action[:, t]], dim=-1)
                        post_input = self.post_net_first(post_input)   
                    else: # other entities
                        post_input = torch.cat([h_mask_t_post[:, k], post_obs], dim=-1)
                        post_input = self.post_net_t(post_input)  # k = {1:K}
                    # concat posterior info with spatial latent. -> For autoreg. update
                    rnn_input = torch.cat([post_input, z_mask_k_prev], dim=-1)
                    # autoregressive update of mask hidden state h^m_{t,k-1} -> h^m_{t,k} 
                    (h_mask_k_post, c_mask_k_post) = self.rnn_mask_post_k(rnn_input, (
                        h_mask_k_post, c_mask_k_post))
                    # predict mask latent variable z^m_{t,k} from h^m_{t,k}.
                    z_mask_loc, z_mask_scale = self.predict_mask(h_mask_k_post) # [B, Zm]
                    z_mask_post = Normal(z_mask_loc, z_mask_scale)  # [B, Zm]
                    z_mask = z_mask_post.rsample()  # z^m_t
                    
                    # for t >= 1; residual update of the mask variable: z^m_{t+1,k} 
                    if t > 0 and z_masks_prev is not None: # z_masks_prev is updated after t > 1.
                        res_m_latent, res_m_scale = torch.split(self.mask_residual_update(torch.cat([z_masks_prev[:, k], z_mask], dim=-1)),
                            2 * [ARCH.Z_MASK_DIM], dim=-1) 
                        z_mask = torch.add(z_masks_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_m_scale) * res_m_latent)
                    z_mask_k_prev = z_mask  # for autoregressive update.
                    # conditioning action for decoding (cVAE scheme)
                    z_masks.append(z_mask) # append the individual latent for prior sampling
                    z_mask = self.mask_cond_dec(torch.cat([z_mask, action[:, t]], dim=-1))
                    mask = self.mask_decoder(z_mask)  # In - [B, Zm], Out - [B, 1, H, W]
                    # append the mask recons
                    masks.append(mask)

                # concat z_mask along the spatial dim (K)
                z_masks_cat = torch.stack(z_masks, dim=1)  # (B, K, Zm)
                masks = torch.stack(masks, dim=1)  # (B, K, 1, H, W) in range (0, 1)

                # SBP to ensure to be summed up to 1.
                if ARCH.BG_DECOMP == 'sbp':
                    masks = self.SBP(masks)
                elif ARCH.BG_DECOMP == 'inv_sbp':
                    masks = self.SBP_inv(masks)  

                B, K, _, H, W = masks.size()

                # Reshape (B, K, 1, H, W) -> (B*K, 1, H, W)
                masks = masks.view(B * K, 1, H, W)
                # add 1e-5 to mask to avoid infinity
                # concatenate images along channel axis (B*K, 4, H, W); torch.repeat == tf.tile
                comp_vae_input = torch.cat([(masks + 1e-5).log(),
                    seq[:, t, None].repeat(1, K, 1, 1, 1).view(B * K, 3, H, W)], dim=1)
                # component latents, each [B*K, L]
                z_comp_loc, z_comp_scale = self.comp_cond_encoder(comp_vae_input, 
                    action[:, t], h_comp_t_post)  # conditional comp encoding
                # get posterior distribution of shape [B, K, Zc]
                z_comp_loc, z_comp_scale = z_comp_loc.reshape(B, K, -1), z_comp_scale.reshape(B, K, -1)
                z_comp_post = Normal(z_comp_loc, z_comp_scale)
                z_comps_raw = z_comp_post.rsample()  # [B*K, Zc]
                # iterate over spaitial component latents
                if t > 0 and z_comps_prev is not None: # z_comps_prev is not None for t > 0.
                    for k in range(ARCH.K):
                        res_c_latent, res_c_scale = torch.split(self.comp_residual_update(torch.cat([z_comps_prev[:, k], z_comps_raw[:, k]], dim=-1)),
                            2 * [ARCH.Z_COMP_DIM], dim=-1)
                        z_comp = torch.add(z_comps_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_c_scale) * res_c_latent)
                        z_comps.append(z_comp)
                    # concatenate the component latents
                    z_comps_cat = torch.stack(z_comps, dim=1)
                else:
                    z_comps_cat = z_comps_raw

                # condition action on decoding component latent; In - [B * K, Zc], [B, A] Out - [B * K, ]
                comps = self.comp_cond_decoder(z_comps_cat.view(B * K, -1), action[:, t])
        
                # make sure we have the right shape for both masks and components
                comps = comps.view(B, K, 3, H, W)
                masks = masks.view(B, K, 1, H, W)

                # generate background recons.
                modes = comps * masks # [B, K, 3, H, W]
                bg = modes.sum(dim=1)

            else:  #------------------------- after cond_steps, priors generation -----------
                # ! 2) Compute priors
                # prior of the mask is also autoregressive.
                h_mask_k_prior = 2 * self.h_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # spatial recurrent state
                c_mask_k_prior = 2 * self.c_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) -1 # spatial recurrent state
                
                # reset the spatially initial latent 
                z_mask_k_prev = 2 * self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) - 1 # first slot along spatial, and updated temporally
                
                for k in range(ARCH.K): # iterate over spatial
                    # ! 2-1) Compute prior distribution over z_masks
                    # prior distribution of agent mask requires the autoreg. info, history, action, and the previous mask (for residual).
                    if k == 0: # since prior generation occurs @ t>0, the input form for sampling differs from training.
                        mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], action[:, t], 
                            z_masks_prev[:, k]], dim=-1)
                        mask_input = self.prior_net_first(mask_input)
                    else:
                        # it does not require action. # TODO (chmin) : verify this
                        mask_input = torch.cat([h_mask_k_prior, h_mask_t_prior[:, k], z_masks_t[-1][:, k]], dim=-1)
                        mask_input = self.prior_net_t(mask_input)


                    # predict the mask latent from prior distribution.
                    z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(mask_input) # In - [B, Hm], Out - [B, Zm]
                    z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)  # [B, Zm]
                    z_mask = z_mask_prior.rsample() # prior sampled variable.
                    # for t >= 1; residual update of the mask variable: z^m_{t+1,k} <- z^m_{t,k} + f(z^m_{t+1,k}, z^m_{t,k}) 
                    if t > 0 and z_masks_prev is not None: # z_masks_prev is updated after t > 1.
                        res_m_latent, res_m_scale = torch.split(self.mask_residual_update(torch.cat([z_masks_prev[:, k], z_mask], dim=-1)),
                            2 * [ARCH.Z_MASK_DIM], dim=-1) 
                        z_mask = torch.add(z_masks_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_m_scale) * res_m_latent)
                    z_masks.append(z_mask) # append the individual latent for prior sampling
                    z_mask_k_prev = z_mask # for autoregressive update.
                    # condition action for decoding (cVAE scheme)
                    z_mask = self.mask_cond_dec(torch.cat([z_mask, action[:, t]], dim=-1))
                    # decode the mask latent
                    mask = self.mask_decoder(z_mask)  # In - [B, Zm], Out - [B, 1, H, W]
                    # append mask latents
                    masks.append(mask)

                    # input for the autoregressive update of amask. the z_mask should be sampled from *prior*
                    rnn_input = torch.cat([z_mask_k_prev, h_mask_t_prior[:, k]], dim=-1)
                    # autoregressive update of mask hidden state h^m_{t,k-1} -> h^m_{t,k} 
                    (h_mask_k_prior, c_mask_k_prior) = self.rnn_mask_prior_k(rnn_input, (h_mask_k_prior, c_mask_k_prior))

                    # prior distribution of agent component requires history, corresponding mask latent, 
                    # and the previous component latent (for residual).
                    z_comp_loc_prior, z_comp_scale_prior, _ = self.predict_comp_prior(z_masks[k], z_comps_prev[:, k],
                        h_comp_t_prior[:, k])
                    # generate the prior distribution of component
                    z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
                    z_comp = z_comp_prior.rsample()

                    if t > 0 and z_comps_prev is not None: # z_comps_prev is not None for t > 0.
                        res_c_latent, res_c_scale = torch.split(self.comp_residual_update(torch.cat([z_comps_prev[:, k], z_comp], dim=-1)),
                            2 * [ARCH.Z_COMP_DIM], dim=-1)
                        z_comp = torch.add(z_comps_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_c_scale) * res_c_latent)
                        z_comps.append(z_comp)

                # temporal concat of the mask and comp latents
                z_masks_cat = torch.stack(z_masks, dim=1)  # [B, K, Zm]
                z_comps_cat = torch.stack(z_comps, dim=1)  # [B, K, Zc]
                # (B, K, 1, H, W) in range (0, 1)
                masks = torch.stack(masks, dim=1) # masks[0]
                B, K, _, H, W = masks.size()


                # SBP to ensure to be summed up to 1.
                if ARCH.BG_DECOMP == 'sbp':
                    masks = self.SBP(masks)
                elif ARCH.BG_DECOMP == 'inv_sbp':
                    masks = self.SBP_inv(masks)
                # decode the component
                comps = self.comp_cond_decoder(z_comps_cat.reshape(B * K, -1), action[:, t])
                
                # make sure we have the right shape for both masks and components
                comps = comps.view(B, K, 3, H, W)
                masks = masks.view(B, K, 1, H, W)
                # masks.reshape(-1, 1, 64, 64)
                # generate background recons.
                modes = comps * masks # [B, K, 3, H, W]
                bg = modes.sum(dim=1)

            # append tensors to visualize
            z_masks_t.append(z_masks_cat)
            z_comps_t.append(z_comps_cat)
            masks_t.append(masks) # masks_t[25][0]
            comps_t.append(comps)
            bgs.append(bg) # bgs[100]

            # ! 3) temporal update of the hidden state of RNNs of masks and components;
            # mode_enc = self.mode_enc(modes)
            # mode_enc = mode_enc.flatten(start_dim=1)  # [B * K, D]
            # bg_encs = self.mode_enc_fc(mode_enc).reshape(B, ARCH.K, -1)
            modes = modes.reshape(B * ARCH.K, 3, H, W)
            mode_enc = self.mode_enc(modes)
            mode_enc = mode_enc.flatten(start_dim=1)  # [B * K, D]
            bg_encs = self.mode_enc_fc(mode_enc).reshape(B, ARCH.K, -1)


            # ! 3-1) temporal encode of z_mask & z_comp into h_post & h_prior -> update temporal latents along T-axis
            # the RNN update should be 
            h_mask_t_post = h_mask_t_post.reshape(B * ARCH.K, -1)
            c_mask_t_post = c_mask_t_post.reshape(B * ARCH.K, -1)
            h_mask_t_prior = h_mask_t_prior.reshape(B * ARCH.K, -1)
            c_mask_t_prior = c_mask_t_prior.reshape(B * ARCH.K, -1)

            # expand the dimension of background encoding for RNN update of masks of K entities.
            # bg_encs_mask = torch.stack([bg_enc] * ARCH.K, dim=1) # [B, K, D]
            temp_mask_inpt = torch.cat([z_masks_cat, bg_encs], dim=-1) # [B, K, D + Zm]
            temp_mask_inpt = temp_mask_inpt.view(B * ARCH.K, -1)
            # encode the two inputs into one: [B, K, D + Zm] -> [B, K, Zm]
            temp_mask_inpt = self.obs_mask_enc_fn(temp_mask_inpt)

            # temporal update of rnn hidden states of mask prior and posterior
            h_mask_t_post, c_mask_t_post = self.rnn_mask_t_post(temp_mask_inpt, (h_mask_t_post, c_mask_t_post))
            h_mask_t_prior, c_mask_t_prior = self.rnn_mask_t_prior(temp_mask_inpt, (h_mask_t_prior, c_mask_t_prior))

            # initialize the RNN hidden states.
            h_comp_t_post = h_comp_t_post.reshape(B * ARCH.K, -1)
            c_comp_t_post = c_comp_t_post.reshape(B * ARCH.K, -1)
            h_comp_t_prior = h_comp_t_prior.reshape(B * ARCH.K, -1)
            c_comp_t_prior = c_comp_t_prior.reshape(B * ARCH.K, -1)

            # expand the dimension of background encoding for RNN update of masks of K entities.
            # bg_encs_comp = torch.stack([bg_enc] * ARCH.K, dim=1)

            temp_comp_inpt = torch.cat([z_comps_cat, bg_encs], dim=-1)
            temp_comp_inpt = temp_comp_inpt.view(B * ARCH.K, -1)
            # encode the two inputs into one: [B, K, D + Zc] -> [B, K, Zc]
            temp_comp_inpt = self.obs_comp_enc_fn(temp_comp_inpt)

            # temporal update of rnn hiddeen states 
            h_comp_t_post, c_comp_t_post = self.rnn_comp_t_post(temp_comp_inpt, (h_comp_t_post, c_comp_t_post))
            h_comp_t_prior, c_comp_t_prior = self.rnn_comp_t_prior(temp_comp_inpt, (h_comp_t_prior, c_comp_t_prior))

            # accumulate the hidden state of mask prior RNN. The agent slot of this represents the trajectory of the agent.
            h_masks_t.append(h_mask_t_prior.reshape(B, ARCH.K, -1)) # [B, K, Hm]

            # update the concat of latents.
            z_masks_prev = z_masks_cat  
            z_comps_prev = z_comps_cat # update the concat of latents.

            depth_inpt = torch.cat([z_masks_cat[:, agent_slot], 
                h_mask_t_post.reshape(B, ARCH.K, -1)[:, agent_slot], action[:, t]], dim=-1) # [B, D]
            if t == 0:
                agent_depth_out = self.get_agent_depth_init(depth_inpt)
                agent_depth_first_loc, agent_depth_first_scale = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 2, dim=-1)
                agent_depth_raw_dist_first = Normal(agent_depth_first_loc, agent_depth_first_scale)
                agent_depth_raw = agent_depth_raw_dist_first.rsample()
            else:
                agent_depth_out = self.get_agent_depth(depth_inpt)
                agent_depth_loc, agent_depth_scale, agent_depth_gate = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 3, dim=-1)
                agent_depth_raw_dist = Normal(agent_depth_loc, agent_depth_scale)
                agent_depth_raw = agent_depth_raw_dist.rsample()
                agent_depth_gate = torch.sigmoid(agent_depth_gate)
                agent_depth_raw = agent_depth_raw_prev + ARCH.Z_DEPTH_UPDATE_SCALE * agent_depth_gate * agent_depth_raw # [B, D, 1]

            agent_depth_raw_list.append(agent_depth_raw)
            agent_depth_raw_prev = agent_depth_raw

        # # ---end of temporal iteration.---
        things = {
            'comps': torch.stack(comps_t, dim=1), # (B, T, K, 3, H, W)
            'masks': torch.stack(masks_t, dim=1), # (B, T, K, 1, H, W)
            'z_comps': torch.stack(z_comps_t, dim=1),  # (B, T, K, Zc)
            'z_masks': torch.stack(z_masks_t, dim=1),  # (B, T, K, Zm)
            'agent_depth_raw': torch.stack(agent_depth_raw_list, dim=1), # [B, T, 1]
            'bg': torch.stack(bgs, dim=1),  # (B, T, 3, H, W)
            'h_masks': torch.stack(h_masks_t, dim=1),
            'enhanced_act': action
        }
        return things

    
    def imagine(self, history, action, z_prevs, episodic_step=0, agent_slot=0):
        """
            Do one-step prior generation with the upddate of dynamics.
            NOTE that the batch dimension is B*T here. The lower bound of 'episodic_step' is 1.
        """
        # [B * T, K, H]
        h_masks_t_prior, c_masks_t_prior, h_comps_t_prior, c_comps_t_prior = history
        
        # trim out the data of last step.
        # [B * T, K, D]
        z_masks_prev, z_comps_prev = z_prevs

        B, K, _ = h_masks_t_prior.size() # B is actually
        A = action.size(-1) 

        if A == ARCH.ACTION_DIM: # if action is not enhanced
            action = self.action_enhance(action) # [B * T, A]

        # reset the spatially initial latent
        z_mask_k_prev = 2 * self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) - 1
        h_mask_k_prior = 2 * self.h_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1 # hidden states of autoreg. RNN
        c_mask_k_prior = 2 * self.c_mask_k_prior.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) - 1# hidden states of autoreg. RNN

        masks = []
        z_masks = []
        z_comps = []

        for k in range(ARCH.K):
            if k == 0:
                mask_input = torch.cat([h_mask_k_prior, h_masks_t_prior[:, k], action,
                    z_masks_prev[:, k]], dim=-1)
                mask_input = self.prior_net_first(mask_input) # note that the shape is [B * T, D]
            else:
                mask_input = torch.cat([h_mask_k_prior, h_masks_t_prior[:, k],
                    z_masks_prev[:, k]], dim=-1)
                mask_input = self.prior_net_t(mask_input)

            # predict the mask latent from prior distribution.
            z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(mask_input)
            z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
            z_mask = z_mask_prior.rsample()
            # for t >= 1; residual update of the mask 
            if z_masks_prev is not None:
                res_m_latent, res_m_scale = torch.split(self.mask_residual_update(torch.cat([z_masks_prev[:, k], z_mask], dim=-1)),
                    2 * [ARCH.Z_MASK_DIM], dim=-1) 
                z_mask = torch.add(z_masks_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_m_scale) * res_m_latent)
            z_masks.append(z_mask)
            z_mask_k_prev = z_mask # for autoregressive update
            # condition action for decoding (cVAE scheme)
            z_mask = self.mask_cond_dec(torch.cat([z_mask, action], dim=-1))
            with torch.no_grad():
                mask = self.mask_decoder(z_mask) # In - [B * T, Zm], Out - [B * T, 1, H, W]
            # if k != agent_slot:
            #     mask = mask.detach()
            masks.append(mask)

            # input for the autoregressive update of a mask.
            rnn_input = torch.cat([z_mask_k_prev, h_masks_t_prior[:, k]], dim=-1)
            # autoregressive update of mask hidden state h^m_{t, k-1} -> h^m_{t, k} 
            (h_mask_k_prior, c_mask_k_prior) = self.rnn_mask_prior_k(rnn_input, (h_mask_k_prior, c_mask_k_prior))

            z_comp_loc_prior, z_comp_scale_prior, _ = self.predict_comp_prior(z_masks[k], z_comps_prev[:, k],
                    h_comps_t_prior[:, k])
            # generate the prior distribution of component
            z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
            z_comp = z_comp_prior.rsample()

            if z_comps_prev is not None:
                res_c_latent, res_c_scale = torch.split(self.comp_residual_update(torch.cat([z_comps_prev[:, k], z_comp], dim=-1)),
                    2 * [ARCH.Z_COMP_DIM], dim=-1)
                z_comp = torch.add(z_comps_prev[:, k], 2.0 * torch.nn.Hardsigmoid()(res_c_scale) * res_c_latent)
                z_comps.append(z_comp)

        # temporal concat of the mask and comp latents
        z_masks_cat = torch.stack(z_masks, dim=1) # [B*T, K, Zm]
        z_comps_cat = torch.stack(z_comps, dim=1) # [B*T, K, Zc]
        # [BT, K, 1, H, W] in range (0, 1)
        # with torch.no_grad():
        masks = torch.stack(masks, dim=1)
        B, K, _, H, W = masks.size() 

        # SBP to ensure to be summed up to 1.
        if ARCH.BG_DECOMP == 'sbp':
            masks = self.SBP(masks)
        elif ARCH.BG_DECOMP == 'inv_sbp':
            masks = self.SBP_inv(masks)
        # decode the component; [B*T*K, 3, 64, 64]
        with torch.no_grad():
            comps = self.comp_cond_decoder(z_comps_cat.reshape(B * K, -1), action)

            # make sure we have the right shape for both masks and components
            comps = comps.view(B, K, 3, H, W)
            masks = masks.view(B, K, 1, H, W)

            # generate background recons; [B*T, 3, 64, 64]
            mode = comps * masks # .sum(dim=1)
            bg = mode.sum(1)
            
            # with torch.no_grad():
            mode = mode.reshape(B * ARCH.K, 3, H, W)
            mode_enc = self.mode_enc(mode)
            mode_enc = mode_enc.flatten(start_dim=1)  # [B * K, D]
            bg_enc = self.mode_enc_fc(mode_enc).reshape(B, ARCH.K, -1)
            bg_enc = bg_enc.detach()

        # update the temporal history state.
        # ! 3-1) temporal encode of z_mask & z_comp into h_post & h_prior -> update temporal latents along T-axis
        # the RNN update should be 
        h_masks_t_prior = h_masks_t_prior.reshape(B * ARCH.K, -1) # [B * T * K, -1]
        c_masks_t_prior = c_masks_t_prior.reshape(B * ARCH.K, -1) # [B * T * K, -1]

        # expand the dimension 
        temp_mask_inpt = torch.cat([z_masks_cat, bg_enc], dim=-1) # [B * T, K, D + Zm]
        temp_mask_inpt = temp_mask_inpt.view(B * ARCH.K, -1)
        # encode the two inputs into one: [B, K, D + Zm] -> [B, K, Zm]
        temp_mask_inpt = self.obs_mask_enc_fn(temp_mask_inpt)
        #   tkfkdgo
        # temporal update of rnn hidden states of mask prior and posterior
        h_masks_t_prior, c_masks_t_prior = self.rnn_mask_t_prior(temp_mask_inpt, (h_masks_t_prior, c_masks_t_prior))

        # initialize the RNN hidden states.
        h_comps_t_prior = h_comps_t_prior.reshape(B * ARCH.K, -1)
        c_comps_t_prior = c_comps_t_prior.reshape(B * ARCH.K, -1)

        # expand the dimension of background encoding for RNN update of masks of K entities.
        temp_comp_inpt = torch.cat([z_comps_cat, bg_enc], dim=-1)
        temp_comp_inpt = temp_comp_inpt.view(B * ARCH.K, -1)
        # encode the two inputs into one: [B, K, D + Zc] -> [B, K, Zc]
        temp_comp_inpt = self.obs_comp_enc_fn(temp_comp_inpt)

        # temporal update of rnn hiddeen states 
        h_comps_t_prior, c_comps_t_prior = self.rnn_comp_t_prior(temp_comp_inpt, (h_comps_t_prior, c_comps_t_prior))
        h_masks_t_prior_reshaped = h_comps_t_prior.reshape(B, ARCH.K, -1)
        depth_inpt = torch.cat([z_masks_cat[:, agent_slot], h_masks_t_prior_reshaped[:, agent_slot], action], dim=-1) # [1, D]
        agent_depth_out = self.get_agent_depth(depth_inpt)
        agent_depth_loc, agent_depth_scale, agent_depth_gate = torch.split(agent_depth_out, [ARCH.Z_DEPTH_DIM] * 3, dim=-1)
        agent_depth_raw_dist = Normal(agent_depth_loc, agent_depth_scale)
        agent_depth_raw = agent_depth_raw_dist.rsample()
        agent_depth_gate = torch.sigmoid(agent_depth_gate)
        agent_depth_raw = self.agent_depth_raw_prev + ARCH.Z_DEPTH_UPDATE_SCALE * agent_depth_gate * agent_depth_raw # [B, D, 1]

        setattr(self, 'agent_depth_raw_prev', agent_depth_raw)

        # TODO (chmin): figure out what to return.
        things = {
            'z_comps': z_comps_cat, # [B*T, K, 32]
            'z_masks': z_masks_cat, # [B*T, K, 16],
            'agent_depth_raw': agent_depth_raw, # [B*T, 1],
            'h_mask_prior': h_masks_t_prior.reshape(B, K, -1), # [B*T, K, 32]
            'c_mask_prior': c_masks_t_prior.reshape(B, K, -1), # [B*T, K, 32]
            'h_comp_prior': h_comps_t_prior.reshape(B, K, -1), # [B*T, K, 64]
            'c_comp_prior': c_comps_t_prior.reshape(B, K, -1), # [B*T, K, 64]
            'masks': masks, 
            'bg' : bg, # [B*T, 3, 64, 64]
            'enhanced_act': action # [B*T, A]
        }
        return things


    # TODO (chmin): dynamics may not be necessary since we update the RNN states 
    # simultaneously with the latent inference.
    # def dynamics(self):
    #     """
    #         Do one step transition dynamics.
    #     """

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, K, _, H, W = masks.size()

        # (B, 1, H, W)
        remained = torch.ones_like(masks[:, 0])
        # remained = torch.ones_like(masks[:, 0]) - fg_mask
        new_masks = []
        for k in range(K):
            if k < K - 1:  # k = 0, 1, ... K-1
                mask = masks[:, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)

        new_masks = torch.stack(new_masks, dim=1)  # (B, K, 1, H, W)

        return new_masks

    @staticmethod
    def SBP_inv(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, K, _, H, W = masks.size()

        # (B, 1, H, W)
        remained = torch.ones_like(masks[:, 0])
        # remained = torch.ones_like(masks[:, 0]) - fg_mask
        new_masks = []
        for k in range(K):
            if k < K - 1:  # k = 0, 1, ... K-1
                mask = masks[:, ARCH.K - 1 - k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)
        new_masks.reverse()
        new_masks = torch.stack(new_masks, dim=1)  # (B, K, 1, H, W)
        return new_masks


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ObsEncoderMix(nn.Module):
    """Background image encoder"""

    def __init__(self):
        embed_size = ARCH.IMG_SHAPE[0] // 16  # ARCH.IMG_SIZE // 4
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 16
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 8
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.BatchNorm2d(64),
            nn.ELU(),
            Flatten(),
            nn.Linear(64 * embed_size ** 2, ARCH.IMG_ENC_DIM),
            nn.ELU(),
        )

    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, D)
        """
        return self.enc(x)


class ResNetEncoder(nn.Module):
    """
    Used in discovery. Input is image plus image - bg
    """

    def __init__(self):
        super(ResNetEncoder, self).__init__()

        assert ARCH.G in [8, 4]
        assert ARCH.IMG_SIZE in [64, 128]
        embed_size = ARCH.IMG_SHAPE[0] // 16  # ARCH.IMG_SIZE // 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet18()
        self.enc = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            Flatten(),
            nn.Linear(64 * embed_size ** 2, ARCH.IMG_ENC_DIM),
            nn.ELU(),
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
        return x


class ActionCompCondEnc(nn.Module):
    """
    Condition action vector (or robot state) to comp latent
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Sequential(
                nn.Linear(2 * ARCH.Z_COMP_DIM + ARCH.ACTION_ENHANCE, ARCH.COMP_COND_HIDDEN_DIM),
                nn.CELU(),
                nn.Linear(ARCH.COMP_COND_HIDDEN_DIM, 2 * ARCH.Z_COMP_DIM))

    def forward(self, h):
        """
        In: [Hm, A] - Out: [Hm]
        :param h: hidden state from rnn_mask - (B, Hm = ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        """
        x = self.fc(h)  # In: [B, Hm], Out: [B, 2 * Zm]
        return x


class ActionCompCondDec(nn.Module):
    """
    Condition action vector (or robot state) to comp latent
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Sequential(
                nn.Linear(ARCH.Z_COMP_DIM + ARCH.ACTION_ENHANCE, ARCH.COMP_COND_HIDDEN_DIM),
                nn.CELU(),
                nn.Linear(ARCH.COMP_COND_HIDDEN_DIM, ARCH.Z_COMP_DIM))

    def forward(self, h):
        """
        In: [Hm, A] - Out: [Hm]
        :param h: hidden state from rnn_mask - (B, Hm = ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        """
        x = self.fc(h)  # In: [B, Hm], Out: [B, 2 * Zm]
        return x


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self):
        nn.Module.__init__(self)
        # if residual, also predcit the gate param in addition to loc and scale
        self.fc = nn.Sequential(
                nn.Linear(ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
                nn.CELU(),
                nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, 2 * ARCH.Z_MASK_DIM))

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference
        :param h: hidden state from rnn_mask - (B, Hm = ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        """
        x = self.fc(h)  # In: [B, Hm], Out: [B, 2 * Zm]
        z_mask_loc, z_mask_scale = torch.split(x, [ARCH.Z_MASK_DIM] * 2, dim=-1)
        z_mask_scale = F.softplus(z_mask_scale) + 1e-4  # [B, Zm]
        return z_mask_loc, z_mask_scale


class PredictMaskPrior(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self):
        nn.Module.__init__(self)
        # self.fc = nn.Linear(2 * ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.Z_MASK_DIM, ARCH.Z_MASK_DIM * 2)
        self.fc = nn.Sequential(
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, ARCH.Z_MASK_DIM * 2),
        )

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference
        :param h: hidden state from rnn_mask - (B, Hm = ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        """
        x = self.fc(h)  # In: [B, Hm], Out: [B, 2 * Zm]
        z_mask_loc = x[:, :ARCH.Z_MASK_DIM]  # [B, Zm]
        z_mask_scale = F.softplus(x[:, ARCH.Z_MASK_DIM:]) + 1e-4  # [B, Zm]
        return z_mask_loc, z_mask_scale


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=ARCH.Z_MASK_DIM, out_channels=256, kernel_size=1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),  # (256 * 4 * 4) -> (256, 4, 4)
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, z_mask):
        """
        Decode z_mask into mask
        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        B = z_mask.size(0)
        z_mask = z_mask.view(B, -1, 1, 1)
        mask = torch.nn.Hardsigmoid()(self.dec(z_mask))  # [B, 1, H, W]
        return mask


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """

    def __init__(self):
        nn.Module.__init__(self)

        embed_size = ARCH.IMG_SHAPE[0] // 8  # divide by 2^N since we do N Convs
        self.enc = nn.Sequential(  # (D_in + 2*padding - kernel) / stride + 1
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32], 32 = floor[(64 + 2*1 -3)/2] + 1
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ELU(),
            # nn.Conv2d(64, 64, 3, 2, 1), # [B, 64, 4, 4]
            # nn.BatchNorm2d(64),
            # nn.ELU(),
            Flatten(),  # [B, 64 * 4 * 4]
            # 16x downsampled: (64, 4, 4)
            nn.Linear(64 * embed_size ** 2, ARCH.Z_COMP_DIM * 2),
        )

    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated
        :param x: (B*K, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.enc(x)  # [B, 2 * Zc]
        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4

        return z_comp_loc, z_comp_scale


class CompCondEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """
    # TODO (chmin): the groupnorm architecture should be determined empirically.
    def __init__(self):
        nn.Module.__init__(self)
        embed_size = ARCH.IMG_SHAPE[0] // 8  # divide by 2^N since we do N Convs
        # embed_size = ARCH.IMG_SHAPE[0] // 16 # divide by 2^N since we do N Convs
        # self.enc = nn.Sequential(  # (D_in + 2*padding - kernel) / stride + 1
        #     nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32], 32 = floor[(64 + 2*1 -3)/2] + 1
        #     nn.CELU(),
        #     nn.GroupNorm(4, 32),

        #     nn.Conv2d(32, 32, 3, 2, 1),  # [B, 32, 16, 16]
        #     nn.CELU(),
        #     nn.GroupNorm(4, 32),

        #     nn.Conv2d(32, 64, 3, 2, 1),  # [B, 64, 8, 8]
        #     nn.CELU(),
        #     nn.GroupNorm(8, 64),

        #     Flatten(),  # [B, 64 * 4 * 4]
        #     nn.Linear(64 * embed_size ** 2, ARCH.Z_COMP_DIM * 2),
        # )

        self.enc = nn.Sequential(  # (D_in + 2*padding - kernel) / stride + 1
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32], 32 = floor[(64 + 2*1 -3)/2] + 1
            nn.BatchNorm2d(32, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # [B, 64, 8, 8]
            nn.BatchNorm2d(64, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            Flatten(),  # [B, 64 * 4 * 4]
            nn.Linear(64 * embed_size ** 2, ARCH.Z_COMP_DIM * 2),
        )

        # self.rob_cond_comp = ActionCompCondEnc()
        # here batch is B * K
        self.comp_cond = nn.Sequential(
                nn.Linear(2 * ARCH.Z_COMP_DIM + ARCH.RNN_CTX_COMP_HIDDEN_DIM + ARCH.ACTION_ENHANCE
                    , ARCH.COMP_COND_HIDDEN_DIM),
                nn.CELU(),
                nn.Linear(ARCH.COMP_COND_HIDDEN_DIM, ARCH.COMP_COND_HIDDEN_DIM),
                nn.CELU(),
                nn.Linear(ARCH.COMP_COND_HIDDEN_DIM, 2 * ARCH.Z_COMP_DIM))



    def forward(self, x, action, h_comp_t=None):
        """
        Predict component latent parameters given image and predicted mask concatenated
        :param x: (B*K, 3+1, H, W). Image and mask concatenated
               cond: the conditioning stat eof the agent of shape (B, A)
               h_comp_t:
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
            h_comp_t: (B*K, Hc)
        """
        BK = x.size(0)  # B * K dim
        x = self.enc(x)  # [B*K, 2 * Zc]
        x = x.view(action.size(0), ARCH.K, x.size(-1))  # [B, K, 2 * Zc]
        # TODO (chmin): validate this!
        action_repeat = action[:, None].repeat(1, ARCH.K, 1)
        x = self.comp_cond(torch.cat([x, h_comp_t, action_repeat], dim=-1))

        # x = self.cond_temporal(torch.cat([x, h_comp_t], dim=-1))
        # action_repeat = action[:, None].repeat(1, ARCH.K, 1)
        # x = self.rob_cond_comp(torch.cat([x, cond_repeat], dim=-1))
        # # first condition temporal info: h^c_{k,t}
        # # then condition the agent info: a_{t-1}

        x = x.view(BK, x.size(-1))
        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4
        return z_comp_loc, z_comp_scale

class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))  # [H + 8, W + 8]
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)  # [2, H + 8, W + 8]
        # (B, 2, H, W)
        coords = coords[None].expand(B, 2, height, width)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)

        self.decoder = nn.Sequential(
            nn.Conv2d(ARCH.Z_COMP_DIM + 2, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # nn.Conv2d(32, 32, 3, 1),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            # 16x downsampled: (32, 4, 4)
            nn.Conv2d(32, 3, 1, 1),
        )

    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = ARCH.IMG_SHAPE
        # (B, L) -> (B, L+2, H, W)
        z_comp = self.spatial_broadcast(z_comp, h + 6, w + 6)  # add 8 as we apply 4 dim reducing Convs
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompCondDecoder(nn.Module):
    """
    Decoder z_comp into component image, conditioned on robot priors.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(ARCH.Z_COMP_DIM + 2, 64, 3, 1),
        #     nn.CELU(),
        #     nn.GroupNorm(8, 64),

        #     nn.Conv2d(64, 32, 3, 1),
        #     nn.CELU(),
        #     nn.GroupNorm(4, 32),

        #     nn.Conv2d(32, 32, 3, 1),
        #     nn.CELU(),
        #     nn.GroupNorm(4, 32),

        #     nn.Conv2d(32, 3, 1, 1),
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(ARCH.Z_COMP_DIM + 2, 32, 3, 1),
            nn.BatchNorm2d(32, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32, track_running_stats=ARCH.LEARN_BATCH_NORM),
            nn.ELU(),
            nn.Conv2d(32, 3, 1, 1),
        )
        # floor [H_in + 2 * padding - dilation * (kernel - 1)  -1 ]


        self.rob_cond_comp = ActionCompCondDec()

    def forward(self, z_comp, cond):
        """-
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        H, W = ARCH.IMG_SHAPE
        # (B, L) -> (B, L+2, H, W)
        BK = z_comp.size(0)
        z_comp = z_comp.view(cond.size(0), ARCH.K, z_comp.size(-1))
        cond_repeat = cond[:, None].repeat(1, ARCH.K, 1)
        z_comp = self.rob_cond_comp(torch.cat([z_comp, cond_repeat], dim=-1))

        z_comp = z_comp.view(BK, z_comp.size(-1))
        z_comp = self.spatial_broadcast(z_comp, H + 6, W  + 6)  # add 8 as we apply 4 dim reducing Convs
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):

    def __init__(self):
        super(CompDecoderStrong, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(ARCH.Z_COMP_DIM, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)

        )

    def forward(self, x):
        """
        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.nn.Hardsigmoid()(self.dec(x))
        return comp


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent. Used to infer z_comp_prior
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.RNN_CTX_COMP_HIDDEN_DIM + ARCH.Z_COMP_DIM
                    , ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.Z_COMP_DIM * 2),
        )

    def forward(self, z_mask, z_comp=None, h_comp_t=None):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        if h_comp_t is None:  # TODO: residual
            x = self.mlp(z_mask)
        elif z_comp is None:
            x = self.mlp(torch.cat([z_mask, h_comp_t], dim=-1))
        else:
            x = self.mlp(torch.cat([z_mask, z_comp, h_comp_t], dim=-1))

        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4
        return z_comp_loc, z_comp_scale, x


class PredictCondComp(nn.Module):
    """
    Predict component latents given mask latent and conditioning robot states. Used to infer z_comp_prior
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM, ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.Z_COMP_DIM * 2),
        )

    def forward(self, h, cond):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4

        return z_comp_loc, z_comp_scale