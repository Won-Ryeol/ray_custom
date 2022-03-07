import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from .arch import ARCH
from .module import Flatten, MLP


class BgModule(nn.Module):
    def __init__(self, action_dim=7):
        nn.Module.__init__(self)
        self.embed_size = ARCH.IMG_SIZE // 16
        # Image encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            
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
        self.dec_fc = nn.Linear(ARCH.Z_CTX_DIM, self.embed_size ** 2 * 128)
        # Decoder latent into background
        self.dec = nn.Sequential(
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
            
            nn.Conv2d(16, 3 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )
        # self.dec = BgDecoder()
        
        self.rnn_post = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.rnn_prior = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.h_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.c_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.h_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.c_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.prior_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM  + ARCH.ACTION_ENHANCE, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
        self.post_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM + ARCH.IMG_ENC_DIM + ARCH.ACTION_ENHANCE
            , 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
    
        self.action_enhance = nn.Sequential(
            nn.Linear(ARCH.ACTION_DIM, ARCH.ACTION_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.ACTION_HIDDEN_DIM, ARCH.ACTION_ENHANCE)
        )




    def forward(self, seq):
        return self.encode(seq)
    
    def anneal(self, global_step):
        pass
    
    def infer(self, history, obs, action, episodic_step=0, first=False):
        """
        Encode input frames into context latents
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, 3, H, W)
        B, C, H, W = obs.size()
        A = action.size(-1)
        action = torch.squeeze(action, 1)
        action = self.action_enhance(action) # [B, T, A]

        # Encode images
        # (B, C, H, W)
        enc = self.enc(obs.reshape(B, 3, H, W))
        # (B, D)
        enc = enc.flatten(start_dim=1)
        # (B*, D)
        enc = self.enc_fc(enc)
        # (B, D)
        enc = enc.view(B, ARCH.IMG_ENC_DIM)
        
        history = [torch.squeeze(t, 1) for t in history] # reduce the temporal axis

        # z_ctx_prev is not needed as it is not residual.
        z_ctx_prev, h_post, c_post = history
        
        # Compute posterior
        # (B, D)

        post_input = torch.cat([h_post, enc, action], dim=-1)
        # (B, D), (B, D)
        params = self.post_net(post_input)
        # (B, D), (B, D)
        loc, scale = torch.chunk(params, 2, dim=-1)
        scale = F.softplus(scale) + 1e-4
        # (B, D)
        z_ctx_post = Normal(loc, scale)
        # (B, D)
        z_ctx = z_ctx_post.rsample()

        # Temporal encode
        h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
        h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B, 3, H, W)
        bg = self.dec(
            self.dec_fc(z_ctx).
                view(B, 128, self.embed_size, self.embed_size)
        )
        
        # Reshape
        bg = bg.view(B, 3, H, W)
        z_ctx = z_ctx.view(B, ARCH.Z_CTX_DIM)
        
        things = dict(
            bg=bg,  # (B, 3, H, W)
            z_ctx=z_ctx,  # (B, D),
            h_ctx_post=h_post,  # (B, D),
            c_ctx_post=c_post,  # (B, D),
            h_ctx_prior=h_prior,  # (B, D),
            c_ctx_prior=c_prior,  # (B, D),
            enhanced_act=action
        )
        
        return things

    def imagine(self, history, z_prevs, episodic_step=0):
        """
            single step generation of imagined trajectory of the agent.
        """
        h_prior, c_prior = history
        z_ctx_prev = z_prevs

        B = h_prior.size(0) # B is actually
        A = action.size(-1) 

        if A == ARCH.ACTION_DIM: # if action is not enhanced
            action = self.action_enhance(action) # [B * T, A]

        h_prior = h_prior.reshape(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = c_prior.reshape(B, ARCH.RNN_CTX_HIDDEN_DIM)

        prior_input = torch.cat([h_prior, action[:, t]], dim=-1)
        params = self.prior_net(prior_input)

        loc, scale = torch.chunk(params, 2, dim=-1)
        scale = F.softplus(scale) + 1e-4
        z_ctx_prior = Normal(loc, scale)
        z_ctx = z_ctx_prior.sample() if sample else loc
        h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))

        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B, 128, self.embed_size, self.embed_size)
        )
        bg = bg.view(B, 3, H, W)

        things = {
            'bg' : bg, # [B, 3, 64, 64]
            'h_prior': h_prior,
            'c_prior': c_prior,
            'z_ctx': z_ctx,
            'enhanced_act': action # [B, A]
        }

        return things

    def encode(self, seq, action, global_step=0):
        """
        Encode input frames into context latents
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        
        A = action.size(-1)
        if A != ARCH.ACTION_ENHANCE:
            action = self.action_enhance(action)  # [B, T, A]

        # Encode images
        # (B*T, C, H, W)
        enc = self.enc(seq.reshape(B * T, 3, H, W))
        # I deliberately do this ugly thing because for future version we may need enc to do bg interaction
        # (B*T, D)
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = self.enc_fc(enc)
        # (B, T, D)
        enc = enc.view(B, T, ARCH.IMG_ENC_DIM)
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        
        # (B,)
        kl_list = []
        z_ctx_list = []
        for t in range(T):
            # Compute posterior
            # (B, D)
            # TODO (chmin): condition action here!
            post_input = torch.cat([h_post, enc[:, t], action[:, t]], dim=-1)
            # (B, D), (B, D)
            params = self.post_net(post_input)
            # (B, D), (B, D)
            loc, scale = torch.chunk(params, 2, dim=-1)
            scale = F.softplus(scale) + 1e-4
            # (B, D)
            z_ctx_post = Normal(loc, scale)
            # (B*T, D)
            z_ctx = z_ctx_post.rsample()
            
            # Compute prior
            prior_input = torch.cat([h_prior, action[:, t]], dim=-1)
            params = self.prior_net(prior_input)
            loc, scale = torch.chunk(params, 2, dim=-1)
            scale = F.softplus(scale) + 1e-4
            z_ctx_prior = Normal(loc, scale)
            
            # Temporal encode
            h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
            h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
            # Compute KL Divergence
            # (B, D)
            kl = kl_divergence(z_ctx_post, z_ctx_prior)
            assert kl.size()[-1] == ARCH.Z_CTX_DIM
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
            kl_list.append(kl.sum(-1))
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        
        # Reshape
        bg = bg.view(B, T, 3, H, W)
        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        # (B, T)
        kl_bg = torch.stack(kl_list, dim=1)
        assert kl_bg.size() == (B, T)
        
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx,  # (B, T, D)
            kl_bg=kl_bg,  # (B, T)
            enhanced_act=action
        )
        
        return things
    
    def generate(self, seq, cond_steps, action, sample):
        """
        Generate new frames given a set of input frames
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        A = action.size(-1)
            if A != ARCH.ACTION_ENHANCE:
            action = self.action_enhance(action)
        # Encode images. Only needed for the first few steps
        # (B*T, C, H, W)
        enc = self.enc(seq[:, :cond_steps].reshape(B * cond_steps, 3, H, W))
        # (B*T, D)
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = self.enc_fc(enc)
        # (B, T, D)
        enc = enc.view(B, cond_steps, ARCH.IMG_ENC_DIM)
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # (B,)
        z_ctx_list = []
        for t in range(T):
            
            if t < cond_steps:
                # Compute posterior
                # (B, D)
                post_input = torch.cat([h_post, enc[:, t], action[:, t]], dim=-1)
                # (B, D), (B, D)
                params = self.post_net(post_input)
                # (B, D), (B, D)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                # (B, D)
                z_ctx_post = Normal(loc, scale)
                # (B*T, D)
                z_ctx = z_ctx_post.sample()
                
                # Temporal encode
                h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            else:
                # Compute prior
                prior_input = torch.cat([h_prior, action[:, t]], dim=-1)
                params = self.prior_net(prior_input)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                z_ctx_prior = Normal(loc, scale)
                z_ctx = z_ctx_prior.sample() if sample else loc
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        bg = bg.view(B, T, 3, H, W)
        # Split into lists of length t
        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx,  # (B, T, D)
            enhanced_act=action
        )
        
        return things
