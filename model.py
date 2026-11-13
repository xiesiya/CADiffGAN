import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_encoder import CausalMetaGraphEncoder
from diffusion import AttentionGuidedDiffusion


class Generator(nn.Module):
    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr,
                 n_drug, attn_dim=64, edge_emb_dim=32, dropout=0.2,
                 timesteps=100, beta_min=1e-4, beta_max=2e-2,
                 attn_diff_weight=1.0):
        super().__init__()
        self.n_hid = n_hid
        self.n_drug = n_drug

        self.encoder = CausalMetaGraphEncoder(
            in_dims=in_dims, n_hid=n_hid, n_adjs=n_adjs,
            n_steps=n_steps, cstr=cstr,
            attn_dim=attn_dim, edge_emb_dim=edge_emb_dim,
            dropout=dropout)

        self.diffusion = AttentionGuidedDiffusion(
            timesteps=timesteps, beta_min=beta_min, beta_max=beta_max,
            attn_weight=attn_diff_weight)

        self.predictor = nn.Sequential(
            nn.Linear(n_drug + n_hid, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_drug),
            nn.Sigmoid(),
        )

    def forward(self, node_feats, node_types, adjs, adj_rows,
                mirna_indices, n_mirna, idxes_seq=None, idxes_res=None):
        node_emb, node_attn = self.encoder(
            node_feats, node_types, adjs, idxes_seq, idxes_res)

        mirna_emb = node_emb[:n_mirna]
        mirna_attn = node_attn[:n_mirna]

        mirna_emb = F.normalize(mirna_emb, p=1, dim=1)

        batch_emb = mirna_emb[mirna_indices]
        batch_attn = mirna_attn[mirna_indices]

        fake_emb, t_batch = self.diffusion(batch_emb, batch_attn)

        M = torch.cat([adj_rows, fake_emb], dim=1)
        pred_rows = self.predictor(M)

        return fake_emb, pred_rows, t_batch

    def alphas(self):
        return self.encoder.alphas()

    def sample_paths(self, eps):
        return self.encoder.sample(eps)

    def parse_architecture(self):
        return self.encoder.parse()


class Discriminator(nn.Module):
    def __init__(self, n_drug, n_hid, weight_clip=0.01):
        super().__init__()
        self.weight_clip = weight_clip
        self.net = nn.Sequential(
            nn.Linear(n_drug + n_hid, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
        )

    def forward(self, adj_rows, embeddings):
        x = torch.cat([adj_rows, embeddings], dim=1)
        return self.net(x)

    def clip_weights(self):
        for p in self.parameters():
            p.data.clamp_(-self.weight_clip, self.weight_clip)

    def gradient_penalty(self, real_adj, fake_adj, real_emb, fake_emb):
        device = real_adj.device
        B = real_adj.size(0)
        alpha = torch.rand(B, 1, device=device)
        interp_adj = (alpha * real_adj + (1 - alpha) * fake_adj).requires_grad_(True)
        interp_emb = (alpha * real_emb + (1 - alpha) * fake_emb).requires_grad_(True)
        pred = self.forward(interp_adj, interp_emb)
        grads = torch.autograd.grad(
            outputs=pred, inputs=[interp_adj, interp_emb],
            grad_outputs=torch.ones_like(pred),
            create_graph=True, retain_graph=True)
        grad_norm = torch.cat([g.view(B, -1) for g in grads], dim=1).norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp
