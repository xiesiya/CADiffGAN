import torch
import torch.nn as nn


def diffusion_schedule(timesteps, beta_min=1e-4, beta_max=2e-2,
                       device=torch.device('cpu')):
    betas = torch.linspace(beta_min, beta_max, timesteps,
                           dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


class AttentionGuidedDiffusion(nn.Module):
    def __init__(self, timesteps=100, beta_min=1e-4, beta_max=2e-2,
                 attn_weight=1.0):
        super().__init__()
        self.timesteps = timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.attn_weight = attn_weight

    def forward(self, x, attn_scores=None):
        device = x.device
        noise_rates, signal_rates = diffusion_schedule(
            self.timesteps, self.beta_min, self.beta_max, device)

        B = x.size(0)
        if attn_scores is None or self.attn_weight == 0:
            t = torch.randint(0, self.timesteps, (1,), device=device).item()
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        else:
            attn_norm = attn_scores - attn_scores.mean()
            t_ratio = torch.sigmoid(-self.attn_weight * attn_norm)
            t_batch = (t_ratio * (self.timesteps - 1)).long().clamp(0, self.timesteps - 1)

        noise = torch.randn_like(x)
        sr = signal_rates[t_batch].unsqueeze(1)
        nr = noise_rates[t_batch].unsqueeze(1)
        noisy_x = sr * x + nr * noise

        return noisy_x, t_batch
