import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, batch_first=True, num_heads=num_heads)
        self.ln = nn.LayerNorm([hidden_dim])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, key_padding_mask):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SimpleLatentDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers=1, eps=1e-5):
        super(SimpleLatentDenoiser, self).__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.mlp_in = nn.Linear(latent_dim, hidden_dim)
        mlp_mid, sa, t_emb = [], [], []
        for _ in range(n_layers):
            mlp_mid.append(nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            sa.append(SelfAttention(hidden_dim=hidden_dim, num_heads=8))
            t_emb.append(nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.mlp_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.mlp_mid = nn.ModuleList(mlp_mid)
        self.sa = nn.ModuleList(sa)
        self.t_emb = nn.ModuleList(t_emb)

    def pos_encoding(self, t, proj_dims):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, proj_dims, 2, device=t.device).float() / proj_dims)
        )
        pos_enc_a = torch.sin(t.unsqueeze(-1).repeat(1, proj_dims // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(-1).repeat(1, proj_dims // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, latents, t, face_mask=None):
        """
        Conditioning made on the latens and their original position
        :param latents: noisy latent vector
        :param pos: 3d coordinate position
        :param t: timestep
        :param face_mask: mask applied on the node
        :return: noise/velocity prediction
        """
        if face_mask is not None:
            face_mask = ~face_mask
        t_pos_encod = self.pos_encoding(t, self.hidden_dim)
        latents = self.mlp_in(latents)
        for sa, mlp, t_emb in zip(self.sa, self.mlp_mid, self.t_emb):
            latents = mlp(latents) + t_emb(t_pos_encod).unsqueeze(1)  # unsqueeze() to include seq_len dimension
            latents = sa(latents, key_padding_mask=face_mask)
        latents = self.mlp_out(latents)
        return latents


if __name__ == '__main__':
    model = SimpleLatentDenoiser(latent_dim=16, hidden_dim=8, n_layers=1, eps=1e-5)
    random_tensor = torch.randn((5, 16))
    timestep = torch.ones((5,))
    print(model(random_tensor, timestep).shape)