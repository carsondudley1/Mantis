import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTimeSeriesForecaster(nn.Module):
    def __init__(
        self,
        input_window=112,
        forecast_horizon=8,
        hidden_dim=1024,
        ffn_dim=2048,
        n_layers=16,
        n_heads=16,
        n_quantiles=9,
        disease_embed_dim=64,
        pop_embed_dim=64,
        binary_feat_dim=32,
        dropout=0.2,
        layer_norm_eps=1e-5
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.n_quantiles = n_quantiles

        # Input embeddings
        self.values_embedding = nn.Linear(2, hidden_dim)  # already preprocessed
        self.disease_embedding = nn.Embedding(3, disease_embed_dim)
        self.disease_norm = nn.LayerNorm(disease_embed_dim, eps=layer_norm_eps)

        self.population_mlp = nn.Sequential(
            nn.Linear(1, pop_embed_dim),
            nn.LayerNorm(pop_embed_dim, eps=layer_norm_eps),
            nn.GELU()
        )

        self.target_type_embedding = nn.Embedding(3, 128)
        self.target_type_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        self.feature_projection = nn.Linear(
            hidden_dim + disease_embed_dim + pop_embed_dim + 128,
            hidden_dim
        )
        self.feature_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder (GRU + output heads)
        self.decoder_init_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.decoder_input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4, eps=layer_norm_eps),
            nn.GELU()
        )

        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_quantiles)
        ])

    def forward(self, values, disease_type, target_type, population, day_indices, valid_mask=None, target_values=None):
        B, L, _ = values.shape

        x = self.values_embedding(values)

        disease_emb = self.disease_embedding(disease_type)
        disease_emb = self.disease_norm(disease_emb).unsqueeze(1).expand(B, L, -1)

        target_emb = self.target_type_embedding(target_type)
        target_emb = self.target_type_norm(target_emb).unsqueeze(1).expand(B, L, -1)

        pop_emb = self.population_mlp(population.unsqueeze(-1)).unsqueeze(1).expand(B, L, -1)

        x = torch.cat([x, disease_emb, pop_emb, target_emb], dim=-1)
        x = self.feature_projection(x)
        x = self.feature_norm(x)

        if valid_mask is not None:
            x = self.encoder(x, src_key_padding_mask=~valid_mask)
        else:
            x = self.encoder(x)

        pooled = x.mean(dim=1)
        dec_hidden = self.decoder_init_proj(pooled).unsqueeze(0).repeat(2, 1, 1)

        decoder_input = values[:, -1, 0].unsqueeze(-1)

        outputs = []
        for t in range(self.forecast_horizon):
            dec_input_emb = self.decoder_input_proj(decoder_input.unsqueeze(-1))
            query = dec_hidden[-1:].transpose(0, 1)
            attn_out, _ = self.decoder_attn(query, x, x, key_padding_mask=~valid_mask if valid_mask is not None else None)

            if len(dec_input_emb.shape) == 2:
                dec_input_emb = dec_input_emb.unsqueeze(1)

            gru_input = torch.cat([attn_out, dec_input_emb], dim=-1)
            _, dec_hidden = self.decoder_gru(gru_input, dec_hidden)

            dec_out = dec_hidden[-1]
            step_qs = torch.cat([proj(dec_out) for proj in self.quantile_projections], dim=-1)
            outputs.append(step_qs.unsqueeze(1))
            decoder_input = step_qs[:, self.n_quantiles // 2].unsqueeze(-1)

        return torch.cat(outputs, dim=1)

    def predict(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(*args, **kwargs)
        if was_training:
            self.train()
        return preds
