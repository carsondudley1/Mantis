import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConvEmbedding(nn.Module):
    """
    Multi-scale convolutional embedding layer that captures patterns at different time scales.

    This module uses parallel convolutions with different kernel sizes to capture:
    - Short-term patterns (e.g., 1-3 days)
    - Medium-term patterns (e.g., weekly cycles)
    - Long-term patterns (e.g., incubation periods, seasonal effects)
    """
    def __init__(self, input_dim=1, output_dim=512, dropout=0.1):
        super().__init__()

        # Different kernel sizes to capture multi-scale patterns
        self.conv_short = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(input_dim, output_dim // 4, kernel_size=15, padding=7)
        self.conv_vlong = nn.Conv1d(input_dim, output_dim // 4, kernel_size=31, padding=15)

        # Layer normalization per scale
        self.norm_short = nn.LayerNorm(output_dim // 4)
        self.norm_med = nn.LayerNorm(output_dim // 4)
        self.norm_long = nn.LayerNorm(output_dim // 4)
        self.norm_vlong = nn.LayerNorm(output_dim // 4)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Final projection combining all scales
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Embedded tensor of shape [batch_size, seq_len, output_dim]
        """
        # Transpose for 1D convolution [batch_size, input_dim, seq_len]
        x_conv = x.transpose(1, 2)

        # Apply convolutions at different scales
        x_short = self.conv_short(x_conv).transpose(1, 2)  # [B, seq_len, output_dim//4]
        x_med = self.conv_med(x_conv).transpose(1, 2)
        x_long = self.conv_long(x_conv).transpose(1, 2)
        x_vlong = self.conv_vlong(x_conv).transpose(1, 2)

        # Apply normalization and activation
        x_short = self.dropout(self.activation(self.norm_short(x_short)))
        x_med = self.dropout(self.activation(self.norm_med(x_med)))
        x_long = self.dropout(self.activation(self.norm_long(x_long)))
        x_vlong = self.dropout(self.activation(self.norm_vlong(x_vlong)))

        # Concatenate multi-scale features
        x_combined = torch.cat([x_short, x_med, x_long, x_vlong], dim=2)

        # Final projection
        x_out = self.final_proj(x_combined)
        x_out = self.final_norm(x_out)

        return x_out

class TemporalPatternAttention(nn.Module):
    """
    Custom attention mechanism that emphasizes temporal patterns.
    Includes relative positional encoding and adaptive attention span.
    """
    def __init__(self, hidden_dim, n_heads, dropout=0.1, max_len=2000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Relative positional bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
        positions = torch.arange(max_len).unsqueeze(1) - torch.arange(max_len).unsqueeze(0)
        positions = positions + max_len - 1  # Shift to [0, 2*max_len-1]
        self.register_buffer('positions', positions)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
                  where True indicates valid positions

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        rel_pos = self.rel_pos_bias[self.positions[:seq_len, :seq_len]]  # [seq_len, seq_len, n_heads]
        rel_pos = rel_pos.permute(2, 0, 1)  # [n_heads, seq_len, seq_len]
        attn_scores = attn_scores + rel_pos.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            invalid = ~mask
            invalid = invalid.unsqueeze(1).unsqueeze(2)
            invalid = invalid.expand(-1, self.n_heads, x.size(1), -1)
            attn_scores = attn_scores.masked_fill(invalid, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]

        # Reshape back: [batch_size, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.out_proj(output)

        return output

class CNNTransformerBlock(nn.Module):
    """
    Hybrid block combining CNN for local pattern extraction with transformer for global context.
    """
    def __init__(self, hidden_dim, ffn_dim, n_heads, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        # Local pattern CNN
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # Custom pattern-oriented attention
        self.pattern_attn = TemporalPatternAttention(hidden_dim, n_heads, dropout)

        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Local CNN processing - using post-norm architecture
        x_conv = x.transpose(1, 2)
        x_conv = self.local_conv(x_conv).transpose(1, 2)
        x = x + self.dropout(x_conv)
        x = self.norm1(x)

        # Global attention - using post-norm architecture
        x_attn = self.pattern_attn(x, mask)
        x = x + self.dropout(x_attn)
        x = self.norm2(x)

        # Feed-forward network - using post-norm architecture
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        x = self.norm3(x)

        return x

class EpidemicPatternMemory(nn.Module):
    """
    Pattern Memory Bank for disease forecasting that stores learned prototype patterns.

    This module maintains a bank of learned disease pattern prototypes that represent
    common patterns observed across various diseases and outbreaks (e.g., seasonal surges,
    intervention responses, reporting anomalies). The model can match input sequences
    against these patterns and incorporate relevant pattern knowledge into its predictions.
    """
    def __init__(self, hidden_dim, num_patterns=256, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        # Learnable disease pattern prototypes - initialized with small random values
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        # Pattern matching network - computes similarity between input and patterns
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns)
        )

        # Projection for retrieved patterns
        self.pattern_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the pattern memory bank.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional boolean mask tensor of shape [batch_size, seq_len]
                  where True indicates valid positions

        Returns:
            Enhanced tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Apply pre-normalization (similar to transformer blocks)
        x_norm = self.norm1(x)

        # Match input patterns against stored prototypes
        pattern_weights = self.pattern_matcher(x_norm)  # [B, seq_len, num_patterns]

        # Apply attention mask if provided
        if mask is not None:
            # Create mask in correct shape: [B, seq_len, 1]
            mask_expanded = mask.unsqueeze(-1).float()
            # Apply mask (invalid positions will have zero weight)
            pattern_weights = pattern_weights * mask_expanded

        # Apply softmax to get attention distribution over patterns
        pattern_weights = F.softmax(pattern_weights, dim=-1)  # [B, seq_len, num_patterns]

        # Retrieve weighted combination of pattern prototypes
        retrieved_patterns = torch.matmul(pattern_weights, self.pattern_bank)  # [B, seq_len, hidden_dim]

        # Project retrieved patterns
        retrieved_patterns = self.pattern_proj(retrieved_patterns)

        # Apply dropout for regularization
        retrieved_patterns = self.dropout(retrieved_patterns)

        # Combine with input using residual connection
        x_with_patterns = x + retrieved_patterns

        # Apply final layer normalization
        return self.norm2(x_with_patterns)

class MultiTimeSeriesForecaster(nn.Module):
    """
    Enhanced foundation model for disease forecasting using a hybrid CNN-Transformer architecture
    with a pattern memory bank. Now supports multiple time series types (cases, hospitalizations, deaths).

    Key enhancements:
    1. Adds time series type embedding to switch between forecasting different types
    2. Pattern memory bank that stores and recalls common disease patterns
    3. Richer temporal features derived from week indices
    4. Two-layer GRU in the decoder for better sequence modeling
    5. Designed for weekly data with 4-week forecast horizon
    """
    def __init__(
        self,
        input_window=112,  # 112 weeks of context
        forecast_horizon=4,  # 4 weeks forecast
        hidden_dim=512,
        ffn_dim=768,
        n_layers=8,
        n_heads=8,
        n_quantiles=9,
        disease_embed_dim=64,
        pop_embed_dim=64,
        binary_feat_dim=32,
        teacher_forcing_ratio=0.1,
        dropout=0.1,
        layer_norm_eps=1e-5
    ):
        super().__init__()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_quantiles = n_quantiles
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # ===== FEATURE EMBEDDINGS =====
        # Multi-scale convolutional embedding for time series data
        self.values_embedding = MultiScaleConvEmbedding(
            input_dim=2,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Disease type embedding
        self.disease_embedding = nn.Embedding(3, disease_embed_dim)
        self.disease_norm = nn.LayerNorm(disease_embed_dim, eps=layer_norm_eps)

        # Population embedding
        self.population_mlp = nn.Sequential(
            nn.Linear(1, pop_embed_dim),
            nn.LayerNorm(pop_embed_dim, eps=layer_norm_eps),
            nn.GELU()
        )

        # Enhanced temporal embeddings - derived from day indices
        self.day_of_week_embed = nn.Embedding(7, 128)        # 0-6 for day of week
        self.day_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Month embedding - approximate by dividing day index by 30
        self.month_embed = nn.Embedding(12, 128)             # 0-11 for month
        self.month_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Day of year embedding - approximate by modulo 365
        self.day_of_year_embed = nn.Embedding(366, 128)      # 0-365 for day of year
        self.day_of_year_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Add target type embedding alongside disease type
        self.target_type_embedding = nn.Embedding(3, 128)  # 3 types: cases, hosp, death
        self.target_type_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Input feature projection
        input_feat_dim = hidden_dim + disease_embed_dim + pop_embed_dim + 128 + 128 + 128 + 128
        self.feature_projection = nn.Linear(input_feat_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # ===== ENCODER BLOCKS =====
        self.encoder_blocks = nn.ModuleList([
            CNNTransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                n_heads=n_heads,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers)
        ])

        # ===== PATTERN MEMORY BANK =====
        # Add pattern memory bank after encoder blocks with 256 patterns
        self.pattern_memory = EpidemicPatternMemory(
            hidden_dim=hidden_dim,
            num_patterns=256,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        # ===== DECODER COMPONENTS =====
        # Initial decoder state projection
        self.decoder_init_proj = nn.Linear(hidden_dim, hidden_dim)

        # Input processing for each decoder step
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4, eps=layer_norm_eps),
            nn.GELU()
        )

        # Cross-attention from decoder to encoder memory
        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Two-layer GRU for maintaining state across autoregressive steps
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=2,  # Enhanced: Using 2 layers
            batch_first=True
        )

        # Output projections for quantiles
        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_quantiles)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for better initial training"""
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.01)
            # Initialize pattern bank with small values
            elif 'pattern_bank' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        values,               # [B, seq_len] scaled inputs
        disease_type,         # [B] long tensor
        target_type,
        population,           # [B] float tensor
        day_indices,          # [B, seq_len] absolute day indices
        valid_mask=None,
        target_values=None
    ):
        """
        Forward pass through the model with enhanced temporal features and pattern memory.

        Args:
            values: Scaled input values [batch_size, seq_len]
            disease_type: Disease type indices [batch_size]
            time_series_type: Time series type indices [batch_size] (0=cases, 1=hosp, 2=death)
            population: Scaled population values [batch_size]
            day_indices: Day indices for temporal patterns [batch_size, seq_len]
            valid_mask: Mask for padding [batch_size, seq_len], True for valid
            target_values: Optional targets for teacher forcing [batch_size, forecast_horizon]

        Returns:
            predictions: Quantile predictions [batch_size, forecast_horizon, n_quantiles]
        """
        batch_size, seq_len, _ = values.shape
        device = values.device

        # ===== INPUT PROCESSING =====
        # Process value data with multi-scale CNN
        value_features = self.values_embedding(values)

        # Process disease type
        disease_emb = self.disease_embedding(disease_type)  # [B, disease_dim]
        disease_emb = self.disease_norm(disease_emb)


        # Process population
        pop_emb = self.population_mlp(population.unsqueeze(-1))  # [B, pop_dim]

        # Enhanced temporal embeddings from day indices
        # 1. Day of week (0-6)
        day_of_week = (day_indices % 7).long()  # [B, seq_len]
        dow_emb = self.day_of_week_embed(day_of_week)  # [B, seq_len, 128]
        dow_emb = self.day_norm(dow_emb)

        # 2. Month approximation (0-11) by dividing day index by 30
        month_approx = ((day_indices // 30) % 12).long()  # [B, seq_len]
        month_emb = self.month_embed(month_approx)  # [B, seq_len, 128]
        month_emb = self.month_norm(month_emb)

        # 3. Day of year approximation (0-365)
        day_of_year = (day_indices % 366).long()  # [B, seq_len]
        doy_emb = self.day_of_year_embed(day_of_year)  # [B, seq_len, 128]
        doy_emb = self.day_of_year_norm(doy_emb)

        # Expand static features to time dimension
        disease_emb_exp = disease_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        pop_emb_exp = pop_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        target_type_emb = self.target_type_embedding(target_type)
        target_type_emb = self.target_type_norm(target_type_emb)
        target_type_emb_exp = target_type_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Concatenate all features
        combined_features = torch.cat([
            value_features, disease_emb_exp, pop_emb_exp,
            target_type_emb_exp, dow_emb, month_emb, doy_emb
        ], dim=-1)

        # Project to hidden dimension
        encoder_input = self.feature_projection(combined_features)
        encoder_input = self.feature_norm(encoder_input)

        # ===== ENCODER =====
        encoder_output = encoder_input
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, valid_mask)

        # ===== PATTERN MEMORY =====
        # Apply pattern memory after encoder blocks
        encoder_output = self.pattern_memory(encoder_output, valid_mask)

        # ===== AUTOREGRESSIVE DECODER =====
        # Get initial decoder state (weighted pooling of encoder outputs)
        if valid_mask is not None:
            # Perform masked pooling
            mask_expanded = valid_mask.unsqueeze(-1).float()
            weighted_sum = (encoder_output * mask_expanded).sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1) + 1e-10
            pooled_state = weighted_sum / mask_sum
        else:
            # Simple mean pooling
            pooled_state = encoder_output.mean(dim=1)  # [B, hidden_dim]

        # Initialize decoder state for 2-layer GRU
        dec_hidden = self.decoder_init_proj(pooled_state)
        # Replicate the same initial state for both GRU layers
        dec_hidden = dec_hidden.unsqueeze(0).repeat(2, 1, 1)  # [2, B, hidden_dim] for 2-layer GRU

        # Initial input (last value from the sequence)
        if seq_len > 0:
            decoder_input = values[:, -1, 0].unsqueeze(-1)
        else:
            decoder_input = torch.zeros(batch_size, 1, device=device)

        # Storage for predictions
        all_quantile_preds = []

        # Determine teacher forcing (only during training)
        use_teacher_forcing = (
            self.training and
            target_values is not None and
            torch.rand(1).item() < self.teacher_forcing_ratio
        )

        # Generate predictions autoregressively
        for t in range(self.forecast_horizon):
            # Embed current input - this should produce [B, hidden_dim//4]
            dec_input_emb = self.decoder_input_proj(decoder_input.unsqueeze(-1))  # [B, hidden_dim//4]

            # Attend to encoder outputs - shape: [B, 1, hidden_dim]
            # Use the top layer of GRU output for attention query
            query = dec_hidden[-1:].transpose(0, 1)  # [B, 1, hidden_dim]
            attn_output, _ = self.decoder_attn(
                query, encoder_output, encoder_output,
                key_padding_mask=None if valid_mask is None else ~valid_mask
            )

            # Ensure dimensions match for concatenation by adding a sequence dimension to dec_input_emb if needed
            # Reshape dec_input_emb to [B, 1, hidden_dim//4]
            if len(dec_input_emb.shape) == 2:
                dec_input_emb = dec_input_emb.unsqueeze(1)

            # Now concatenate along feature dimension
            gru_input = torch.cat([attn_output, dec_input_emb], dim=-1)  # [B, 1, hidden_dim+hidden_dim//4]

            # Update decoder state through 2-layer GRU
            _, dec_hidden = self.decoder_gru(gru_input, dec_hidden)

            # Generate quantile predictions using the top GRU layer's output
            dec_output = dec_hidden[-1, :, :]  # [B, hidden_dim]
            quantile_preds = []
            for q_proj in self.quantile_projections:
                q_pred = q_proj(dec_output)  # [B, 1]
                quantile_preds.append(q_pred)

            # Stack quantiles
            step_pred = torch.cat(quantile_preds, dim=1)  # [B, n_quantiles]
            all_quantile_preds.append(step_pred.unsqueeze(1))  # [B, 1, n_quantiles]

            # Determine next input
            median_idx = self.n_quantiles // 2
            if use_teacher_forcing and t < target_values.size(1):
                decoder_input = target_values[:, t].unsqueeze(-1)
            else:
                decoder_input = step_pred[:, median_idx].unsqueeze(-1)

        # Stack predictions across time steps
        predictions = torch.cat(all_quantile_preds, dim=1)  # [B, forecast_horizon, n_quantiles]
        return predictions

    def predict(
        self,
        values,
        disease_type,
        target_type,
        population,
        day_indices,
        valid_mask=None
    ):
        """Inference helper with no teacher forcing"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(
                values,
                disease_type,
                target_type,
                population,
                day_indices,
                valid_mask=valid_mask,
                target_values=None
            )
        if was_training:
            self.train()
        return preds
