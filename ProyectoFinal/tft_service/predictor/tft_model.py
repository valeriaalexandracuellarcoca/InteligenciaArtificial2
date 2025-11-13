import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- CLASE 1: GatedLinearUnit ---
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        values = self.linear1(x)
        gates = torch.sigmoid(self.linear2(x))
        output = values * gates
        return self.dropout(output)

# --- CLASE 2: GatedResidualNetwork ---
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=None, output_size=None,
                 dropout=0.1, context_size=None):
        super().__init__()
        if hidden_size is None: hidden_size = input_size
        if output_size is None: output_size = input_size
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.linear2 = nn.Linear(hidden_size + context_size, hidden_size)
        else:
            self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        if input_size != output_size:
            self.skip_projection = nn.Linear(input_size, output_size)
        else:
            self.skip_projection = None
    def forward(self, x, context=None):
        residual = x
        x = F.elu(self.linear1(x))
        if context is not None and self.context_size is not None:
            x = torch.cat([x, context], dim=-1)
        x = F.elu(self.linear2(x))
        x = self.glu(x)
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        return self.layer_norm(x + residual)

# --- CLASE 3: VariableSelectionNetwork ---
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_variables, hidden_size, dropout=0.1):
        super().__init__()
        self.num_variables = num_variables
        self.input_size = input_size
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_variables)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_size * num_variables, hidden_size, num_variables, dropout
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, variables):
        processed_vars = []
        for i, var in enumerate(variables):
            processed = self.variable_grns[i](var)
            processed_vars.append(processed)
        all_vars = torch.cat(variables, dim=-1)
        selection_weights = self.selection_grn(all_vars)
        selection_weights = self.softmax(selection_weights)
        weighted_vars = []
        for i, processed_var in enumerate(processed_vars):
            weight = selection_weights[..., i:i+1]
            weighted_var = processed_var * weight
            weighted_vars.append(weighted_var)
        selected_output = torch.cat(weighted_vars, dim=-1)
        return selected_output, selection_weights

# --- CLASE 4: InterpretableMultiHeadAttention ---
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_kv = key.size(1)
        residual = query
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.w_o(attention_output)
        return self.layer_norm(output + residual), attention_weights
    def scaled_dot_product_attention(self, Q, K, V, mask=None, dropout=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# --- CLASE 5: TemporalFusionTransformer ---
class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 num_static_vars=0, num_historical_vars=1, num_future_vars=0,
                 sequence_length=5, prediction_length=1, hidden_size=64,
                 num_heads=4, num_quantiles=3, dropout=0.1,
                 categorical_indices=None, vocab_sizes=None):
        super().__init__()
        self.num_static_vars = num_static_vars
        self.num_historical_vars = num_historical_vars
        self.num_future_vars = num_future_vars
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.vocab_sizes = vocab_sizes if vocab_sizes is not None else []
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_size) for v in self.vocab_sizes])
        self.historical_projection = nn.Linear(1, hidden_size)
        if num_future_vars > 0:
            self.future_projection = nn.Linear(1, hidden_size)
        if num_static_vars > 0:
            self.static_projection = nn.Linear(num_static_vars, hidden_size)
        if num_static_vars > 0:
            self.static_vsn = VariableSelectionNetwork(hidden_size, num_static_vars, hidden_size, dropout)
        if num_historical_vars > 0:
            self.historical_vsn = VariableSelectionNetwork(hidden_size, num_historical_vars, hidden_size, dropout)
        if num_future_vars > 0:
            self.future_vsn = VariableSelectionNetwork(hidden_size, num_future_vars, hidden_size, dropout)
        
        self.lstm_encoder = nn.LSTM(
            hidden_size * num_historical_vars, hidden_size, batch_first=True, dropout=dropout
        )
        decoder_input_size = hidden_size * num_historical_vars
        if num_future_vars > 0:
            decoder_input_size = hidden_size * num_future_vars
        self.lstm_decoder = nn.LSTM(
            decoder_input_size, hidden_size, batch_first=True, dropout=dropout
        )
        
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
        self.post_attention_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout)
        )
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, num_quantiles)
    
    def forward(self, historical_data, future_data=None, static_data=None):
        historical_vars = []
        for i in range(self.num_historical_vars):
            if i in self.categorical_indices:
                cat_idx = self.categorical_indices.index(i)
                cat_values = historical_data[:, :, i].long()
                embedded = self.embeddings[cat_idx](cat_values)
                historical_vars.append(embedded)
            else:
                cont_value = historical_data[:, :, i:i+1]
                projected = self.historical_projection(cont_value)
                historical_vars.append(projected)

        selected_historical, hist_weights = self.historical_vsn(historical_vars)
        encoded_seq, (hidden, cell) = self.lstm_encoder(selected_historical)

        if future_data is not None and self.num_future_vars > 0:
            future_vars = []
            for i in range(self.num_future_vars):
                cont_value = future_data[:, :, i:i+1]
                projected = self.future_projection(cont_value)
                future_vars.append(projected)
            selected_future, future_weights = self.future_vsn(future_vars)
            decoder_input = selected_future
        else:
            decoder_input = selected_historical[:, -self.prediction_length:, :]

        decoded_seq, _ = self.lstm_decoder(decoder_input, (hidden, cell))
        attention_output, attention_weights = self.attention(decoded_seq, decoded_seq, decoded_seq)
        processed_output = self.post_attention_grn(attention_output)
        ff_output = self.feed_forward(processed_output)
        ff_output = self.ff_layer_norm(ff_output + processed_output)
        predictions = self.output_projection(ff_output)
        return predictions, attention_weights
