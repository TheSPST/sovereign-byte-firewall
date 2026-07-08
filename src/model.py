import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

class CausalSelfAttention(nn.Module):
    """
    Highly optimized Causal Multi-Head Self-Attention layer using
    PyTorch's native F.scaled_dot_product_attention.
    """
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Project and split into Query, Key, Value
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3 * C)
        q = qkv[:, :, :C]
        k = qkv[:, :, C:2*C]
        v = qkv[:, :, 2*C:]
        
        # Reshape to (B, nhead, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Use native scaled dot-product attention which optimizes CUDA and MPS execution.
        # We specify is_causal=True to enforce the causal mask.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )
        
        # Transpose and concatenate back to (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm GPT-style Transformer Block combining optimized causal
    self-attention and a multi-layer perceptron (MLP).
    """
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-LN residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NetworkBytePatcher(nn.Module):
    """
    Foundational, encoder-free byte-level anomaly patcher model.
    Defaults to an ultra-lightweight configuration for <1ms packet-chunk inference.
    Incorporates positional embeddings to resolve protocol offsets.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, max_patch_size=64, max_sequence_length=8192, dropout=0.0):
        super().__init__()
        self.vocab_size = 256  # Byte value range 0x00 - 0xFF
        self.max_patch_size = max_patch_size
        self.max_sequence_length = max_sequence_length
        
        # Token and Positional Embeddings
        self.byte_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_sequence_length, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, self.vocab_size)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.max_sequence_length, f"Input sequence length {T} exceeds max_sequence_length {self.max_sequence_length}"
        
        # Create positional indices
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # Shape: (1, T)
        
        # Embed bytes and add position information
        x_emb = self.byte_embedding(x)  # Shape: (B, T, d_model)
        x_pos = self.pos_embedding(positions)  # Shape: (1, T, d_model)
        
        h = self.dropout(x_emb + x_pos)
        
        # Pass through causal transformer layers
        for block in self.blocks:
            h = block(h)
            
        h = self.ln_f(h)
        return self.predictor(h)  # Shape: (B, T, vocab_size)

    def compute_entropy(self, logits):
        """
        Calculates Shannon Entropy across logits.
        Formula: H_t = -sum(P(x_i) * log2(P(x_i)))
        """
        probs = F.softmax(logits, dim=-1)
        # Avoid log(0) with adding epsilon (1e-9)
        return -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)

    def generate_patch_lengths(self, x, entropy_threshold=5.0):
        """
        Dynamically groups byte sequences into patches based on entropy threshold spikes,
        enforcing a strict max_patch_size ceiling.
        """
        self.eval()
        device_type = x.device.type
        
        # Setup device-appropriate mixed precision context
        if device_type == 'cuda':
            autocast_context = torch.amp.autocast(device_type='cuda')
        elif device_type == 'cpu':
            autocast_context = torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)
        else:
            autocast_context = nullcontext()

        with torch.no_grad():
            with autocast_context:
                logits = self.forward(x)
                entropies = self.compute_entropy(logits)
          
        batch_patch_lengths = []
        for b in range(x.size(0)):
            current_patch_len = 0
            lengths = []
            for t in range(x.size(1)):
                current_patch_len += 1
                # Trigger boundary if entropy spikes OR if we reach the hard ceiling (max_patch_size)
                if entropies[b, t] > entropy_threshold or current_patch_len >= self.max_patch_size:
                    lengths.append(current_patch_len)
                    current_patch_len = 0
            if current_patch_len > 0:
                lengths.append(current_patch_len)
            batch_patch_lengths.append(lengths)
        return batch_patch_lengths
