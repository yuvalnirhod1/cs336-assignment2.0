import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum, reduce, repeat

def normal_init(t: torch.Tensor, sigma_squred=1):
    sigma = np.sqrt(sigma_squred)
    torch.nn.init.trunc_normal_(t, 0, sigma_squred, -3 * sigma, 3 * sigma)

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weights = torch.zeros(out_features, in_features, device=device, dtype=dtype)

        sigma = 2 / (in_features + out_features)
        normal_init(weights, sigma)

        self.w = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")

class Embeddings(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embeddings = torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype)
        normal_init(embeddings, sigma_squred=1)

        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot_vectors = F.one_hot(token_ids, num_classes=self.embeddings.shape[0]).to(self.embeddings.dtype)
        return einsum(self.embeddings, one_hot_vectors, "num_embeddings embeddings_dim, ... num_embeddings -> ... embeddings_dim")

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _RMS(self, x: torch.Tensor):
        return rearrange(torch.sqrt((reduce(torch.square(x), "... n -> ...", "sum") / self.scale.shape[0]) + self.eps), "... batch -> ... batch 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        div = self._RMS(x) 
        mat_mul = einsum(x, self.scale, "... hidden_dim, hidden_dim-> ... hidden_dim")
        return (mat_mul / div).to(in_type)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(np.round(((8 / 3) * d_model) / 64.0) * 64)

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, torch.sigmoid(x), "... dim, ... dim -> ... dim")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_value = self._SiLU(self.W1(x))
        gated_value = einsum(silu_value, self.W3(x), "... dim, ... dim -> ... dim")
        return self.W2(gated_value)

class RoPE(torch.nn.Module):
    _instance = None

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # Build base theta matrix
        k = torch.arange(1, d_k // 2 + 1, device=device)
        i = rearrange(torch.arange(max_seq_len, device=device), "a -> a 1")
        div = rearrange(torch.pow(theta, (2 * k  - 2) / d_k), "a -> 1 a")
        theta_mat = i / div
        self.device = device

        # Build cos(theta_i) matrix
        cos = torch.cos(theta_mat)

        # Build sin(theta_i) matrix
        sin = torch.sin(theta_mat)

        # Combine it to cos -sin sin cos
        rotation_matrix = rearrange(torch.stack([cos, -sin, sin, cos]), "stacked max_seq hidden_dim -> max_seq (hidden_dim stacked)")
        self.register_buffer("rotation_matrix", rotation_matrix)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _rotate(self, x: torch.Tensor, rotation_matrix: torch.tensor) -> torch.Tensor:
        x = repeat(x, "... i -> ... (i 2)")
        x = rearrange(x, "... (i b c) -> ... (i c b)", b=2, c=2)
        mul = einsum(x, rotation_matrix, "... i, ... i -> ... i")
        return reduce(mul, " ... (i 2) -> ... i", "sum")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotation_matrix = self.rotation_matrix[token_positions]
        return self._rotate(x, rotation_matrix)

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x = x - torch.amax(x, dim=i, keepdim=True)
    exp = torch.exp(x)
    summation = torch.sum(exp, dim=i, keepdim=True)
    return exp / summation

def dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = k.shape[-1]
    d_v = v.shape[-1]

    query_keys = einsum(q, k, "... query_seq_len key_dim, ... key_seq_len key_dim -> ... query_seq_len key_seq_len") / np.sqrt(d_k)

    if mask is not None:
        query_keys = query_keys + torch.where(mask, 0.0, -torch.inf).to(query_keys.device)

    matches = softmax(query_keys, -1)
    return einsum(matches, v, "... query_seq_len target_seq_len, ... target_seq_len value_dim -> ... query_seq_len value_dim")

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta = None, max_seq_len = None, d_keys: int = None, d_values: int = None, device=None, dtype=None):
        super().__init__()

        if d_keys is None:
            d_keys = d_model // num_heads 
        if d_values is None:
            d_values = d_model // num_heads

        self.num_heads = num_heads

        self.WQ = Linear(d_model, num_heads * d_keys, device=device, dtype=dtype)
        self.WK = Linear(d_model, num_heads * d_keys, device=device, dtype=dtype)
        self.WV = Linear(d_model, num_heads * d_values, device=device, dtype=dtype)
        self.WO = Linear(num_heads * d_values, d_model, device=device, dtype=dtype)

        if theta and max_seq_len:
            self.rope = RoPE(theta=theta, d_k=d_keys, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        q = rearrange(self.WQ(x), "... head_index (head_num hidden_dim) -> ... head_num head_index hidden_dim", head_num=self.num_heads)
        k = rearrange(self.WK(x), "... head_index (head_num hidden_dim) -> ... head_num head_index hidden_dim", head_num=self.num_heads)
        v = rearrange(self.WV(x), "... head_index (head_num hidden_dim) -> ... head_num head_index hidden_dim", head_num=self.num_heads)

        mask = torch.ones((q.shape[-2], v.shape[-2]))
        mask = torch.triu(mask, diagonal=1) == 0

        if hasattr(self, "rope"):
            assert token_positions is not None, "If you want rope you must supply token positions!"
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attention_output = dot_product_attention(q, k, v, mask=mask)
        return self.WO(rearrange(attention_output, "... head_num head_index hidden_dim -> ... head_index (head_num hidden_dim)", head_num=self.num_heads))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.first_rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)

        self.second_rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_block = x + self.attention(self.first_rmsnorm(x), token_positions=torch.arange(0, x.shape[-2]))
        return first_block + self.swiglu(self.second_rmsnorm(first_block))

class TransformerLM(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, vocab_size: int, context_length: int, num_layers: int,
                 d_ff: int = None, theta=None, device=None, dtype=None):
        super().__init__()
        attentions = [TransformerBlock(d_model, num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length,
                                                  device=device, dtype=dtype) for _ in range(num_layers)]
        self.attentions = torch.nn.Sequential(*attentions)

        self.embeddings = Embeddings(vocab_size, d_model, device=device, dtype=dtype)

        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.w_final = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        return self.w_final(self.norm_final(self.attentions(self.embeddings(x))))

def main():
    hidden_dims = 64 
    batch_size = 2
    seq_length = 4 

    transformer_block = TransformerBlock(hidden_dims, 4)
    x = torch.randn(batch_size, seq_length, hidden_dims)
    y = transformer_block(x)
    print(y)

if __name__ == "__main__":
    main()