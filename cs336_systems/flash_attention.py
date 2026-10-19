import torch
import triton
import numpy as np
import triton.language as tl

QUERY_TILE_SIZE = 16
KEY_TILE_SIZE = 32 

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_casual: bool = False) -> torch.Tensor:
        D = Q.shape[-1]

        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        batch_size = Q.shape[0]

        print(n_keys, KEY_TILE_SIZE)

        assert Q.shape[0] == K.shape[0], "Batch size not equal"

        output = torch.zeros(Q.shape, dtype=torch.float32, device=Q.device)
        logsumexp = torch.zeros((batch_size, n_queries), dtype=torch.float32, device=Q.device)

        scale = np.sqrt(D)
        # ctx.D = D,
        # ctx.Q_TILE_SIZE = QUERY_TILE_SIZE
        # ctx.K_TILE_SIZE = KEY_TILE_SIZE

        ctx.save_for_backward(Q, V, K)
        ctx.is_casual = is_casual

        print(batch_size, n_queries // QUERY_TILE_SIZE)
        flash_fwd_kernel[(n_queries // QUERY_TILE_SIZE, batch_size)](
            Q, K, V, 
            output, logsumexp,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            n_queries, n_keys,
            scale,
            # ctx.D,
            # ctx.QUERY_TILE_SIZE,
            # ctx.KEY_TILE_SIZE
            D,
            QUERY_TILE_SIZE,
            KEY_TILE_SIZE,
            is_casual
        )

        
        ctx.save_for_backward(output, logsumexp)
        return output

    def backward(ctx, x, weight):
        pass

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_casual: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    Q_block_ptr = tl.make_block_ptr(
       Q_ptr + batch_index * stride_qb,
       shape=(N_QUERIES, D),
       strides=(stride_qq, stride_qd),
       offsets=(query_tile_index * Q_TILE_SIZE, 0),
       block_shape=(Q_TILE_SIZE, D),
       order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
       O_ptr + batch_index * stride_ob,
       shape=(N_QUERIES, D),
       strides=(stride_oq, stride_od),
       offsets=(query_tile_index * Q_TILE_SIZE, 0),
       block_shape=(Q_TILE_SIZE, D),
       order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
       L_ptr + batch_index * stride_lb,
       shape=(N_QUERIES,),
       strides=(stride_lq,),
       offsets=(query_tile_index * Q_TILE_SIZE,),
       block_shape=(Q_TILE_SIZE,),
       order=(0, ),
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    logexpsum = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    max = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        K = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")


        attention_scores = tl.dot(Q, K.T) / scale # Bq X Bk 16 x 32

        if is_casual:
            query_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_index = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = query_index[:, None] < key_index[None, :]
            mask *= -1e6
            # tl.device_print("Mat", mask)
            attention_scores += mask

        new_max = tl.maximum(max, tl.max(attention_scores, axis=1)) # Bq 16

        # probabilities = tl.exp(attention_scores - new_max[:, None]) # Bq X Bk
        mid = attention_scores - new_max[:, None] # Bq X Bk
        probabilities = tl.exp(mid)
        # tl.device_print("pro:", probabilities)
        logexpsum = tl.exp(max - new_max) * logexpsum + tl.sum(probabilities, 1) # Bq
        output = tl.exp(max - new_max)[:, None] * output + tl.dot(probabilities, V)

        max = new_max
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    output = output / logexpsum[:, None] 
    logexpsum = max + tl.log(logexpsum)

    tl.store(O_block_ptr, output, boundary_check=(0,1))
    tl.store(L_block_ptr, logexpsum, boundary_check=(0,))

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dim = 32
    batch_size = 1
    Q = torch.eye(dim)
    Q = Q.unsqueeze(0).expand(batch_size, -1, -1)
    K = torch.eye(dim)
    K = K.unsqueeze(0).expand(batch_size, -1, -1)
    V = torch.eye(dim)
    V = V.unsqueeze(0).expand(batch_size, -1, -1)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    output = FlashAttention.forward(FlashAttention, Q, K, V, True)
    print(output)

if __name__ == "__main__":
    main()