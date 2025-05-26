import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from padding import expand_matrices

def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    lst = [0] * length
    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, 2*int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return lst

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_channel_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, n)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / fmax).view(m, -1)
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(m, n), (x_amax / fmax).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def per_expert_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    num_groups, m, n = x.shape
    x_padded = torch.zeros((num_groups, ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(num_groups, m, 1, n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e5m2)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))


def gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = torch.bfloat16, dtype_a = torch.float8_e4m3fn, dtype_b = torch.float8_e4m3fn):
    ref_dw = torch.empty(M, N, device = "cuda", dtype = dtype_out)
    x = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)

    num_groups = len(tokens_per_expert)

    y = torch.randn(num_groups, N, K, device = "cuda", dtype = torch.bfloat16)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (N + 127) // 128, K // 128), device='cuda', dtype=torch.float))
    x_fp8, x_scale = per_token_cast_to_fp8(x)

    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # prepare data
    t_start = 0
    for i, tokens in enumerate(tokens_per_expert):
        tokens = int(tokens)
        x_tmp = x[t_start: t_start+tokens]; 
        weight = y[i]

        ref_dw[t_start: t_start+tokens] = (x_tmp @ weight.T)

        t_start += tokens

    # breakpoint()
    return x_fp8, x_scale, y_fp8[0], y_fp8[1], ref_dw

if __name__=='__main__':
    from typing import Tuple
    import random
    
    print('Testing grouped contiguous GEMM:')

    dtype_a = torch.float8_e4m3fn
    dtype_b = torch.float8_e4m3fn
    dtype_out = torch.bfloat16


    N_ = [3072, ]
    K_ = [8192, ]
    num_groups_ = [32,]
    M_ = [16384, ]

    cases = []
    for N in N_:
        for K in K_:
            for num_groups in num_groups_:
                for M in M_:
                    cases.append([M, N, K, num_groups])

    for row, [M, N, K, num_groups] in enumerate(cases):
        tokens_per_expert = generate_random_list(num_groups, M)
        tokens_per_expert = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.long)

        x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd = gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = dtype_out, dtype_a = dtype_a, dtype_b = dtype_b)
        size_per_group = tokens_per_expert
        
        for i in range(3):
            output_compact = deep_gemm.m_grouped_TMA_adaptive_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)
        
        def test_func_compact():
            output_compact = deep_gemm.m_grouped_TMA_adaptive_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)
        t_compact = bench_kineto(test_func_compact, 'fp8_gemm', suppress_kineto_output=True)

        
        expand_x_fp8, expand_x_scale, size_per_group_padding = expand_matrices(x_fp8, x_scale, tokens_per_expert)
        m_indices = torch.zeros(size_per_group_padding.sum(), device='cuda', dtype=torch.int)
        size_end = size_per_group_padding.cumsum(0)
        size_start = size_end - size_per_group_padding
        for i, (start, end) in enumerate(zip(size_start, size_end)):
            m_indices[start:end] = i

        output_padding = output_compact.new_empty((size_per_group_padding.sum(), N))
        for i in range(3):
            expand_x_fp8, expand_x_scale, size_per_group_padding = expand_matrices(x_fp8, x_scale, tokens_per_expert)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (expand_x_fp8, expand_x_scale), 
                (weights_fp8, weights_scale),
                output_padding, m_indices)
        def test_func_expand():
            expand_x_fp8, expand_x_scale, size_per_group_padding = expand_matrices(x_fp8, x_scale, tokens_per_expert)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (expand_x_fp8, expand_x_scale), 
                (weights_fp8, weights_scale),
                output_padding, m_indices)
        t_padding = bench_kineto(test_func_expand, 'expand_128x_kernel', suppress_kineto_output=True)
        t_padded_gemm = bench_kineto(test_func_expand, 'fp8_gemm', suppress_kineto_output=True)

        A_bytes = expand_x_fp8.numel() + x_fp8.numel()
        Ascale_bytes = 4 * (x_scale.numel() + expand_x_scale.numel())
        pad_mem_bandwidth = (A_bytes + Ascale_bytes) / 1e9 / t_padding
        acceleration = ((t_padding + t_padded_gemm) / t_compact-1) * 100
        memory_saving =  (1 - M / output_padding.shape[0]) * 100

        print(f"    Our time {t_compact * 1e3:.3f} ms , padding + grouped gemm time {(t_padding* 1e3 + t_padded_gemm* 1e3):.3f} ms")
        print(f'    > Performance: {pad_mem_bandwidth:4.0f} GB/s | Accelaration {acceleration:.3f} % | Memory saving {memory_saving:.3f} %')

        mask = output_padding.abs().sum(dim=1) != 0
        output_tmp = output_padding[mask, :]
        is_equal = torch.allclose(output_compact, output_tmp, atol = 0, rtol = 0)
        print(f"    > The result of our kernel is identical to deep gemm after removal padded data? {is_equal}")

        
        
