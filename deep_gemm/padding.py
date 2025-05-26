# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import random

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch.profiler import ProfilerActivity, profile

@triton.jit
def get_group_id(m, group_offsets, g_start, num_experts):
    id = 0
    off_out = 0
    offnxt_out = 0
    for group_id in tl.range(g_start, num_experts):
        group_off = tl.load(group_offsets + group_id)
        group_off_nxt = tl.load(group_offsets + group_id  + 1)
        if m >=  group_off and m < group_off_nxt:
            id = group_id
            off_out = group_off
            offnxt_out = group_off_nxt
    return id, off_out, offnxt_out

@triton.jit
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.jit
def expand_128x_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    output_scales_ptr,
    group_pad_offs,
    token_cumdiffs,
    token_ends,
    num_experts: tl.constexpr,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    M_EXPAND,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dequant+transpose+quant kernel."""

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_tiles_pad = tl.cdiv(M_EXPAND, BLOCK_M) * num_pid_n

    for tile_id in tl.range(start_pid+num_tiles, num_tiles_pad, BLOCKS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_an = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)
        offs_bm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        offs_bn = offs_an

        output_offset = (
            output_ptr
            + offs_bm[:, None] * N
            + offs_bn[None, :]
            
        )

        output_scale_offset = (
            output_scales_ptr
            + offs_bm[:, None] * ((N + 127) // 128)
            + pid_n
        )

        out = tl.zeros((BLOCK_M, BLOCK_M), dtype = output_ptr.dtype.element_ty)
        tl.store(output_offset, out)
        tl.store(output_scale_offset, 0)


    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        group, _, _ = get_group_id(pid_m * BLOCK_M, group_pad_offs, 0, num_experts)
        token_cumdiff = tl.load(token_cumdiffs + group)
        token_end = tl.load(token_ends + group)

        offs_am = (pid_m * BLOCK_M - token_cumdiff + tl.arange(0, BLOCK_M))
        offs_an = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)


        offs_bm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        offs_bn = offs_an

        input_offset = (
            input_ptr
            + offs_am[:, None] * N
            + offs_an[None, :] * 1
        )

        scale_offset = (
            scale_ptr
            + offs_am[:, None] * ((N + 127) // 128)
            + pid_n
        )

        output_offset = (
            output_ptr
            + offs_bm[:, None] * N
            + offs_bn[None, :]
            
        )

        output_scale_offset = (
            output_scales_ptr
            + offs_bm[:, None] * ((N + 127) // 128)
            + pid_n
        )

        mask_input = (offs_am[:, None] < token_end) & (offs_an < N)
        mask_scale = (offs_am[:, None] < token_end) 

        input_block = tl.load(
            input_offset, mask=mask_input, other=0.0
        )
        output_block_scale = tl.load(
            scale_offset, mask=mask_scale, other=0.0
        )

        tl.store(output_offset, input_block)
        tl.store(output_scale_offset, output_block_scale)


def expand_matrices(
    input_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequant a per-group per-tile quantized tensor. Do per-block quantization
    along another dimension. And finally transpose it.

    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N]. It is
            per-tile quantized along N dim.
        input_scales (Tensor): The input scale. Shape [M, group_num_n].
            group_num_n=N//group_size.
        size_per_group (Tensor): The seq length of each expert. The sum of it
            should be equal to M.
        group_size (int): The group size of the quantization. Default to 128.

    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-tile
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, group_num_m].
            group_num_m=(M_EXPAND+group_size-1)//group_size.
    """
    M, N = input_tensor.shape # (302873, 7168)
    assert N % group_size == 0
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    group_pad_off = torch.zeros(size_per_group.shape[0] + 1, device = "cuda", dtype = torch.int32)
    
    BLOCK_M = group_size
    size_per_group_padding = triton.cdiv(size_per_group, BLOCK_M) * BLOCK_M
    group_pad_off[1:] = size_per_group_padding.cumsum(0)

    num_experts = size_per_group.shape[0]
    
    M_EXPAND = M + group_size * num_experts - M % group_size
    output_tensor = input_tensor.new_empty((M_EXPAND, N), dtype=dtype)
    output_scales = scale_tensor.new_empty(
        (M_EXPAND, (N + 127) // 128))

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS, )

    M_pad = size_per_group_padding.sum()

    token_diff = size_per_group_padding - size_per_group
    token_cumdiff = token_diff.cumsum(0)
    token_pad_end = size_per_group_padding.cumsum(0) - token_cumdiff
    token_end = size_per_group.cumsum(0) 

    token_cumdiff = token_diff.cumsum(0) - token_diff

    wrap_triton(expand_128x_kernel)[grid](
        input_tensor,
        scale_tensor,
        output_tensor,
        output_scales,
        group_pad_off,
        token_cumdiff,
        token_end,
        num_experts,
        M_pad,
        M,
        N,
        M_EXPAND,
        fmax=fmax,
        fmin=fmin,
        BLOCK_M=group_size,
        BLOCK_N=group_size,
    )
    
    return output_tensor[:M_pad, :], output_scales[:M_pad, :], size_per_group_padding

if __name__ == "__main__":
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

    def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

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

        print('Testing grouped contiguous GEMM:')

    dtype_a = torch.float8_e4m3fn
    dtype_b = torch.float8_e4m3fn

    dtype_out = torch.bfloat16

    N, K = (8192, 3072)
    tokens_per_expert = generate_random_list(4, 65536)
    tokens_per_expert = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.long)
    num_groups = len(tokens_per_expert)
    M = sum(tokens_per_expert)

    x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd = gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = dtype_out, dtype_a = dtype_a, dtype_b = dtype_b)
    for i in range(3):
        expand_x_fp8, expand_x_scale, size_per_group_padding = expand_matrices(x_fp8, x_scale, tokens_per_expert)
    
    from deep_gemm import bench_kineto
    def test_func_expand():
            expand_x_fp8, expand_x_scale, size_per_group_padding = expand_matrices(x_fp8, x_scale, tokens_per_expert)

    t_padding = bench_kineto(test_func_expand, 'expand_128x_kernel', suppress_kineto_output=True)

    A_bytes = expand_x_fp8.numel() + x_fp8.numel()
    Ascale_bytes = 4 * (x_scale.numel() + expand_x_scale.numel())

    print(f' > Performance: {t_padding * 1e6:4.0f} us | '
              f'{(A_bytes + Ascale_bytes) / 1e9 / t_padding:4.0f} GB/s')