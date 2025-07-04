#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;

    // For grouped GEMM
    int* m_indices_pad;
    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    uint64_t m_pad;

    __device__ __forceinline__ explicit Scheduler(const uint64_t m_pad_,
                                                  int* m_indices_pad = nullptr) {
        m_pad = m_pad_;
        num_aligned_m_blocks = uint32_t(m_pad / BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->m_indices_pad = m_indices_pad;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->m_indices_pad = m_indices_pad;
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
        auto in_group_idx = block_idx % num_blocks_per_group;
        m_block_idx = in_group_idx / num_n_blocks_in_group;
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }


    __device__ __forceinline__ uint64_t get_off_am(uint32_t shape_m, uint32_t m_block_idx, const uint64_t* group_pad_offs,
                        const uint64_t* token_cumdiff){
        
        uint64_t off_am;
        uint32_t group = __ldg(m_indices_pad + m_block_idx);
        off_am = m_block_idx * BLOCK_M - token_cumdiff[group];
        // off_am = (m_block_idx == (m_pad / BLOCK_M - 1)) ? (shape_m - BLOCK_M) : (m_block_idx * BLOCK_M - token_cumdiff[group]);

        if (m_block_idx == (m_pad / BLOCK_M - 1)) {
            off_am = shape_m - BLOCK_M;
            // if (threadIdx.x == 0) printf("enter if\n");
        }
        // else{
        //     uint32_t group = __ldg(m_indices_pad + m_block_idx);
        //     off_am = m_block_idx * BLOCK_M - token_cumdiff[group];
        // }
        return off_am;
    }

    __device__ __forceinline__ void get_cm_info(uint64_t& off_cm, uint32_t& cm_left, uint32_t shape_m, uint32_t m_block_idx,
                        const uint64_t* token_cumdiff, const uint64_t* token_pad_end){

        uint32_t group = __ldg(m_indices_pad + m_block_idx);
        off_cm = m_block_idx * BLOCK_M - token_cumdiff[group];
        cm_left = __ldg(token_pad_end + group) - off_cm;
        // printf("group %d, cm_left %d \n", group, cm_left);
        return;
    }



    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(m_indices_pad + m_block_idx);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(m_indices_pad + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};
#pragma clang diagnostic pop

} // namespace deep_gemm
