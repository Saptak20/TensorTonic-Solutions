#include <cuda_runtime.h>
#include <float.h>
#include <climits>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float shared_vals[256];
    __shared__ int shared_idxs[256];

    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + tid;

    float my_value = -INFINITY;
    int my_index = INT_MAX;
    if (global_index < N) {
        my_value = input[global_index];
        my_index = global_index;
    }

    shared_vals[tid] = my_value;
    shared_idxs[tid] = my_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = shared_vals[tid + stride];
            int other_index = shared_idxs[tid + stride];
            bool other_is_better = other_value > shared_vals[tid] ||
                                   (other_value == shared_vals[tid] && other_index < shared_idxs[tid]);
            if (other_is_better) {
                shared_vals[tid] = other_value;
                shared_idxs[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = shared_vals[0];
        block_idxs[blockIdx.x] = shared_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float shared_vals[256];
    __shared__ int shared_idxs[256];

    int tid = threadIdx.x;

    float best_value = -INFINITY;
    int best_index = INT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float candidate_value = block_vals[i];
        int candidate_index = block_idxs[i];
        bool candidate_is_better = candidate_value > best_value ||
                                   (candidate_value == best_value && candidate_index < best_index);
        if (candidate_is_better) {
            best_value = candidate_value;
            best_index = candidate_index;
        }
    }

    shared_vals[tid] = best_value;
    shared_idxs[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = shared_vals[tid + stride];
            int other_index = shared_idxs[tid + stride];
            bool other_is_better = other_value > shared_vals[tid] ||
                                   (other_value == shared_vals[tid] && other_index < shared_idxs[tid]);
            if (other_is_better) {
                shared_vals[tid] = other_value;
                shared_idxs[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}