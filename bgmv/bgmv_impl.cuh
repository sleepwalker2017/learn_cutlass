#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>

#include "vec_dtypes.cuh"

namespace cg = cooperative_groups;

// nthrs = (32, 4)
// grid (16, 1)
template <int feat_in, int feat_out, typename T>
__global__ void bgmv_shrink_kernel(T* __restrict__ Y, const T* __restrict__ X,
                                   const T* __restrict__ W,
                                   const int64_t* __restrict__ indicies,
                                   int64_t num_layers, int64_t layer_idx,
                                   float scale) {
  auto block = cg::this_thread_block();
  size_t j = blockIdx.x;
  size_t batch_idx = blockIdx.y; // 每16个 block处理一个 batch, 每个 block 处理输出的一个元素
  constexpr size_t vec_size = 16 / sizeof(T); // vec_size = 8
  constexpr size_t tx = 32;
  constexpr size_t ty = 4;
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size; // 每个 block 能处理的元素个数  32 * 4 * 8=1024 个元素

  __shared__ T W_shared[num_pipeline_stages * tile_size];
  __shared__ T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  // pipeline load W/X and compute WX;

  // idx = 0
  // j = [0..15]
  // feat_in = 256
  // feat_out = 16
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W + (idx * feat_out + j) * feat_in +   // 每个 block 处理256(feat_in)个数据
                         (threadIdx.y * tx + threadIdx.x) * vec_size, // 每个线程偏移 2 个元素
                     cuda::aligned_size_t<16>(16), pipe);
  cuda::memcpy_async(
      X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
      X + (batch_idx * feat_in) + (threadIdx.y * tx + threadIdx.x) * vec_size, //每16个 block 都处理一样的数据,每个线程偏移 2 个元素
      cuda::aligned_size_t<16>(16), pipe);
  pipe.producer_commit();
  size_t copy_idx, compute_idx;
  float y = 0.f;
  flashinfer::vec_t<T, vec_size> x_vec, w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size;
       ++tile_idx) {
#if 0
	  if(threadIdx.x == 0 and blockIdx.x ==0) {
		printf("tile_size %d, tile_idx %d\n", tile_size, tile_idx);
	  }
#endif
    copy_idx = tile_idx % num_pipeline_stages;
    // pipeline stage: async copy W fragment
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size, // 每个线程偏移 
                         W + (idx * feat_out + j) * feat_in + // tile_idx =0
                             tile_idx * tile_size +    // tile_idx = 0
                             (threadIdx.y * tx + threadIdx.x) * vec_size, // 每个线程偏移
                         cuda::aligned_size_t<16>(16), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size, // 每个线程偏移
                         X + (batch_idx * feat_in) + tile_idx * tile_size + // 
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<16>(16), pipe);
    }
    pipe.producer_commit();

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // pipeline stage: compute WX
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] = sum; 
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) {
      y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  // final pipeline stage
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  y_warpwise[threadIdx.y] =
      ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
          ? sum
          : 0.f;
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y += y_warpwise[i];
  }

  block.sync();
  pipe.consumer_release();

  // write Y;
  if (block.thread_rank() == 0) {
    Y[batch_idx * feat_out + j] += y;
  }
}

// nthrs = (2, 16, 4)
/*
feat_in = 8, feat_out = 128, T=fp16

 sizeof(T) = 2

x : [1, 8]
w : [8, 128]
*/

// (1, 32, 4)    (1, 1)
template <int feat_in, int feat_out, typename T>
__global__ void bgmv_expand_kernel(T* __restrict__ Y, const T* __restrict__ X,
                                   const T* __restrict__ W,
                                   const int64_t* __restrict__ indicies,
                                   int64_t num_layers, int64_t layer_idx,
                                   float scale) {
  auto block = cg::this_thread_block();
  constexpr size_t vec_size = 16 / sizeof(T);  // vec_size = 8, 也就是每个向量包含 8 个 fp16
  constexpr size_t tx = feat_in / vec_size;    // tx = 8/8 = 1, 
  static_assert(feat_in % vec_size == 0);
  constexpr size_t ty = 32 / tx; 	       // ty = 32 32 个线程一组，一组处理多少数据？
  static_assert(32 % tx == 0);
  constexpr size_t tz = 4; 		       // tz = 4, 也就是一个 block 固定 128 个线程？
  size_t tile_idx = blockIdx.x;			// 每128个 线程 处理一个啥？ // 没理解 tile 是啥意思。
  size_t batch_idx = blockIdx.y;		// 每多少个 block 处理一个 batch ?
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx; // 0

  // load X;  这是加载到寄存器啊？
  flashinfer::vec_t<T, vec_size> x_vec;
  x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size); // 每个线程处理 8 个

  // load W;
  flashinfer::vec_t<T, vec_size> w_vec; // 这是8个 fp16 数据 依然是 8 个

  //				0 
  int tmp = block.thread_rank(); // 每个线程加载 8 个数据，一样的,

  // tile_idx 说明，每个 block 负责生成 feat_in*ty*tz 这些行的数据？

  // tile_idx * tz * ty * feat_in + thread_rank * vec_size
  w_vec.load(W + (idx * feat_out + tile_idx * tz * ty) * feat_in +
             block.thread_rank() * vec_size);

  float sum = 0.f;

  // 每个线程加载 8 个x， 8 个 w，
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
  //  tx个线程来处理一行，这里是分块的，所以需要进行一个规约。
  cg::thread_block_tile g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += g.shfl_down(sum, offset);
  }
  // 规约之后，得到真正的和
  sum = g.shfl(sum, 0);

  // 和真正的增加
  if (threadIdx.x == 0) {
    Y[batch_idx * feat_out + tile_idx * (tz * ty) + threadIdx.z * ty +
      threadIdx.y] += sum;
  }
}

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W, const int64_t* __restrict__ indicies,
                 int64_t batch_size, int64_t num_layers, int64_t layer_idx,
                 float scale) {
  size_t vec_size = 16 / sizeof(T);
  if constexpr (feat_in < feat_out) {
    size_t tx = feat_in / vec_size;
    size_t ty = 32 / tx;
    size_t tz = 4;
    dim3 nblks(feat_out / (ty * tz), batch_size);
    dim3 nthrs(tx, ty, tz);
    //printf("feat_in %d feat_out %d, tx %d, ty %d, tz %d, block.x %d, block.y %d\n", feat_in, feat_out, tx, ty, tz, feat_out / (ty * tz), batch_size);
    bgmv_expand_kernel<feat_in, feat_out>
        <<<nblks, nthrs>>>(Y, X, W, indicies, num_layers, layer_idx, scale);
  } else {
    assert(feat_in % (vec_size * 32) == 0);
    dim3 nblks(feat_out, batch_size); // 16, 1
    dim3 nthrs(32, 4);
    //printf("feat_in %d feat_out %d\n", feat_in, feat_out);
    bgmv_shrink_kernel<feat_in, feat_out>
        <<<nblks, nthrs>>>(Y, X, W, indicies, num_layers, layer_idx, scale);
  }
}

#define INST_BGMV(feat_in, feat_out, T)                                    \
  template void bgmv_kernel<feat_in, feat_out>(                            \
      T* __restrict__ Y, const T* __restrict__ X, const T* __restrict__ W, \
      const int64_t* __restrict__ indicies, int64_t batch_size,            \
      int64_t num_layers, int64_t layer_idx, float scale);

#define INST_BGMV_TWOSIDE(T, narrow, wide) \
  INST_BGMV(narrow, wide, T)               \
  INST_BGMV(wide, narrow, T)

