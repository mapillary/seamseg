#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include "nms.h"
#include "utils/common.h"
#include "utils/cuda.cuh"

template<typename T> struct VectType;
template<> struct VectType<float> {
  typedef float4 value;
  typedef float4* ptr;
  typedef const float4* const_ptr;
};
template<> struct VectType<double> {
  typedef double4 value;
  typedef double4* ptr;
  typedef const double4* const_ptr;
};

template<typename T>
__device__ inline T area(T tl0, T tl1, T br0, T br1) {
  return max(br0 - tl0, T(0)) * max(br1 - tl1, T(0));
}

template<typename T>
__device__ inline T iou(typename VectType<T>::value bbx0, typename VectType<T>::value bbx1) {
  auto ptl0 = max(bbx0.x, bbx1.x);
  auto ptl1 = max(bbx0.y, bbx1.y);
  auto pbr0 = min(bbx0.z, bbx1.z);
  auto pbr1 = min(bbx0.w, bbx1.w);
  auto intersection = area(ptl0, ptl1, pbr0, pbr1);
  auto area0 = area(bbx0.x, bbx0.y, bbx0.z, bbx0.w);
  auto area1 = area(bbx1.x, bbx1.y, bbx1.z, bbx1.w);
  return intersection / (area0 + area1 - intersection);
}

template<typename T>
__global__ void comp_mat_kernel(const int64_t num, const int64_t blocks, const float threshold,
                                const T* __restrict__ bbx, int64_t* __restrict__ comp_mat) {
  // Find position in grid
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int row_size = min(num - row_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const int col_size = min(num - col_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  auto _bbx = reinterpret_cast<typename VectType<T>::const_ptr>(bbx);

  // Load data to block storage
  __shared__ typename VectType<T>::value block_bbx[THREADS_PER_BLOCK];
  if (threadIdx.x < col_size) {
    block_bbx[threadIdx.x] = _bbx[THREADS_PER_BLOCK * col_start + threadIdx.x];
  }
  __syncthreads();

  // Perform actual computation
  if (threadIdx.x < row_size) {
    const int cur_box_idx = THREADS_PER_BLOCK * row_start + threadIdx.x;
    const auto cur_box = _bbx[cur_box_idx];

    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }

    int64_t t = 0;
    for (int i = start; i < col_size; ++i) {
      if (iou<T>(cur_box, block_bbx[i]) >= threshold) {
        t |= int64_t(1) << i;
      }
    }
    comp_mat[cur_box_idx * blocks + col_start] = t;
  }
}

at::Tensor comp_mat_cuda(const at::Tensor& bbx, float threshold) {
  int64_t num = bbx.size(0);
  int64_t blocks = div_up(num, THREADS_PER_BLOCK);

  auto comp_mat = at::zeros({num, blocks}, bbx.options().dtype(at::kLong));

  dim3 blk(blocks, blocks, 1);
  dim3 thd(THREADS_PER_BLOCK, 1, 1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(bbx.scalar_type(), "comp_mat_cuda", ([&] {
    comp_mat_kernel<scalar_t><<<blk, thd, 0, stream>>>(
        num, blocks, threshold, bbx.data<scalar_t>(), comp_mat.data<int64_t>());
  }));

  return comp_mat;
}
