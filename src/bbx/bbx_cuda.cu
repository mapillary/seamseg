#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include "bbx.h"
#include "utils/cuda.cuh"

template<typename T>
__device__ inline T clamp(T x, T a, T b) {
  return max(a, min(b, x));
}

template<typename T, typename index_t>
__global__ void mask_count_kernel(const at::PackedTensorAccessor<T, 2, at::RestrictPtrTraits, index_t> bbx,
                                  const at::PackedTensorAccessor<T, 2, at::RestrictPtrTraits, index_t> int_mask,
                                  at::PackedTensorAccessor<T, 1, at::RestrictPtrTraits, index_t> count) {
  index_t num = bbx.size(0), height = int_mask.size(0), width = int_mask.size(1);
  index_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num) {
    auto _bbx = bbx[n];

    int i0 = clamp(static_cast<index_t>(_bbx[0]), index_t(0), height - 1),
        j0 = clamp(static_cast<index_t>(_bbx[1]), index_t(0), width - 1),
        i1 = clamp(static_cast<index_t>(_bbx[2]), index_t(0), height - 1),
        j1 = clamp(static_cast<index_t>(_bbx[3]), index_t(0), width - 1);

    count[n] = int_mask[i1][j1] - int_mask[i0][j1] - int_mask[i1][j0] + int_mask[i0][j0];
  }
}

at::Tensor mask_count_cuda(const at::Tensor& bbx, const at::Tensor& int_mask) {
  // Get dimensions
  auto num = bbx.size(0);

  // Create output
  auto count = at::zeros({num}, bbx.options());

  // Run kernel
  dim3 threads(getNumThreads(num));
  dim3 blocks((num + threads.x - 1) / threads.x);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(bbx.scalar_type(), "mask_count_cuda", ([&] {
    if (at::cuda::detail::canUse32BitIndexMath(int_mask)) {
      auto _bbx = bbx.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
      auto _int_mask = int_mask.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
      auto _count = count.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int32_t>();

      mask_count_kernel<scalar_t, int32_t><<<blocks, threads, 0, stream>>>(_bbx, _int_mask, _count);
    } else {
      auto _bbx = bbx.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int64_t>();
      auto _int_mask = int_mask.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int64_t>();
      auto _count = count.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, int64_t>();

      mask_count_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(_bbx, _int_mask, _count);
    }
  }));

  return count;
}