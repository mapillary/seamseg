#include <ATen/ATen.h>

#include "bbx.h"

template<typename T>
inline T clamp(T x, T a, T b) {
  return std::max(a, std::min(b, x));
}

at::Tensor mask_count_cpu(const at::Tensor& bbx, const at::Tensor& int_mask) {
  // Get dimensions
  auto num = bbx.size(0), height = int_mask.size(0), width = int_mask.size(1);

  // Create output
  auto count = at::zeros({num}, bbx.options());

  AT_DISPATCH_FLOATING_TYPES(bbx.scalar_type(), "mask_count_cpu", ([&] {
    auto _bbx = bbx.accessor<scalar_t, 2>();
    auto _int_mask = int_mask.accessor<scalar_t, 2>();
    auto _count = count.accessor<scalar_t, 1>();

    for (int64_t n = 0; n < num; ++n) {
      auto i0 = clamp(static_cast<int64_t>(_bbx[n][0]), int64_t(0), int64_t(height - 1)),
           j0 = clamp(static_cast<int64_t>(_bbx[n][1]), int64_t(0), int64_t(width - 1)),
           i1 = clamp(static_cast<int64_t>(_bbx[n][2]), int64_t(0), int64_t(height - 1)),
           j1 = clamp(static_cast<int64_t>(_bbx[n][3]), int64_t(0), int64_t(width - 1));

      _count[n] = _int_mask[i1][j1] - _int_mask[i0][j1] - _int_mask[i1][j0] + _int_mask[i0][j0];
    }
  }));

  return count;
}
