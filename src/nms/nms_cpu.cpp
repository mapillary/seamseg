#include <cmath>
#include <list>
#include <vector>

#include <ATen/ATen.h>

#include "nms.h"
#include "utils/common.h"

template<typename T>
inline T area(T tl0, T tl1, T br0, T br1) {
  return std::max(br0 - tl0, T(0)) * std::max(br1 - tl1, T(0));
}

template<typename T>
inline T iou(at::TensorAccessor<T, 1> &bbx0, at::TensorAccessor<T, 1> &bbx1) {
  auto ptl0 = std::max(bbx0[0], bbx1[0]);
  auto ptl1 = std::max(bbx0[1], bbx1[1]);
  auto pbr0 = std::min(bbx0[2], bbx1[2]);
  auto pbr1 = std::min(bbx0[3], bbx1[3]);
  auto intersection = area(ptl0, ptl1, pbr0, pbr1);
  auto area0 = area(bbx0[0], bbx0[1], bbx0[2], bbx0[3]);
  auto area1 = area(bbx1[0], bbx1[1], bbx1[2], bbx1[3]);
  return intersection / (area0 + area1 - intersection);
}

at::Tensor comp_mat_cpu(const at::Tensor& bbx, float threshold) {
  int64_t num = bbx.size(0);
  int64_t blocks = div_up(num, THREADS_PER_BLOCK);

  auto comp_mat = at::zeros({num, blocks}, bbx.options().dtype(at::ScalarType::Long));

  AT_DISPATCH_FLOATING_TYPES(bbx.scalar_type(), "comp_mat_cpu", ([&] {
    auto _bbx = bbx.accessor<scalar_t, 2>();
    auto _comp_mat = comp_mat.accessor<int64_t, 2>();

    for (int64_t i = 0; i < num; ++i) {
      auto _bbx_i = _bbx[i];
      auto _comp_mat_i = _comp_mat[i];

      for (int64_t j = i + 1; j < num; ++j) {
        auto _bbx_j = _bbx[j];
        auto iou_ij = iou<scalar_t>(_bbx_i, _bbx_j);

        if (iou_ij >= threshold) {
          int64_t block_idx = j / THREADS_PER_BLOCK;
          int64_t bit_idx = j % THREADS_PER_BLOCK;

          _comp_mat_i[block_idx] |= int64_t(1) << bit_idx;
        }
      }
    }
  }));

  return comp_mat;
}

at::Tensor nms_cpu(const at::Tensor& comp_mat, const at::Tensor& idx, int n_max) {
  int64_t num = comp_mat.size(0);

  auto _comp_mat = comp_mat.accessor<int64_t, 2>();
  auto _idx = idx.data<int64_t>();

  // Copy to C++ data structures
  std::list<int64_t> candidates;
  std::copy(_idx, _idx + num, std::back_inserter(candidates));

  std::vector<int64_t> selection;
  size_t n_max_ = n_max > 0 ? n_max : num;

  // Run actual nms
  while (!candidates.empty() && selection.size() < n_max_) {
    // Select first element
    auto i = candidates.front();
    selection.push_back(i);
    candidates.pop_front();

    // Remove conflicts
    candidates.remove_if([&_comp_mat,&i] (const int64_t &j) {
      auto ii = std::min(i, j), jj = std::max(i, j);

      auto block_idx = jj / THREADS_PER_BLOCK;
      auto bit_idx = jj % THREADS_PER_BLOCK;
      return _comp_mat[ii][block_idx] & (int64_t(1) << bit_idx);
    });
  }

  // Copy to output
  auto selection_tensor = at::zeros(selection.size(), comp_mat.options());
  std::copy(selection.begin(), selection.end(), selection_tensor.data<int64_t>());

  return selection_tensor;
}