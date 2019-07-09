#include <torch/extension.h>

#include "bbx.h"
#include "utils/checks.h"

at::Tensor extract_boxes(const at::Tensor& mask, int n_instances){
  AT_CHECK(mask.ndimension() == 3, "Input mask should be 3D");

  at::Tensor bbx = at::full({n_instances, 4}, -1, mask.options().dtype(at::kFloat));

  AT_DISPATCH_ALL_TYPES(mask.scalar_type(), "extract_boxes", ([&]{
    auto _mask = mask.accessor<scalar_t, 3>();
    auto _bbx = bbx.accessor<float, 2>();

    for (int c = 0; c < _mask.size(0); ++c) {
      for (int i = 0; i < _mask.size(1); ++i) {
        for (int j = 0; j < _mask.size(2); ++j) {
          int64_t id = static_cast<int64_t>(_mask[c][i][j]);
          if (id < n_instances) {
            if (_bbx[id][0] < 0 || _bbx[id][0] > i) _bbx[id][0] = i;
            if (_bbx[id][1] < 0 || _bbx[id][1] > j) _bbx[id][1] = j;
            if (_bbx[id][2] < 0 || _bbx[id][2] <= i) _bbx[id][2] = i + 1;
            if (_bbx[id][3] < 0 || _bbx[id][3] <= j) _bbx[id][3] = j + 1;
          }
        }
      }
    }
  }));

  return bbx;
}

at::Tensor mask_count(const at::Tensor& bbx, const at::Tensor& int_mask) {
  AT_CHECK(bbx.ndimension() == 2, "Input bbx should be 2D");
  AT_CHECK(bbx.size(1) == 4, "Input bbx must be N x 4");
  AT_CHECK(int_mask.ndimension() == 2, "Input mask should be 2D");

  if (bbx.is_cuda()) {
    return mask_count_cuda(bbx, int_mask);
  } else {
    return mask_count_cpu(bbx, int_mask);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_boxes", &extract_boxes, "Extract bounding boxes from image of instance IDs");
  m.def("mask_count", &mask_count, "Count the number of non-zero entries in different regions of a mask");
}

