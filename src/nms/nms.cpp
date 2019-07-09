#include <torch/extension.h>

#include "nms.h"
#include "utils/checks.h"

at::Tensor nms(const at::Tensor& bbx, const at::Tensor& scores, float threshold, int n_max) {
  // Check inputs
  AT_CHECK(bbx.scalar_type() == scores.scalar_type(), "bbx and scores must have the same type");
  AT_CHECK(bbx.size(0) == scores.size(0), "bbx and scores must have the same length");
  AT_CHECK(bbx.size(1) == 4 && bbx.ndimension() == 2, "bbx must be an N x 4 tensor");
  AT_CHECK(bbx.is_contiguous(), "bbx must be a contiguous tensor");

  at::Tensor comp_mat;
  if (bbx.is_cuda()) {
    comp_mat = comp_mat_cuda(bbx, threshold);
    comp_mat = comp_mat.toBackend(at::Backend::CPU);
  } else {
    comp_mat = comp_mat_cpu(bbx, threshold);
  }

  // Sort scores
  auto sorted_and_idx = scores.sort(0, true);
  auto idx = std::get<1>(sorted_and_idx);

  // Run actual non-maxima suppression on CPU
  return nms_cpu(comp_mat, idx.toBackend(at::Backend::CPU), n_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "Perform non-maxima suppression, always return result as CPU Tensor");
}
