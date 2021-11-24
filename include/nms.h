// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <ATen/ATen.h>

const int64_t THREADS_PER_BLOCK = sizeof(int64_t) * 8;

at::Tensor comp_mat_cpu(const at::Tensor& bbx, float threshold);
at::Tensor comp_mat_cuda(const at::Tensor& bbx, float threshold);

at::Tensor nms_cpu(const at::Tensor& comp_mat, const at::Tensor& scores, int n_max);
