// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <ATen/ATen.h>

at::Tensor mask_count_cpu(const at::Tensor& bbx, const at::Tensor& int_mask);
at::Tensor mask_count_cuda(const at::Tensor& bbx, const at::Tensor& int_mask);
