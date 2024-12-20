/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_main.h"
#include <cstdio>
#include <tuple>
#include <string>
#include <pybind11/pybind11.h>


//---------------------------------------------------

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}


pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["rasterize_gaussians_fwd"] = EncapsulateFunction(RasterizeGaussiansCUDAJAX);
  return dict;
}

auto BuildRasterizeDescriptor(int image_height, int image_width, int P, float tan_fovx, float tan_fovy) {
  RasterizeDescriptor d;
  d.image_height = image_height;
  d.image_width = image_width;
  d.P = P;
  d.tan_fovx = tan_fovx;
  d.tan_fovy = tan_fovy;
  return PackDescriptor(d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registrations", &Registrations, "custom call registrations");
  m.def("build_gaussian_rasterize_descriptor", &BuildRasterizeDescriptor);
}
