// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
//#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceGuard.h>
#include <xpu/Stream.h>


#include "upfirdn2d.h"

//------------------------------------------------------------------------

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_xpu(), "x must reside on XPU device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(f.numel() <= INT_MAX, "f is too large");
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK((x.size(0)-1)*x.stride(0) + (x.size(1)-1)*x.stride(1) + (x.size(2)-1)*x.stride(2) + (x.size(3)-1)*x.stride(3) <= INT_MAX, "x memory footprint is too large");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
    TORCH_CHECK(downx >= 1 && downy >= 1, "downsampling factor must be at least 1");

    // Create output tensor.
    const at::OptionalDeviceGuard device_guard(device_of(x));
    int outW = ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
    int outH = ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());
    TORCH_CHECK(y.numel() <= INT_MAX, "output is too large");
    TORCH_CHECK((y.size(0)-1)*y.stride(0) + (y.size(1)-1)*y.stride(1) + (y.size(2)-1)*y.stride(2) + (y.size(3)-1)*y.stride(3) <= INT_MAX, "output memory footprint is too large");

    // Initialize CUDA kernel parameters.
    upfirdn2d_kernel_params p;
    p.x             = x.data_ptr();
    p.f             = f.data_ptr<float>();
    p.y             = y.data_ptr();
    p.up = sycl::int2(upx, upy);
    p.down = sycl::int2(downx, downy);
    p.pad0 = sycl::int2(padx0, pady0);
    p.flip          = (flip) ? 1 : 0;
    p.gain          = gain;
    p.inSize = sycl::int4((int)x.size(3), (int)x.size(2), (int)x.size(1),
                          (int)x.size(0));
    p.inStride = sycl::int4((int)x.stride(3), (int)x.stride(2),
                            (int)x.stride(1), (int)x.stride(0));
    p.filterSize = sycl::int2((int)f.size(1), (int)f.size(0));
    p.filterStride = sycl::int2((int)f.stride(1), (int)f.stride(0));
    p.outSize = sycl::int4((int)y.size(3), (int)y.size(2), (int)y.size(1),
                           (int)y.size(0));
    p.outStride = sycl::int4((int)y.stride(3), (int)y.stride(2),
                             (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor =
        (p.inStride.z() == 1) ? p.inSize.w() : p.inSize.w() * p.inSize.z();
    p.sizeMinor = (p.inStride.z() == 1) ? p.inSize.z() : 1;

    // Choose CUDA kernel.
    upfirdn2d_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        spec = choose_upfirdn2d_kernel<scalar_t>(p);
    });

    // Set looping options.
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;

    // Compute grid size.
    sycl::range<3> blockSize(1, 1, 1), gridSize(1, 1, 1);
    if (spec.tileOutW < 0) // large
    {
        blockSize = sycl::range<3>(1, 32, 4);
        gridSize = sycl::range<3>(
            p.launchMajor, (p.outSize.x() - 1) / (blockSize[1] * p.loopX) + 1,
            ((p.outSize.y() - 1) / blockSize[2] + 1) * p.launchMinor);
    }
    else // small
    {
        blockSize = sycl::range<3>(1, 1, 256);
        gridSize = sycl::range<3>(
            p.launchMajor, (p.outSize.x() - 1) / (spec.tileOutW * p.loopX) + 1,
            ((p.outSize.y() - 1) / spec.tileOutH + 1) * p.launchMinor);
    }

    // Launch CUDA kernel.
    void* args[] = {&p};
    /*
    DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    /*
    DPCT1123:31: The kernel function pointer cannot be used in the device code.
    You need to call the kernel function with the correct argument(s) directly.
    According to the kernel function definition, adjusting the dimension of the
    sycl::nd_item may also be required.
    */
//   ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
//       ->parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
//                      [=](sycl::nd_item<3> item_ct1) {
//                        (spec.kernel)();
//                      });
//   AT_CUDA_CHECK(0);
    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upfirdn2d", &upfirdn2d);
}

//------------------------------------------------------------------------
