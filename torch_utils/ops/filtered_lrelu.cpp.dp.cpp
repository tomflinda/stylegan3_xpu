// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/core/DeviceGuard.h>

#include "filtered_lrelu.h"
#include <c10/xpu/XPUStream.h>

//------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si,
    int up, int down, int px0, int px1, int py0, int py1, int sx, int sy, float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
{
    // Set CUDA device.
    TORCH_CHECK(x.is_xpu(), "x must reside on XPU device");
    const c10::OptionalDeviceGuard device_guard(device_of(x));

    // Validate arguments.
    TORCH_CHECK(fu.device() == x.device() && fd.device() == x.device() && b.device() == x.device(), "all input tensors must reside on the same device");
    TORCH_CHECK(fu.dtype() == torch::kFloat && fd.dtype() == torch::kFloat, "fu and fd must be float32");
    TORCH_CHECK(b.dtype() == x.dtype(), "x and b must have the same dtype");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat, "x and b must be float16 or float32");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK((fu.dim() == 1 || fu.dim() == 2) && (fd.dim() == 1 || fd.dim() == 2), "fu and fd must be rank 1 or 2");
    TORCH_CHECK(fu.size(0) <= INT_MAX && fu.size(-1) <= INT_MAX, "fu is too large");
    TORCH_CHECK(fd.size(0) <= INT_MAX && fd.size(-1) <= INT_MAX, "fd is too large");
    TORCH_CHECK(fu.numel() > 0, "fu is empty");
    TORCH_CHECK(fd.numel() > 0, "fd is empty");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(1), "b must be a vector with the same number of channels as x");
    TORCH_CHECK(up >= 1 && down >= 1, "up and down must be at least 1");

    // Figure out how much shared memory is available on the device.
    int maxSharedBytes = 0;
    /*
    DPCT1009:49: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1019:48: local_mem_size in SYCL is not a complete equivalent of
    cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust the
    code.
    */
    (DPCT_CHECK_ERROR(
        maxSharedBytes =
            dpct::get_device(x.device().index()).get_local_mem_size()));
    maxSharedBytes = maxSharedBytes >> 2;
    int sharedKB = maxSharedBytes >> 10;

    // Populate enough launch parameters to check if a CUDA kernel exists.
    filtered_lrelu_kernel_params p;
    p.up      = up;
    p.down    = down;
    p.fuShape = sycl::int2((int)fu.size(-1),
                           fu.dim() == 2
                               ? (int)fu.size(0)
                               : 0); // shape [n, 0] indicates separable filter.
    p.fdShape =
        sycl::int2((int)fd.size(-1), fd.dim() == 2 ? (int)fd.size(0) : 0);
    filtered_lrelu_kernel_spec test_spec =
        choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
    if (!test_spec.exec)
    {
        // No kernel found - return empty tensors and indicate missing kernel with return code of -1.
        return std::make_tuple(torch::Tensor(), torch::Tensor(), -1);
    }

    // Input/output element size.
    int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;

    // Input sizes.
    int64_t xw = (int)x.size(3);
    int64_t xh = (int)x.size(2);
    int64_t fut_w = (int)fu.size(-1) - 1;
    int64_t fut_h = (int)fu.size(0)  - 1;
    int64_t fdt_w = (int)fd.size(-1) - 1;
    int64_t fdt_h = (int)fd.size(0)  - 1;

    // Logical size of upsampled buffer.
    int64_t cw = xw * up + (px0 + px1) - fut_w;
    int64_t ch = xh * up + (py0 + py1) - fut_h;
    TORCH_CHECK(cw > fdt_w && ch > fdt_h, "upsampled buffer must be at least the size of downsampling filter");
    TORCH_CHECK(cw <= INT_MAX && ch <= INT_MAX, "upsampled buffer is too large");

    // Compute output size and allocate.
    int64_t yw = (cw - fdt_w + (down - 1)) / down;
    int64_t yh = (ch - fdt_h + (down - 1)) / down;
    TORCH_CHECK(yw > 0 && yh > 0, "output must be at least 1x1");
    TORCH_CHECK(yw <= INT_MAX && yh <= INT_MAX, "output is too large");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), yh, yw}, x.options(), x.suggest_memory_format());

    // Allocate sign tensor.
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    int64_t sw_active = 0; // Active width of sign tensor.
    if (writeSigns)
    {
        sw_active = yw * down - (down - 1) + fdt_w;     // Active width in elements.
        int64_t sh = yh * down - (down - 1) + fdt_h;    // Height = active height.
        int64_t sw = (sw_active + 15) & ~15;            // Width  = active width in elements, rounded up to multiple of 16.
        TORCH_CHECK(sh <= INT_MAX && (sw >> 2) <= INT_MAX, "signs is too large");
        s = so = torch::empty({x.size(0), x.size(1), sh, sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }
    else if (readSigns)
        sw_active = s.size(3) << 2;

    // Validate sign tensor if in use.
    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
        TORCH_CHECK(s.size(2) <= INT_MAX && s.size(3) <= INT_MAX, "signs is too large");
    }

    // Populate rest of CUDA kernel parameters.
    p.x         = x.data_ptr();
    p.y         = y.data_ptr();
    p.b         = b.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.fu        = fu.data_ptr<float>();
    p.fd        = fd.data_ptr<float>();
    p.pad0 = sycl::int2(px0, py0);
    p.gain = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.flip      = (flip_filters) ? 1 : 0;
    p.xShape = sycl::int4((int)x.size(3), (int)x.size(2), (int)x.size(1),
                          (int)x.size(0));
    p.yShape = sycl::int4((int)y.size(3), (int)y.size(2), (int)y.size(1),
                          (int)y.size(0));
    p.sShape = (readSigns || writeSigns)
                   ? sycl::int2((int)s.size(3), (int)s.size(2))
                   : sycl::int2(0, 0); // Width is in bytes. Contiguous.
    p.sOfs = sycl::int2(sx, sy);
    p.swLimit = (sw_active + 3) >> 2; // Rounded up to bytes.

    // x, y, b strides are in bytes.
    p.xStride = sycl::long4(sz * x.stride(3), sz * x.stride(2),
                            sz * x.stride(1), sz * x.stride(0));
    p.yStride = sycl::long4(sz * y.stride(3), sz * y.stride(2),
                            sz * y.stride(1), sz * y.stride(0));
    p.bStride = sz * b.stride(0);

    // fu, fd strides are in elements.
    p.fuStride =
        sycl::long3(fu.stride(-1), fu.dim() == 2 ? fu.stride(0) : 0, 0);
    p.fdStride =
        sycl::long3(fd.stride(-1), fd.dim() == 2 ? fd.stride(0) : 0, 0);

    // Determine if indices don't fit in int32. Support negative strides although Torch currently never produces those.
    bool index64b = false;
    if (std::abs(p.bStride * x.size(1)) > INT_MAX) index64b = true;
    if (std::min(x.size(0) * p.xStride.w(), 0l) +
            std::min(x.size(1) * p.xStride.z(), 0l) +
            std::min(x.size(2) * p.xStride.y(), 0l) +
            std::min(x.size(3) * p.xStride.x(), 0l) <
        -INT_MAX) index64b = true;
    if (std::max(x.size(0) * p.xStride.w(), 0l) +
            std::max(x.size(1) * p.xStride.z(), 0l) +
            std::max(x.size(2) * p.xStride.y(), 0l) +
            std::max(x.size(3) * p.xStride.x(), 0l) >
        INT_MAX) index64b = true;
    if (std::min(y.size(0) * p.yStride.w(), 0l) +
            std::min(y.size(1) * p.yStride.z(), 0l) +
            std::min(y.size(2) * p.yStride.y(), 0l) +
            std::min(y.size(3) * p.yStride.x(), 0l) <
        -INT_MAX) index64b = true;
    if (std::max(y.size(0) * p.yStride.w(), 0l) +
            std::max(y.size(1) * p.yStride.z(), 0l) +
            std::max(y.size(2) * p.yStride.y(), 0l) +
            std::max(y.size(3) * p.yStride.x(), 0l) >
        INT_MAX) index64b = true;
    if (s.numel() > INT_MAX) index64b = true;

    // Choose CUDA kernel.
    filtered_lrelu_kernel_spec spec = { 0 };
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_cuda", [&]
    {
        if constexpr (sizeof(scalar_t) <= 4) // Exclude doubles. constexpr prevents template instantiation.
        {
            // Choose kernel based on index type, datatype and sign read/write modes.
            if      (!index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, true,  false>(p, sharedKB);
            else if (!index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, true >(p, sharedKB);
            else if (!index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, false>(p, sharedKB);
            else if ( index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, true,  false>(p, sharedKB);
            else if ( index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, true >(p, sharedKB);
            else if ( index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, false>(p, sharedKB);
        }
    });
    TORCH_CHECK(spec.exec, "internal error - CUDA kernel not found") // This should not happen because we tested earlier that kernel exists.

    // Launch CUDA kernel.
    void* args[] = {&p};
    int bx = spec.numWarps * 32;
    int gx = (p.yShape.x() - 1) / spec.tileOut.x() + 1;
    int gy = (p.yShape.y() - 1) / spec.tileOut.y() + 1;
    int gz = p.yShape.z() * p.yShape.w();

    // Repeat multiple horizontal tiles in a CTA?
    if (spec.xrep)
    {
        p.tilesXrep = spec.xrep;
        p.tilesXdim = gx;

        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }
    else
    {
        p.tilesXrep = 0;
        p.tilesXdim = 0;
    }

    // Launch filter setup kernel.
    DPCT_CHECK_ERROR(dpct::kernel_launcher::launch(
        spec.setup, 1, 1024, args, 0,
        &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));

    // Copy kernels to constant memory.
    if      ( writeSigns && !readSigns) (copy_filters<true,  false>(&static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));
    else if (!writeSigns &&  readSigns) (copy_filters<false, true >(&static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));
    else if (!writeSigns && !readSigns) (copy_filters<false, false>(&static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));

    // Set cache and shared memory configurations for main kernel.
    /*
    DPCT1009:55: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1027:54: The call to cudaFuncSetCacheConfig was replaced with 0 because
    SYCL currently does not support configuring shared memory on devices.
    */
    (0);
    if (spec.dynamicSharedKB) // Need dynamically allocated shared memory?
        /*
        DPCT1009:57: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        /*
        DPCT1027:56: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        (0);
    /*
    DPCT1009:59: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1027:58: The call to cudaFuncSetSharedMemConfig was replaced with 0
    because SYCL currently does not support configuring shared memory on
    devices.
    */
    (0);

    // Launch main kernel.
    const int maxSubGz = 65535; // CUDA maximum for block z dimension.
    for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
    {
        p.blockZofs = zofs;
        int subGz = std::min(maxSubGz, gz - zofs);
        DPCT_CHECK_ERROR(dpct::kernel_launcher::launch(
            spec.exec, dpct::dim3(gx, gy, subGz), bx, args,
            spec.dynamicSharedKB << 10,
            &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));
    }

    // Done.
    return std::make_tuple(y, so, 0);
}

//------------------------------------------------------------------------

static torch::Tensor filtered_lrelu_act(torch::Tensor x, torch::Tensor si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
{
    // Set CUDA device.
    TORCH_CHECK(x.is_xpu(), "x must reside on XPU device");
    const c10::OptionalDeviceGuard device_guard(device_of(x));

    // Validate arguments.
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat || x.dtype() == torch::kDouble, "x must be float16, float32 or float64");

    // Output signs if we don't have sign input.
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    if (writeSigns)
    {
        int64_t sw = x.size(3);
        sw = (sw + 15) & ~15; // Round to a multiple of 16 for coalescing.
        s = so = torch::empty({x.size(0), x.size(1), x.size(2), sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }

    // Validate sign tensor if in use.
    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
        TORCH_CHECK(s.size(2) <= INT_MAX && (s.size(3) << 2) <= INT_MAX, "signs tensor is too large");
    }

    // Initialize CUDA kernel parameters.
    filtered_lrelu_act_kernel_params p;
    p.x         = x.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.xShape = sycl::int4((int)x.size(3), (int)x.size(2), (int)x.size(1),
                          (int)x.size(0));
    p.xStride = sycl::long4(x.stride(3), x.stride(2), x.stride(1), x.stride(0));
    p.sShape = (readSigns || writeSigns)
                   ? sycl::int2((int)s.size(3) << 2, (int)s.size(2))
                   : sycl::int2(0, 0); // Width is in elements. Contiguous.
    p.sOfs      = sycl::int2(sx, sy);

    // Choose CUDA kernel.
    void* func = 0;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_act_cuda", [&]
    {
        if (writeSigns)
            func = choose_filtered_lrelu_act_kernel<scalar_t, true, false>();
        else if (readSigns)
            func = choose_filtered_lrelu_act_kernel<scalar_t, false, true>();
        else
            func = choose_filtered_lrelu_act_kernel<scalar_t, false, false>();
    });
    TORCH_CHECK(func, "internal error - CUDA kernel not found");

    // Launch CUDA kernel.
    void* args[] = {&p};
    int bx = 128; // 4 warps per block.

    // Logical size of launch = writeSigns ? p.s : p.x
    uint32_t gx = writeSigns ? p.sShape.x() : p.xShape.x();
    uint32_t gy = writeSigns ? p.sShape.y() : p.xShape.y();
    uint32_t gz =
        p.xShape.z() * p.xShape.w(); // Same as in p.sShape if signs are in use.
    gx = (gx - 1) / bx + 1;

    // Make sure grid y and z dimensions are within CUDA launch limits. Kernel loops internally to do the rest.
    const uint32_t gmax = 65535;
    gy = std::min(gy, gmax);
    gz = std::min(gz, gmax);

    // Launch.
    DPCT_CHECK_ERROR(dpct::kernel_launcher::launch(
        func, dpct::dim3(gx, gy, gz), bx, args, 0,
        &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())));
    return so;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("filtered_lrelu",      &filtered_lrelu);      // The whole thing.
    m.def("filtered_lrelu_act_", &filtered_lrelu_act);  // Activation and sign tensor handling only. Modifies data tensor in-place.
}

//------------------------------------------------------------------------
