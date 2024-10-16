#ifndef UTILS_H
#define UTILS_H

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <xpu/Stream.h>

inline sycl::queue &getCurrentXPUQueue() {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream dpcpp_stream = impl.getStream(impl.getDevice());
  return xpu::get_queue_from_stream(dpcpp_stream);
}

#define AT_CUDA_CHECK(EXPR) (EXPR)

#endif
