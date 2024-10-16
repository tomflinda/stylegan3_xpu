import torch
import intel_extension_for_pytorch as ipex
import numpy as np

from torch_utils.ops import bias_act
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import upfirdn2d

# filtered_lrelu.filtered_lrelu()
# upfirdn2d.upfirdn2d()


def load_bias():
    x = torch.randn(10, 5, dtype=torch.float32).to('xpu')
    b = torch.randn(5, dtype=torch.float32).to('xpu')
    output = bias_act.bias_act(x=x, b=b, dim=1, act='relu', alpha=0.2, gain=1, clamp=5, impl='xpu')
    print(output)

def load_filtered_lrelu():
    x = torch.randn(4, 3, 64, 64, dtype=torch.float32).to('xpu')
    b = torch.randn(3, dtype=torch.float32).to('xpu')
    fu = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32).to('xpu') / 16
    fd = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32).to('xpu') / 16
    output = filtered_lrelu.filtered_lrelu(
        x, fu=fu, fd=fd, b=b, up=2, down=2, padding=1, gain=np.sqrt(2), 
        slope=0.2, clamp=None, flip_filter=False, impl='xpu'
    )
    print(output)

def load_upfirdn2d():
    x = torch.randn(4,3,64,64).to('xpu')
    f = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32).to('xpu') / 16
    output = upfirdn2d.upfirdn2d(x, f=f, up=2, down=2, padding=1, flip_filter=False, gain=1, impl='xpu')
    print(output)

if __name__ == '__main__':
    load_bias()
    #load_filtered_lrelu()
    #load_upfirdn2d()
