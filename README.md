# MEMFOF

[MEMFOF](https://github.com/msu-video-group/memfof),
as proposed in the paper ["MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation" (ICCV 2025)](https://arxiv.org/abs/2506.23151).

Forked to support installation with pip or other package managers.

## Usage

Install with your favorite package manager:

```shell
# install with pip
$ pip install git+https://github.com/yklcs/memfof

# install with uv
$ uv add git+https://github.com/yklcs/memfof
```

The MEMFOF model is exposed in `memfof.MEMFOF`:

```python
from memfof import MEMFOF
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
model = MEMFOF.from_pretrained(model_id).eval().to(device)

with torch.inference_mode():
    input = torch.randint(0, 256, [1, 3, 3, 1080, 1920], device=device)
    bwd_flow, fwd_flow = model(input)["flow"][-1].unbind(dim=1)
```
