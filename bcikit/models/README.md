# Models

## Multitask Model

Using multi-task learning to capture signals simultaneously from the fovea efficiently and the neighboring targets in the peripheral vision generate a visual response map. A calibration-free user-independent solution, desirable for clinical diagnostics. A stepping stone for an objective assessment of glaucoma patients’ visual field. Learn more about this model at https://jinglescode.github.io/ssvep-multi-task-learning/.

```python
import torch
from bcikit.models import MultitaskSSVEP

model = MultitaskSSVEP(num_channel=10,
    num_classes=40,
    signal_length=1000,
    filters_n1=4,
    kernel_window_ssvep=59,
    kernel_window=19,
    conv_3_dilation=4,
    conv_4_dilation=4,
)

x = torch.ones((20, 10, 1000))
print("Input shape:", x.shape)
y = model(x)
print("Output shape:", y.shape)
```

## EEGNet (Compact)

EEGNet: Compact Convolutional Neural Network (Compact-CNN) https://arxiv.org/pdf/1803.04566.pdf

```python
import torch
from bcikit.models import CompactEEGNet
model = CompactEEGNet(
    num_channel=10,
    num_classes=4,
    signal_length=1000,
)
print(model)
x = torch.rand(1,10,1000)
y = model(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)
```
