# Models

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
