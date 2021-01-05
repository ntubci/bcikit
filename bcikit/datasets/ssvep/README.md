# Dataset

## SSVEP

### HS-SSVEP
A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces
Yijun Wang, Xiaogang Chen, Xiaorong Gao, Shangkai Gao
https://ieeexplore.ieee.org/document/7740878

```python
from bcikit.datasets.ssvep import HSSSVEP
dataset = HSSSVEP(root="_data/hsssvep", subject_id=1, verbose=True)
print(dataset.data.shape)
print(dataset.targets)
```
