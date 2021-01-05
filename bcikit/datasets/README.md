# Dataset

## Motor imagery

## SSVEP

### OpenBMI (SSVEP)
EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy.
(Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee, John Williamson, Siamac Fazli, Seong-Whan Lee.)
https://academic.oup.com/gigascience/article/8/5/giz002/5304369

```python
from bcikit.datasets.ssvep import OpenBMISSVEP
dataset = OpenBMISSVEP(root="_data/openbmissvep", subject_id=1, session=1, verbose=True)
print(dataset.data.shape) # (1, 100, 62, 4000)
print(dataset.targets)
```

### HS-SSVEP
A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces
(Yijun Wang, Xiaogang Chen, Xiaorong Gao, Shangkai Gao)
https://ieeexplore.ieee.org/document/7740878

```python
from bcikit.datasets.ssvep import HSSSVEP
dataset = HSSSVEP(root="_data/hsssvep", subject_id=1, verbose=True)
print(dataset.data.shape) # (1, 240, 64, 1000)
print(dataset.targets)
```

## Emotion

## Event-related potential

## Music

## Mental states
