## Writing code and documentation

Coding styles, folder structure, and documentation are inspired by [PyTorch](https://github.com/pytorch/pytorch). Like PyTorch, we use [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. Preferably, the length of a line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups. For naming conventions, read the [Python Naming Conventions](https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html) documentation. For reference, this is a good example, refer to this [code](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d) for this [docstrings](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)

## Important notes

- file names: all folders and file names should be lowercase, separate words with underscore `_`, for example `auto_augment.py`
- class name: all class name should follow the UpperCaseCamelCase convention, for example `class AutoAugmentPolicy()`
- package paths: use `__init__.py` to define import path
- method and function names: should be all lower case
- constant: must be fully capitalized
- prefix underscore `_` for local functions and variables
- we try to avoid adding additional packages. See the `Adding additional packages` section below.
- remove unnecessary print statements, use `verbose` to print logs for debugging.

## Adding additional packages

We try our best to implement with common packages such as Numpy and packages listed in [requirements.txt](https://github.com/ntubci/bcikit/blob/main/requirements.txt). This reduces packages conflict and makes it easy for installation. But if your implementation requires to use packages not listed in [requirements.txt](https://github.com/ntubci/bcikit/blob/main/requirements.txt), add the package name and version into [requirements.txt](https://github.com/ntubci/bcikit/blob/main/requirements.txt). 
