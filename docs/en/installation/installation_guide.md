# Installation Guide

## 1 System Requirements

Installation Requirements:
+ Python >= 3.8 (3.8 is recommended!)

Supported Systems:
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

## 2 Preparations before Installation

We **STRONGLY** suggest you to create a Python environment via 
[Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox python=3.8
conda activate openbox
```

Then we recommend you to update your `pip`, `setuptools` and `wheel` as follows:
```bash
pip install --upgrade pip setuptools wheel
```

## 3 Install OpenBox

### 3.1 Installation from PyPI

To install OpenBox from PyPI, simply run the following command:

```bash
pip install openbox
```

For advanced features, {ref}`install SWIG <installation/install_swig:swig installation guide>`
first and then run `pip install "openbox[extra]"`. 

### 3.2 Manual Installation from Source

To install the newest OpenBox from the source code, please run the following commands:
```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
pip install .
```

Also, for advanced features, {ref}`install SWIG <installation/install_swig:swig installation guide>`
first and then run `pip install ".[extra]"`.

### 3.3 Test for Installation

You can run the following code to test your installation:

```python
from openbox import run_test

if __name__ == '__main__':
    run_test()
```

If successful, you will receive the following message:

```
===== Congratulations! All trials succeeded. =====
```

If you encountered any problem during installation, please refer to the **Trouble Shooting** section.

## 4 Installation for Advanced Features (Optional)

To use advanced features such as `pyrfr` (probabilistic random forest) surrogate and get hyper-parameter 
importance from history, please {ref}`install SWIG <installation/install_swig:swig installation guide>` 
first, and then run:
```bash
pip install "openbox[extra]"
```

If you encounter problems installing `pyrfr`, please refer to 
{ref}`Pyrfr Installation Guide <installation/install_pyrfr:pyrfr installation guide>`.

## 5 Trouble Shooting

If you encounter problems not listed below, please 
[File an issue](https://github.com/PKU-DAIR/open-box/issues) on GitHub.

### Windows

+ For Windows users who have trouble building wheel for some packages, e.g. ConfigSpace or pyrfr, 
  the error message is like 'ERROR: Failed building wheel for XXX' or 'Microsoft Visual C++ 14.0 is required', 
  please refer to [Install Microsoft Visual C++ Dependencies](./install_microsoft_vc.md).

+ 'Error: \[WinError 5\] Access denied'. Please open the command prompt with administrative privileges or 
  append `--user` to the command line.

+ For Windows users who have trouble installing lazy_import, please refer to 
  [tips](./install-lazy_import-on-windows.md). (Deprecated in 0.7.10)

### macOS

+ For macOS users who have trouble installing pyrfr, please refer to [tips](./install-pyrfr-on-macos.md).

+ For macOS users who have trouble building scikit-learn, this [documentation](./openmp_macos.md) might help. 

+ For macOS users who failed building wheel for lightgbm like [Issue #57](
  https://github.com/PKU-DAIR/open-box/issues/57), the [LightGBM official installation guide](
  https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos) might help. 
