# Installation Guide

## 1 System Requirements

Installation Requirements:
+ Python >= 3.7 (3.7 is recommended!)

Supported Systems:
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

## 2 Preparations before Installation

We **STRONGLY** suggest you to create a Python environment via [Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox python=3.7
conda activate openbox
```

Then we recommend you to update your `pip` and `setuptools` as follows:
```bash
pip install pip setuptools --upgrade
```

## 3 Install OpenBox

### 3.1 Installation from PyPI

To install OpenBox from PyPI, simply run the following command:

```bash
pip install openbox
```

### 3.2 Manual Installation from Source

To install OpenBox using the source code, please run the following commands:
```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

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

## 4 Installation for Advanced Usage (Optional)

To use advanced features such as `pyrfr` (probabilistic random forest) surrogate and get hyper-parameter 
importance from history, please refer to [Pyrfr Installation Guide](./install_pyrfr.md) to install `pyrfr`.

## 5 Trouble Shooting

If you encounter problems not listed below, please [File an issue](https://github.com/PKU-DAIR/open-box/issues) 
on GitHub or email us via *liyang.cs@pku.edu.cn*.

### Windows

+ 'Error: \[WinError 5\] Access denied'. Please open the command prompt with administrative privileges or 
append `--user` to the command line.

+ 'ERROR: Failed building wheel for ConfigSpace'. Please refer to [tips](./install_configspace_on_win_fix_vc.md).

+ For Windows users who have trouble installing lazy_import, please refer to [tips](./install-lazy_import-on-windows.md). (Deprecated in 0.7.10)

### macOS

+ For macOS users who have trouble installing pyrfr, please refer to [tips](./install-pyrfr-on-macos.md).

+ For macOS users who have trouble building scikit-learn, this [documentation](./openmp_macos.md) might help. 
