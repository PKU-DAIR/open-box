# 安装指南

## 1 系统要求

安装要求：
+ Python >= 3.8 （推荐版本为Python 3.8）

支持系统：
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

## 2 预先准备

我们**强烈建议**您为OpenBox创建一个单独的Python环境，例如通过
[Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox python=3.8
conda activate openbox
```

我们建议您在安装OpenBox之前通过以下命令更新`pip`，`setuptools`和`wheel`：
```bash
pip install --upgrade pip setuptools wheel
```

## 3 安装 OpenBox

### 3.1 使用 PyPI 安装

只需运行以下命令：

```bash
pip install openbox
```

如需使用高级功能，请先{ref}`安装SWIG <installation/install_swig:swig 安装教程>`
，然后运行 `pip install "openbox[extra]"`。

### 3.2 从源代码手动安装

使用以下命令通过Github源码安装最新版本的OpenBox:
```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
pip install .
```

同样，如需使用高级功能，请先{ref}`安装SWIG <installation/install_swig:swig 安装教程>`
，然后运行 `pip install ".[extra]"`。

### 3.3 安装测试

运行以下代码以测试您安装是否成功：

```python
from openbox import run_test

if __name__ == '__main__':
    run_test()
```

如果成功，将输出以下信息：

```
===== Congratulations! All trials succeeded. =====
```

如果您在安装过程中遇到任何问题，请参考 **疑难解答** 。

## 4 进阶功能安装（可选）

如果您想使用更高级的功能，比如使用 `pyrfr` （概率随机森林）作为代理模型，或根据历史计算超参数重要性，
请先{ref}`安装SWIG <installation/install_swig:swig 安装教程>`，然后运行：
```bash
pip install "openbox[extra]"
```

如果您在安装`pyrfr`时遇到问题，请参考 {ref}`Pyrfr安装教程 <installation/install_pyrfr:pyrfr 安装教程>`。

## 5 疑难解答

如果以下未能解决您的安装问题, 请在Github上[提交issue](https://github.com/PKU-DAIR/open-box/issues) 。

### Windows

+ 对于在为某些包（例如 ConfigSpace 或 pyrfr）构建 wheel 时遇到问题的 Windows 用户，
  错误信息类似于 'ERROR: Failed building wheel for XXX' 或 'Microsoft Visual C++ 14.0 is required'，
  请参考 [安装 Microsoft Visual C++ 依赖项](./install_microsoft_vc.md)。

+ 'Error: \[WinError 5\] 拒绝访问'。请使用管理员权限运行命令行，或在命令后添加`--user`。

+ 对于 Windows 用户，如果您在安装 lazy_import 时遇到了困难，请参考 
  [提示](./install-lazy_import-on-windows.md)。(Deprecated in 0.7.10)

### macOS

+ 对于 macOS 用户，如果您在安装 pyrfr 时遇到了困难，请参考 [提示](./install-pyrfr-on-macos.md)。

+ 对于 macOS 用户，如果您在编译 scikit-learn 时遇到了困难。 [该教程](./openmp_macos.md) 或许对您有帮助。

+ 对于 macOS 用户，如果您在安装 lightgbm 时像 [Issue #57](https://github.com/PKU-DAIR/open-box/issues/57) 一样遇到
  "Failed building wheel for lightgbm"，[LightGBM官方安装教程](
  https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos) 或许对您有帮助。
