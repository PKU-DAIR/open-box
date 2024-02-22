# 安装 Microsoft Visual C++ 依赖项

对于在为某些包（例如 ConfigSpace 或 pyrfr）构建 wheel 时遇到问题的 Windows 用户，
错误信息类似于 'ERROR: Failed building wheel for XXX' 或 'Microsoft Visual C++ 14.0 is required'，
本文档将帮助您在 Windows 上安装 Microsoft Visual C++ 依赖项。

## 1. 升级 setuptools

首先，请将 setuptools 升级到最新版本：

```bash
pip install --upgrade setuptools
```

## 2. 下载并安装 Microsoft Visual C++ 构建工具

访问 Visual Studio 的官方网站并选择 **下载**：<https://visualstudio.microsoft.com/downloads/>

滚动到页面底部，找到 **用于 Visual Studio 的工具** 部分，并下载 **Visual Studio 生成工具**。

下载后，运行安装程序，选择 **使用 C++ 的桌面开发**，并在右侧面板中勾选 **至少前两个选项**：

<img src="../../imgs/installation/install_vc_build_tools.png" width="90%" class="align-center">

然后单击 **安装** 开始安装。

## 3. 重新安装包

安装 Microsoft Visual C++ 依赖项后，您可以重新安装之前无法构建 wheel 的包。

例如，如果您想安装 pyrfr，请运行以下命令：

```bash
pip install pyrfr
```

-----
参考：<https://zhuanlan.zhihu.com/p/165008313>
