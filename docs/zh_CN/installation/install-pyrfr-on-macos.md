# 在 macOS 上安装 pyrfr

1. 下载 [pyrfr源码](https://pypi.org/project/pyrfr/#files).
2. 解压文件，进入 `pyrfr` 目录。
3. 按如下方法修改 `setup.py` 中的 `extra_compile_args` ：
    ```python
    extra_compile_args = ['-O2', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
    ```
4. 运行以下命令：
    ```bash
    env CC="/usr/bin/gcc -stdlib=libc++ -mmacosx-version-min=10.7" pip install .
    ```
