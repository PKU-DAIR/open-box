# 在 Windows 上安装 lazy_import 

如果你在 Windows 上安装 `lazy_import` 时遇到了 `UnicodeDecodeError` ，试试以下解决方案：


1. 下载 [the source code of lazy_import from PyPI](https://pypi.org/project/lazy-import/#files).
2. 解压文件，打开 `setup.py`.
3. 修改第五行和第六行如下：
    ```python
    with open('README.rst', encoding='utf-8') as infile:
        readme = infile.read()
    ```
4. 运行下列命令，从源代码安装 lazy_import：
    ```bash
    python setup.py install
    ```
