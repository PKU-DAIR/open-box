# How to install lazy_import on Windows

If you encounter `UnicodeDecodeError` when installing `lazy_import` on Windows,
try the following steps:

1. Download [the source code of lazy_import from PyPI](https://pypi.org/project/lazy-import/#files).
2. Unzip the file and open `setup.py`.
3. Modify line 5 and 6 as follows:
    ```python
    with open('README.rst', encoding='utf-8') as infile:
        readme = infile.read()
    ```
4. Run the following command to install lazy_import from source code:
    ```bash
    python setup.py install
    ```
