# How to install pyrfr on macOS

1. Download [the source code of pyrfr](https://pypi.org/project/pyrfr/#files).
2. Unzip the file and go to `pyrfr`'s directory.
3. Modify the `extra_compile_args` in `setup.py` as follows
    ```python
    extra_compile_args = ['-O2', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
    ```
4. Run the following command:
    ```bash
    env CC="/usr/bin/gcc -stdlib=libc++ -mmacosx-version-min=10.7" pip install .
    ```
