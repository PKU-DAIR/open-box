## 在 macOS 上开启 OpenMP （为了编译scikit-learn）

macOS 上的默认编译器 Apple clang (也称 `/usr/bin/gcc`) 不支持OpenMP，因此你在编译 scikit-learn 时可能遇到问题。

下面我们提供一种解决方案： 使用 Homebrew 安装 `libomp` 来扩展默认的 Apple clang 编译器。 

请注意，如果你电脑使用的是 Apple Silicon M1 硬件，以下方法可能无效。你需要参考 [Scikit-learn 官方文档](https://scikit-learn.org/dev/developers/advanced_installation.html#mac-osx) 中的另一种解决方案。



----

首先，安装mac的命令行工具：

```bash
$ xcode-select --install 
```

安装mac的 [Homebrew](https://brew.sh) 包管理器。

安装 LLVM OpenMP 库。 

```bash
$ brew install libomp
```

设置以下环境变量：


```bash
$ export CC=/usr/bin/clang
$ export CXX=/usr/bin/clang++
$ export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
$ export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
$ export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
$ export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
```


现在你应该可以正确编译 scikit-learn 了。
若还有问题，请参考 [这里](https://github.com/scikit-learn/scikit-learn/issues/13371) 或 [Scikit-learn 官方文档](https://scikit-learn.org/dev/developers/advanced_installation.html#mac-osx) 。
