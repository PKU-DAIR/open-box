## Enable OpenMP on macOS For Building scikit-learn

The default C compiler on macOS, Apple clang (confusingly aliased as `/usr/bin/gcc`), does not directly support OpenMP. 

Below we present a solution to this issue: install `libomp` with Homebrew to extend the default Apple clang compiler. 

Please note that if you are using Apple Silicon M1 hardware, this solution might not work. You may need to refer to [Scikit-learn Official Documentation](https://scikit-learn.org/dev/developers/advanced_installation.html#mac-osx) for another solution.


----

First, install the macOS command line tools:

```bash
$ xcode-select --install 
```

Install the [Homebrew](https://brew.sh) package manager for macOS. 

Install the LLVM OpenMP library. 

```bash
$ brew install libomp
```

Set the following environment variables:


```bash
$ export CC=/usr/bin/clang
$ export CXX=/usr/bin/clang++
$ export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
$ export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
$ export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
$ export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
```



Now, you should be able to compile your scikit-learn properly. 
For further questions, please refer to [this](https://github.com/scikit-learn/scikit-learn/issues/13371) or [Scikit-learn Official Documentation](https://scikit-learn.org/dev/developers/advanced_installation.html#mac-osx). 
