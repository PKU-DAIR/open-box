# SWIG 安装教程

本教程将指导您安装 SWIG 3.0。

## Linux

```bash
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

## MacOS

1. 从 [SWIG 下载页面](http://www.swig.org/download.html) 下载SWIG。
点击 **All releases** ，而后选择 **'swig'** 文件夹。我们推荐安装版本 3.0.12。在下载后解压文件。


2. 从 [这里](http://www.pcre.org) 下载pcre。
下载 'pcre-8.44.tar.bz2' ，把它重命名为 'pcre-8.44.tar'。而后把它放到 SWIG 文件夹里。

3. 在SWIG文件夹下运行以下命令：

```bash
./Tools/pcre-build.sh
./autogen.sh
./configure
make
sudo make install
```

## Windows

1.  从 [SWIG 下载页面](http://www.swig.org/download.html) 下载 SWIG 的 Windows 版。
点击 **All releases** ，而后选择 **'swigwin'** 文件夹。我们推荐安装版本 3.0.12。在下载后解压文件。

2. 把 SWIG 文件夹添加到系统路径.
