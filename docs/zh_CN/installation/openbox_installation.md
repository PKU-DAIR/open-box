# OpenBox安装教程

本教程将介绍OpenBox的安装方法（包括Windows 10、macOS和Linux系统）

## 系统环境需求

+ Python >= 3.6
+ SWIG == 3.0.12

在安装OpenBox之前，请确保已经正确安装了SWIG（后面将介绍安装方法）。

## 0 *Python环境准备

我们推荐使用conda为OpenBox系统创建单独的Python环境，在安装conda以后，可以通过以下命令创建名为openbox的Python 3.7环境（也可以使用其他名称）：

```bash
conda create -n openbox python=3.7
```

然后输入以下命令进入该环境：

```bash
conda activate openbox
```

注意，后续安装与使用OpenBox过程中，请切换到该环境下进行操作。

## 1 SWIG安装方法

在安装OpenBox之前，请确保已经正确安装了SWIG。推荐安装版本为3.0.12。

### 1.1 Windows 10系统安装SWIG

首先，从SWIG官网下载SWIG for Windows（<http://www.swig.org/download.html>）。进入官网后，点击**All releases**按钮，然后选择**swigwin**文件夹，选择**swig-3.0.12**，下载[swigwin-3.0.12.zip](https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/swigwin-3.0.12.zip/download)压缩包。

将压缩包解压到合适的目录后，将解压文件夹路径（即"xxx/xxx/swigwin-3.0.12/"）添加至**Path**系统环境变量。添加系统环境变量的方法可以参照：<https://jingyan.baidu.com/article/00a07f3876cd0582d128dc55.html>。

### 1.2 macOS系统安装SWIG

（参考：<https://blog.csdn.net/yht1107030154/article/details/77968500>）

首先，从SWIG官网下载SWIG（<http://www.swig.org/download.html>）。进入官网后，点击**All releases**按钮，然后选择**swig**文件夹（注意不要选择macswig文件夹），选择**swig-3.0.12**，下载[swig-3.0.12.tar.gz](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz/download)压缩包。

使用以下命令或者双击解压压缩包：

```bash
tar -zxvf swig-3.0.12.tar.gz
```

接下来，下载pcre。地址：https://ftp.pcre.org/pub/pcre/ 。请下载[pcre-8.44.tar.bz2](https://ftp.pcre.org/pub/pcre/pcre-8.44.tar.bz2)。

下载后，请将文件**重命名**为**‘pcre-8.44.tar’**，然后放入解压后的**swig-3.0.12文件夹**。

使用命令行，切换到swig-3.0.12文件夹下，依次执行命令：

```bash
./autogen.sh 
./configure 
make 
sudo make install
```

### 1.3 Linux系统安装SWIG

请使用以下命令安装SWIG：

```bash
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

## 2 安装OpenBox

在所有系统下均可使用以下方法安装OpenBox。（请注意，如果您使用conda，需要先切换至对应的环境）

### 2.1 通过PyPI安装（推荐）

使用以下命令通过PyPI安装OpenBox：

```bash
pip install openbox
```

请确保所有依赖与系统安装成功。

### 2.2 通过源码安装

使用以下命令通过Github源码安装OpenBox：

```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## 3 测试样例

### 3.1 Linux, macOS系统测试样例

您可以运行以下代码，测试OpenBox是否安装成功：

```python
import numpy as np
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter
from openbox.optimizer.generic_smbo import SMBO

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y

# Run
bo = SMBO(branin, config_space, max_runs=50, task_id='quick_start')
history = bo.run()
print(history)
```

运行结束后，将输出类似如下结果：

```python
+---------------------------------------------+
| Parameters              | Optimal Value     |
+-------------------------+-------------------+
| x1                      | -3.138277         |
| x2                      | 12.254526         |
+-------------------------+-------------------+
| Optimal Objective Value | 0.398096578033325 |
+-------------------------+-------------------+
| Num Configs             | 50                |
+-------------------------+-------------------+
```

恭喜您已完成安装并通过测试！

### 3.2 Windows 10系统测试样例

对于Windows 10系统，由于一些特殊原因，您可以运行以下代码，测试OpenBox是否安装成功。需要注意：
+ 需要在目标函数定义内部import依赖的包
+ 调用SMBO.run()需要在`if __name__ == '__main__':`语句内部

```python
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter
from openbox.optimizer.generic_smbo import SMBO

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])

# Define Objective Function
def branin(config):
    import numpy as np  # for Windows user, please import related packages in objective function
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y

# Run
if __name__ == '__main__':  # for Windows user, this line is necessary
    bo = SMBO(branin, config_space, max_runs=50, task_id='quick_start')
    history = bo.run()
    print(history)
```

运行结束后，将输出类似如下结果：

```python
+---------------------------------------------+
| Parameters              | Optimal Value     |
+-------------------------+-------------------+
| x1                      | -3.138277         |
| x2                      | 12.254526         |
+-------------------------+-------------------+
| Optimal Objective Value | 0.398096578033325 |
+-------------------------+-------------------+
| Num Configs             | 50                |
+-------------------------+-------------------+
```

恭喜您已完成安装并通过测试！

## 4 额外说明

### macOS系统安装pyrfr库失败解决方法

首先，从PyPI下载pyrfr源码。访问<https://pypi.org/project/pyrfr/#files>并下载[pyrfr-0.8.2.tar.gz](https://files.pythonhosted.org/packages/74/5f/3b2dd73fea58c5c893ae10156b5e135706b4136b810c1e0cf5fe089f944b/pyrfr-0.8.2.tar.gz)。双击压缩包或使用以下命令解压：

```bash
tar -zxvf pyrfr-0.8.2.tar.gz
```

修改`setup.py`文件中的`extra_compile_args`如下：

```python
extra_compile_args = ['-O2', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
```

最后，运行以下命令：

```bash
env CC="/usr/bin/gcc -stdlib=libc++ -mmacosx-version-min=10.7" pip install .
```

