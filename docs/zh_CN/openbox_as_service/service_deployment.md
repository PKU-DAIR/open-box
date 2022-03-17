# OpenBox 服务部署

本教程帮助你部署一个 **OpenBox** 服务。 
如果你是一个 **OpenBox** 服务的用户，请参考 [服务教程](./service_tutorial)


## 1 从源代码安装 OpenBox

安装要求：

+ Python >= 3.6

+ SWIG == 3.0

关于 SWIG的安装，请参考 [SWIG 安装指南](../installation/install_swig.md)

在安装OpenBox前，请保证SWIG已经被成功安装。

接下来，将OpenBox的源代码clone到本地。命令如下：

```bash
git clone https://github.com/PKU-DAIR/open-box.git
cd open-box
python setup.py install
```


## 2 初始化 MongoDB

**OpenBox** 使用 [MongoDB](https://www.mongodb.com) 来存储用户信息和任务信息。

### 2.1 安装和运行 MongoDB

请在运行**OpenBox**服务前先安装和运行MongoDB。
对于MongoDB的安装指南，请参考以下链接：


+ <https://docs.mongodb.com/guides/server/install/>

+ <https://docs.mongodb.com/manual/installation/>

你需要先创建一个 MongoDB 用户，在启动 MongoDB 时设置 **auth=true** 。
请记住你数据库的 **IP** 和 **端口号**。 

### 2.2 修改 service.conf 文件

在安装了MongoDB后，修改 <font color=#FF0000>**"open-box/conf/service.conf"**</font> 来设置数据库信息。
如果这是你第一次运行，创建 <font color=#FF0000>**service.conf**</font> 文件：将模版配置文件从 <font color=#FF0000>**"open-box/conf/template/service.conf.template"**</font> 复制到 
<font color=#FF0000>**"open-box/conf/"**</font> 并重命名为 <font color=#FF0000>**service.conf**</font>。

**service.conf** 的内容如下：

```
[database]
database_address=127.0.0.1
database_port=27017
user=xxxx
password=xxxx
```

请根据提示设置你数据库的IP，端口号，用户名和密码。

**注意：** 我们已经把 **service.conf** 添加到了 **.gitignore**。
不要把这个文件push到 **Github** ，否则会泄漏你的隐私信息。


## 3 设置 Email 注册服务

### 3.1 准备工作

**OpenBox**需要一个电子邮件地址，以便在用户注册新帐户时发送激活链接。
请开启SMTP身份验证，然后您可能会从电子邮件提供商处收到用于身份验证的密钥。

### 3.2 修改 openbox/artifact/artifact/settings.py

接下来，修改 <font color=#FF0000>**"openbox/artifact/artifact/settings.py"**</font> 来为注册服务设置邮件信息。
请填写以下内容：

```python
EMAIL_HOST = 'smtp.xxxx.com'
EMAIL_PORT = 465
EMAIL_HOST_USER = 'xxxx@xxxx.com'
EMAIL_HOST_PASSWORD = 'xxxx'
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER
```

+ **EMAIL_HOST:** 邮件注册服务提供商的 SMTP 主机。 例如，'smtp.gmail.com'。
+ **EMAIL_PORT:** 邮件注册服务提供商的 SMTP 端口号。 从邮件注册服务提供商获取端口号。 如果 465 没能奏效，你可以再试一下 25，587，或者其它端口号。
+ **EMAIL_HOST_USER:** 你注册服务的邮件地址。
+ **EMAIL_HOST_PASSWORD:** 你 SMTP 验证的密钥。

**注意：** 不要把带有你隐私信息的文件push到 **Github** 。

## 4 开始/停止 OpenBox 服务

最后，在设置好数据库和注册服务后，你可以开始 **OpenBox** 服务。

通过下列命令运行 <font color=#FF0000>**manage_service.sh script**</font> 来开始服务：

```bash
cd <path to the source code>/open-box
./scripts/manage_service.sh start
```

这个脚本会在后台运行 **OpenBox**。
默认的服务端口号是11425。
你可以修改脚本来改变服务端口号。

接下来，访问 <http://127.0.0.1:11425/user_board/index/> (用你自己的 ip:port 替换 "127.0.0.1:11425")
来看是否你的服务已经成功开始了。
你也可以创建一个账户，运行一个任务来测试你的**OpenBox**服务。
请参考 [用户指南](./service_tutorial) 以获得详细的指南。

通过下列命令运行 <font color=#FF0000>**manage_service.sh script**</font> 来停止服务：

```bash
cd <path to the source code>/open-box
./scripts/manage_service.sh stop
```

