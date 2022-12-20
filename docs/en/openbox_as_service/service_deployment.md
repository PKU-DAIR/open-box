# OpenBox Service Deployment

This tutorial helps you deploy an **OpenBox** service. If you are an **OpenBox** service user, please refer to 
the [Service User Tutorial](./service_tutorial).


## 1 Install OpenBox from Source

Installation Requirements:

+ Python >= 3.7

+ SWIG == 3.0

To install SWIG, please refer to {ref}`SWIG Installation Guide <installation/install_swig:swig installation guide>`.

Make sure that SWIG is installed correctly installing OpenBox.

Then we recommend you to update your `pip`, `setuptools` and `wheel` as follows:
```bash
pip install --upgrade pip setuptools wheel
```

Next, clone the source code to the server where you want to deploy OpenBox service,
and install OpenBox from source code. The commands are as follows:

```bash
git clone https://github.com/PKU-DAIR/open-box.git
cd open-box
pip install ".[service]"
```


## 2 Initialize MongoDB

**OpenBox** uses [MongoDB](https://www.mongodb.com) to store user and task information.

### 2.1 Install and Run MongoDB

Please install and run MongoDB before running **OpenBox** service. 
For MongoDB installation guides, refer to the following links:

+ <https://docs.mongodb.com/guides/server/install/>

+ <https://docs.mongodb.com/manual/installation/>

You need to create a MongoDB user and set **auth=true** when starting MongoDB.
Please record the **IP** and **port** of your database.

### 2.2 Modify service.conf File

After starting MongoDB, modify <font color=#FF0000>**"open-box/conf/service.conf"**</font> to set database information.
If this is your first time running the service, create the <font color=#FF0000>**service.conf**</font> file by copying 
the template config file from <font color=#FF0000>**"open-box/conf/template/service.conf.template"**</font> to 
<font color=#FF0000>**"open-box/conf/"**</font> and rename it to <font color=#FF0000>**service.conf**</font>.

The contents of **service.conf** are as follows:

```
[database]
database_address=127.0.0.1
database_port=27017
user=xxxx
password=xxxx
```

Please set database IP, port, user and password accordingly.

**Caution:** We have added **service.conf** to **.gitignore**. Do not push this file to **Github** to
prevent the disclosure of private information.


## 3 Set up Email Registration Service

### 3.1 Prepare an Email for Registration Service

**OpenBox** requires an email address to send activation link when users register new accounts. 
Please enable SMTP authentication and then you may receive a secret key for authentication from the email provider.

### 3.2 Modify openbox/artifact/artifact/settings.py

Then, modify <font color=#FF0000>**"openbox/artifact/artifact/settings.py"**</font> to set email information for
registration service. Please fill in the following lines:

```python
EMAIL_HOST = 'smtp.xxxx.com'
EMAIL_PORT = 465
EMAIL_HOST_USER = 'xxxx@xxxx.com'
EMAIL_HOST_PASSWORD = 'xxxx'
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER
EMAIL_ACTIVE_ENABLE = False
```

+ **EMAIL_HOST:** SMTP host of email registration service provider. E.g., 'smtp.gmail.com'.
+ **EMAIL_PORT:** SMTP port of email registration service provider. Get the port from email service provider. You may try port 25,587 
or other ports if port 465 doesn't work.
+ **EMAIL_HOST_USER:** Your email address for registration service.
+ **EMAIL_HOST_PASSWORD:** Your secret key for SMTP authentication.
+ **EMAIL_ACTIVE_ENABLE:** Enable or disable the mail activation function.

**Caution:** Do not push the file with private information to **Github** to prevent leakage.

## 4 Migrate Database
```bash
cd <path to the source code>/open-box
./scripts/manage_service.sh migrate
```

## 5 Start/Stop OpenBox Service

Finally, after setting up the database and registration service, you can start up the **OpenBox** service.

To **start the service**, run the <font color=#FF0000>**manage_service.sh script**</font> by the following commands:

```bash
cd <path to the source code>/open-box
./scripts/manage_service.sh start
```

The script will run **OpenBox** service in the background. The default service port is 11425.
You can modify the script to change service port.

Then, visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" with your server ip:port)
to see whether your service starts successfully.
You may also try to create an account and run a task to test your **OpenBox** service. 
For more detailed guidance, please refer to the [Service User Tutorial](./service_tutorial).

To **stop the service**, run the <font color=#FF0000>**manage_service.sh script**</font> by the following commands:

```bash
cd <path to the source code>/open-box
./scripts/manage_service.sh stop
```
