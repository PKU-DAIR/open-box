# SWIG Installation Guide

This tutorial helps you install SWIG 3.0.

## Linux

```bash
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

## MacOS

1. Download SWIG from [SWIG Download Page](http://www.swig.org/download.html).
Click on **All releases** for older releases. Then select the **'swig'** folder. 
We recommend to install version 3.0.12. Unzip the files after downloading.

2. Download pcre from [here](http://www.pcre.org).
Download 'pcre-8.44.tar.bz2' and rename it as 'pcre-8.44.tar'.
Then put it into the SWIG folder.

3. Run the following commands under the SWIG folder:

```bash
./Tools/pcre-build.sh
./autogen.sh
./configure
make
sudo make install
```

## Windows

1. Download SWIG for Windows from [SWIG Download Page](http://www.swig.org/download.html).
Clicking on **All releases** for older releases. Then select the **'swigwin'** folder.
We recommend to install version 3.0.12. Unzip the files after downloading.

2. Add the SWIG folder to SYSTEM PATH.
