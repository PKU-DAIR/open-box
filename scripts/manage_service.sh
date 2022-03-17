#!/bin/bash
case $1 in                                        
   migrate) cd `dirname $0`/..; python openbox/artifact/manage.py migrate;;
   start) cd `dirname $0`/..; nohup python openbox/artifact/manage.py runserver 0.0.0.0:11425 &;;
   stop) netstat -nlpt | grep 11425 | awk -F ' ' '{print $7}' | cut -d / -f 1 | xargs kill -9;;
   *) echo "require migrate|start|stop" ;;
esac
