#!/bin/bash

sleep 3
log="/home/nl/ip_test_log.log"
configFile="/home/nl/XYHL_config.ini">>$log
path="/home/nl/by/UITest/"
clear




ReadINIfile()
{   
	Key=$1
	Section=$2
  	Configfile=$3
	ReadINI=`awk -F '=' '/\['$Section'\]/{a=1}a==1&&$1~/'$Key'/{print $2;exit}' $Configfile`  
 	echo "$ReadINI"  
} 

local_ip=`ReadINIfile "local_ip" "XYHL_config" "$configFile"`
rsu_ip=`ReadINIfile "rsu_ip" "XYHL_config" "$configFile"`
cam_ip=`ReadINIfile "cam_ip" "XYHL_config" "$configFile"`

var=`date +----------%Y-%m-%d,%H:%M:%S----------`
echo $var>>$log


if ping -c1 $local_ip>>$log
then
local_ip_con=1
echo "local_ip $local_ip is online">>$log
else
local_ip_con=0
echo "local_ip $local_ip is offline">>$log
fi

if ping -c1 $rsu_ip>>$log
then
echo "rsu_ip $rsu_ip is online">>$log
rsu_ip_con=1
else
echo "rsu_ip $rsu_ip is offline">>$log
rsu_ip_con=0
fi

if ping -c1 $cam_ip>>$log
then
echo "cam_ip $cam_ip is online">>$log
cam_ip_con=1
else
echo "cam_ip $cam_ip is offline">>$log
cam_ip_con=0
fi


if [ $local_ip_con -eq 1 ]
then
echo "local_ip $local_ip is online">>$log
else
echo "local_ip $local_ip is offline">>$log
fi

if [ $rsu_ip_con -eq 1 ]
then
echo "rsu_ip $rsu_ip is online">>$log
else
echo "rsu_ip $rsu_ip is offline">>$log
fi

if [ $cam_ip_con -eq 1 ]
then
echo "cam_ip $cam_ip is online">>$log
else
echo "cam_ip $cam_ip is offline">>$log
fi

ip_valid=`expr $local_ip_con + $rsu_ip_con + $cam_ip_con`

if [ $ip_valid -eq 3 ]
then
echo "all ip is connected">>$log
else
echo "the ConfigFile need to be Modified">>$log
fi

cd $path>>$log
/home/nl/anaconda3/bin/python3.6 Run.py>>$log



