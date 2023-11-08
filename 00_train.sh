#!/bin/sh
 
now="$(date +'%Y%m%d')"
echo "$now"
exp_forder=experiments/exp
destination=${exp_forder=experiments/exp}"_"${now}
echo "$destination"
if [ -d $destination ]; then
   # 目錄 /path/to/dir 存在
   echo $destination "exists."
else

   # 目錄 /path/to/dir 不存在
   echo $destination " does not exists."
   mkdir $destination
fi

cp *.py $destination/
cp -r model/ $destination/

echo " cp all files to " $destination
cd $destination
python ai_main.py
