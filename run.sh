#!/bin/bash
today=`date +%Y-%m-%d.%H:%M:%S`
touch /doccano/$today.log
python /doccano/save_labels.py 1 2>&1 | tee -a /doccano/$today.log
python /doccano/train.py 2>&1 | tee -a /doccano/$today.log
python /doccano/predict.py 2>&1 | tee -a /doccano/$today.log
python /doccano/update_db.py 2>&1 | tee -a /doccano/$today.log

