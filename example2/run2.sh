#!/bin/bash
today=`date +%Y-%m-%d.%H:%M:%S`
touch /doccano/$today.log
python /doccano/save_labels.py 1,3 2>&1 | tee -a /doccano/$today.log
python /doccano/merge_labels.py 2>&1 | tee -a /doccano/$today.log
python /doccano/train2.py 2>&1 | tee -a /doccano/$today.log
python /doccano/update_labels.py 2>&1 | tee -a /doccano/$today.log
python /doccano/predict2.py 2>&1 | tee -a /doccano/$today.log
python /doccano/update_db.py 2>&1 | tee -a /doccano/$today.log

