#!/bin/bash
ce_bt=`date '+%s'`
set -m
CURRENT_DIR=$(cd $(dirname $0); pwd)
export no_proxy=bcebos.com
export py_version=python
export MODEL_PATH=${CURRENT_DIR}/../models
export DATA_PATH=${CURRENT_DIR}/../data
export TOOLS_PATH=${CURRENT_DIR}/../tools
export HOME=${CURRENT_DIR}
cd ${MODEL_PATH}
rm -rf ./*
cd ${TOOLS_PATH}
$py_version ${TOOLS_PATH}/download_models.py
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}
run_dirs=(test_det_model test_class_model)
if [[ -z $1 ]];then
    card_number=1
else
    card_number=($1)
fi
case_number=${#run_dirs[@]}
#TODO后续动态获取显卡梳理
#card_number=$(nvidia-smi -L | wc -l)
step=${#card_number[@]}
EXIT_CODE=0;
function caught_error() {
    for job in `jobs -p`; do
        echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?";
            let EXIT_CODE=EXIT_CODE+1;
        fi
    done
}
#trap 'caught_error' CHLD
for((i=0;i<case_number;i+=step))
do
    trap 'caught_error' CHLD
    for((j=0;j<step;j++))
    do
        if [[ -z "${run_dirs[i+j]}" ]];then
            break
        else
           echo "${run_dirs[i+j]}"
           cd $CURRENT_DIR/${run_dirs[i+j]}/ && bash run.sh ${card_number[j]}  2>&1 &
        fi
    done
    wait
done
#展示结果并设置返回值
cd $CURRENT_DIR
echo "show me the result"
find . -name result.txt | xargs cat
ce_et=`date '+%s'`
total_cost=$(expr $ce_et - $ce_bt)
echo "total_cost: $total_cost s"
#TODO 展示错误case
if [[ $EXIT_CODE -gt 0 ]];
then
     exit $EXIT_CODE
else
     exit 0
fi
