#!/bin/bash
[[ -n $1 ]] && export CUDA_VISIBLE_DEVICES=3
clas_ce_bt=`date '+%s'`
CLAS_CURRENT_DIR=$(cd $(dirname $0); pwd)
export no_proxy=bcebos.com
export py_version=python
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}
rm -rf result.txt ./infer_result/*
cases=`find $CLAS_CURRENT_DIR -name "test*.py" | sort`
echo $cases
ignore=""
       
bug=0

echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        $py_version -m pytest -n auto --disable-warnings -sv ${file}
        if [[ $? -ne 0 && $? -ne 5 ]]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done
clas_ce_et=`date '+%s'`

echo "total bugs: "${bug} >> result.txt
cat result.txt
clas_total_cost=$(expr $clas_ce_et - $clas_ce_bt)
echo "total_cost: $clas_total_cost s"
exit ${bug}
