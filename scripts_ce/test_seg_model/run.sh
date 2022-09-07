#!/bin/bash
ci_bt=`date '+%Y%m%d%H%M%S'`
CURRENT_DIR=$(cd $(dirname $0); pwd)
export no_proxy=bcebos.com
export py_version=python
export MODEL_PATH=${CURRENT_DIR}/../models
export DATA_PATH=${CURRENT_DIR}/../data
export TOOLS_PATH=${CURRENT_DIR}/../tools
cd ${MODEL_PATH}
rm -rf ./*
cd ${TOOLS_PATH}
$py_version ${TOOLS_PATH}/download_models.py
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}
rm -rf result.txt ./infer_result/*
cases=`find ./ -name "test*.py" | sort`
echo $cases
ignore=""
       
bug=0

job_bt=`date '+%Y%m%d%H%M%S'`
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        $py_version -m pytest --disable-warnings -sv ${file}
        if [[ $? -ne 0 && $? -ne 5 ]]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done
job_et=`date '+%Y%m%d%H%M%S'`

echo "total bugs: "${bug} >> result.txt
#if [ ${bug} != 0 ]; then
#    cp result.txt ${output_dir}/result_${py_version}.txt
#fi
cat result.txt
total_cost=$(expr $job_et - $ci_bt)
case_cost=$(expr $job_et - $job_bt)
echo "case_cost: $case_cost s"
echo "total_cost: $total_cost s"
exit ${bug}
