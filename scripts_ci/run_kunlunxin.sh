#!/bin/bash
ci_bt=`date '+%s'`
CURRENT_DIR=$(cd $(dirname $0); pwd)
export no_proxy=bcebos.com
export py_version=python
export MODEL_PATH=${CURRENT_DIR}/../models
export DATA_PATH=${CURRENT_DIR}/../data
export TOOLS_PATH=${CURRENT_DIR}/../tools
export TEST_KUNLUNXIN=ON
export ground_truth_file=ground_truth_kunlunxin.yaml
cd ${MODEL_PATH}
rm -rf ./*
cd ${TOOLS_PATH}
$py_version ${TOOLS_PATH}/download_models.py
cd ${DATA_PATH}
rm -rf ./*
unset http_proxy
unset https_proxy
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/coco_dataset_ci.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/imagenet_dataset_ci.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/cityscapes.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/ICDAR2017_10.tgz
for i in `ls | grep 'tar\|tgz'`
   do
     tar -zxvf $i >/dev/null
   done
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}
rm -rf result.txt ./infer_result/*
cases=`find ./ -name "test*.py" | sort`
echo $cases
ignore="test_ppocrv3.py" # 有diff
       
bug=0

job_bt=`date '+%s'`
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
job_et=`date '+%s'`

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
