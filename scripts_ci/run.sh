#!/bin/bash
ci_bt=`date '+%s'`
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
cd ${DATA_PATH}
rm -rf ./*
unset http_proxy
unset https_proxy
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/coco_dataset_ci.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/imagenet_dataset_ci.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/cityscapes.tgz
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/ICDAR2017_10.tar
for i in `ls ./*.tgz`
   do
     tar -zxvf $i >/dev/null
   done
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}
rm -rf result.txt ./infer_result/*
cases=`find ./ -name "test*.py" | sort`
echo $cases
ignore="test_efficientnetb0_small.py
        test_efficientnetb7.py
        test_ghostnet_x1_3_ssld.py
        test_inceptionv3.py
        test_mobilenetv1_ssld.py
        test_mobilenetv1_x0_25.py
        test_mobilenetv2_ssld.py
        test_mobilenetv3_large_x1_0_ssld.py
        test_mobilenetv3_small_x0_35_ssld.py
        test_pphgnet_base_ssld.py
        test_pphgnet_tiny_ssld.py
        test_pp_lcnet.py
        test_pp_lcnetv2.py
        test_shufflenetv2_x0_25.py"
       
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
