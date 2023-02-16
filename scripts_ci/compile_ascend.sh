#!/bin/bash

CURRENT_DIR=$(cd $(dirname $0); pwd)
export CODE_PATH=${CURRENT_DIR}/../code
export fastdeploy_dir=${CODE_PATH}/FastDeploy
echo ${fastdeploy_dir}


rm -rf ${fastdeploy_dir}
if [ ! -d ${fastdeploy_dir} ]; then
    cd ${CODE_PATH}
    git clone https://github.com/PaddlePaddle/FastDeploy.git -b develop
fi

cd ${fastdeploy_dir}

$py_version -m pip install -r requirements.txt
export no_proxy=bcebos.com
export WITH_ASCEND=ON
export ENABLE_VISION=ON
export ENABLE_VISION_VISUALIZE=ON
export ENABLE_PADDLE_FRONTEND=ON
export ENABLE_DEBUG=ON
export ENABLE_TEXT=ON
export ENABLE_FDTENSOR_FUNC=ON
export WITH_TESTING=ON
$py_version setup.py build
$py_version setup.py bdist_wheel
$py_version -m pip install --upgrade --no-deps --force-reinstall ./dist/*.whl

cd ${CURRENT_DIR}