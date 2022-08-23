#!/bin/bash

CURRENT_DIR=$(cd $(dirname $0); pwd)
AGILE_COMPILE_BRANCH=$1
AGILE_PULL_ID=$2
export CODE_PATH=${CURRENT_DIR}/../code
export fastdeploy_dir=${CODE_PATH}/FastDeploy
echo ${fastdeploy_dir}


rm -rf ${fastdeploy_dir}
if [ ! -d ${fastdeploy_dir} ]; then
    cd ${CODE_PATH}
    git clone https://github.com/PaddlePaddle/FastDeploy.git -b develop
    git checkout ${AGILE_COMPILE_BRANCH}
    git fetch origin pull/${AGILE_PULL_ID}/head:pr_${AGILE_PULL_ID}
    git checkout pr_${AGILE_PULL_ID}
    git merge ${AGILE_COMPILE_BRANCH}
fi

cd ${fastdeploy_dir}

function install() {
    unset http_proxy
    unset https_proxy
    export WITH_GPU=ON
    export ENABLE_ORT_BACKEND=ON
    export ENABLE_PADDLE_BACKEND=ON
    export ENABLE_TRT_BACKEND=ON
    export CUDA_DIRECTORY=/usr/local/cuda-11.2
    export TRT_DIRECTORY=${CURRENT_DIR}/../thirdparty/TensorRT-8.4.1.5
    export ENABLE_VISION=ON
    export ENABLE_VISION_VISUALIZE=ON
    export ENABLE_PADDLE_FRONTEND=ON
    export ENABLE_DEBUG=ON
    export ENABLE_TEXT=ON
    export ENABLE_FDTENSOR_FUNC=ON
    export WITH_TESTING=ON
    $py_version -m pip install -r requirements.txt
    $py_version setup.py build
    $py_version setup.py bdist_wheel
    $py_version -m pip install --upgrade --no-deps --force-reinstall ./dist/*.whl
}

install
cd ${CURRENT_DIR}
