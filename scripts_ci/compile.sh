#!/bin/bash

CURRENT_DIR=$(cd $(dirname $0); pwd)
export CODE_PATH=../${CURRENT_DIR}/code
export fastdeploy_dir=${CODE_PATH}/FastDeploy
CURRENT_DIR=$(cd $(dirname $0); pwd)
echo ${fastdeploy_dir}


rm -rf ${fastdeploy_dir}
if [ ! -d ${fastdeploy_dir} ]; then
    cd ${CODE_PATH}
    git clone https://github.com/PaddlePaddle/FastDeploy.git -b develop
fi

cd ${fastdeploy_dir}
git pull

function install() {
    unset http_proxy
    unset https_proxy
    export WITH_GPU=ON
    export ENABLE_ORT_BACKEND=ON
    export ENABLE_PADDLE_BACKEND=ON
    export ENABLE_TRT_BACKEND=ON
    export CUDA_DIRECTORY=/usr/local/cuda-11.2
    export TRT_DIRECTORY=../${CURRENT_DIR}/thirdparty/TensorRT-8.4.1.5
    export ENABLE_VISION=ON
    export ENABLE_VISION_VISUALIZE=ON
    export ENABLE_PADDLE_FRONTEND=ON
    export ENABLE_DEBUG=ON
    export ENABLE_TEXT=ON
    export ENABLE_FDTENSOR_FUNC=ON
    export WITH_TESTING=ON
    python setup.py build
    python setup.py bdist_wheel
    pip3 install --upgrade --no-deps --force-reinstall ./dist/*.whl
}

install
cd ${CURRENT_DIR}
