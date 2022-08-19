# Fastdeploy CI 部署流程

## 编译选项配置
- 下载TensorRT到thirdparty目录（默认：TRT_DIRECTORY=thirdparty/TensorRT-8.4.1.5）
- 确保cuda安装在/usr/local目录下（默认：CUDA_DIRECTORY=/usr/local/cuda-11.2）
>> **注意**: 若需修改以上编译选项，请手动更改[compile.sh](scripts_ci/compile.sh)
             
## 运行CI
```
bash run.sh ${AGILE_COMPILE_BRANCH} ${AGILE_PULL_ID}
```
CI中的testcase共19个PaddleClas模型、1个PPYoloe模型、8个外部的Detection模型。默认CI运行5个PaddleClas模型、1个PPYoloe、8个外部的Detection模型
>> **注意**: 若需运行更多的PaddleCla模型，请手动更改[run.sh](scripts_ci/run.sh)中的ignore字段

若从git拉取代码过慢，可自行全局设置代理，CI代码中在不需要代理的地方已经unset代理
