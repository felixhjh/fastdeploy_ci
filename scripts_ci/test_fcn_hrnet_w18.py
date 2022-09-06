from segmentation_test_case_base import *

class TestFCN_HRNet_W18Test(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="cityscapes", model_dir_name="FCN_HRNet_W18_cityscapes_without_argmax_infer", model_name="fcn_hrnet_w18", csv_path="./infer_result/fcn_hrnet_w18_result.csv")
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 112, 112], [1, 3, 512, 1024], [1, 3, 1024, 2048])


