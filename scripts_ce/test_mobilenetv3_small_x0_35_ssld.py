from classification_test_case_base import *

class TestMobileNetV3_small_x0_35_ssldTest(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="MobileNetV3_small_x0_35_ssld_infer", model_name="mobilenetv3_small_x0_35_ssld", csv_path="./infer_result/mobilenetv3_small_x0_35_ssld_result.csv")
        self.diff = 0.0001
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])


