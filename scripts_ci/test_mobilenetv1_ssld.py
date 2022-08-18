from classification_test_case_base import *

class TestMobileNetV1_ssldTest(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="MobileNetV1_ssld_infer", model_name="mobilenetv1_ssld", csv_path="./infer_result/mobilenetv1_ssld_result.csv")
        super().setup_class(self)
        
    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("inputs", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])


