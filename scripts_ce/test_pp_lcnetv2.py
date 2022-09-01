from classification_test_case_base import *

class TestPPLCNetV2Test(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="PPLCNetV2_base_infer", model_name="pp_lcnetv2_base", csv_path="./infer_result/pp_lcnetv2_base_result.csv")
        self.diff = 0.0001
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])
        


