from classification_test_case_base import *

class TestPPHGNet_tiny_ssldTest(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="PPHGNet_tiny_ssld_infer", model_name="pphgnet_tiny_ssld", csv_path="./infer_result/pphgnet_tiny_ssld_result.csv")
        self.diff = 0.004
        super().setup_class(self)
        
    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])


