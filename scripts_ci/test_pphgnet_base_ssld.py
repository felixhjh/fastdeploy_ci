from classification_test_case_base import *

class TestPPHGNet_base_ssldTest(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="PPHGNet_base_ssld_infer", model_name="pphgnet_base_ssld", csv_path="./infer_result/pphgnet_base_ssld_result.csv")
        super().setup_class(self)
        
    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])


