from classification_test_case_base import *

class TestEfficientNetB0SmallTest(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="EfficientNetB0_small_infer", model_name="efficientnetb0small", csv_path="./infer_result/efficientnetb0small_result.csv")
        super().setup_class(self)
        
    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("inputs", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])



