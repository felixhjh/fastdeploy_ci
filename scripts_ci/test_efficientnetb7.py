from classification_test_case_base import *

class TestEfficientNetB7Test(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="EfficientNetB7_infer", model_name="efficientnetb7", csv_path="./infer_result/efficientnetb7_result.csv")
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 600, 600], [1, 3, 600, 600], [1, 3, 600, 600])


