from classification_test_case_base import *

class TestGhostNet_x0_5Test(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="GhostNet_x0_5_infer", model_name="ghostnet_x0_5", csv_path="./infer_result/ghostnet_x0_5_result.csv")
        self.diff = 0.0002
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("inputs", [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])


