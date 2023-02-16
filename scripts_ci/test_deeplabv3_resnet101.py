from segmentation_test_case_base import *

class TestDeeplabv3_ResNet101_OS8Test(CaseBase):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="cityscapes", model_dir_name="Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer", model_name="deeplabv3_resnet101", csv_path="./infer_result/deeplabv3_resnet101_result.csv")
        super().setup_class(self)

    def set_trt_info(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("x", [1, 3, 112, 112], [1, 3, 512, 1024], [1, 3, 1024, 2048])

    @pytest.mark.skipif(TEST_NNADAPTER=="OFF", reason="Test NNADAPTER is OFF.")
    def test_nnadapter(self):
        getattr(self.option, TEST_NNADAPTER)()
        result = self.run_predict()
        ret = check_result(result, self.util.ground_truth, "test_nnadapter", self.model_name, 1e-2, self.csv_save_path)


