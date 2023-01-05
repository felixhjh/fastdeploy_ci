from util import *
import fastdeploy as fd
import os
import pytest
TEST_NNADAPTER=os.getenv("TEST_NNADAPTER", "OFF")

@pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
class TestYoloRp6Test(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="", model_name="yolorp6", csv_path="./infer_result/yolorp6_result.csv")
        self.onnxmodel = os.path.join(self.util.model_path, "yolor-p6-paper-541-1280-1280.onnx")
        self.image_file_path = os.path.join(self.util.data_path, "val2017_50")
        self.annotation_file_path = os.path.join(self.util.data_path, "annotations/instances_val2017_50.json")
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        #print_log(["stderr.log", "stdout.log"], iden="after predict")
        pass
    
    def test_ort_cpu(self):
        self.option.use_ort_backend()
        self.option.use_cpu()
        model = fd.vision.detection.YOLOR(self.onnxmodel, "None", self.option)
        model.size = [1280, 1280]
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_ort_cpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.YOLOR(self.onnxmodel, "None", self.option)
        model.size = [1280, 1280]
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_ort_gpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)


    def test_trt(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("images", [1, 3, 1280, 1280])
        model = fd.vision.detection.YOLOR(self.onnxmodel, "None", self.option)
        model.size = [1280, 1280]
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_trt", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

