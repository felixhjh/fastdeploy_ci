from util import *
import fastdeploy as fd
import os
import pytest
TEST_KUNLUNXIN=os.getenv("TEST_KUNLUNXIN","OFF")

@pytest.mark.skipif(TEST_KUNLUNXIN=="ON", reason="test kunlunxin.")
class TestYolov5Test(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="", model_name="yolov5s", csv_path="./infer_result/yolov5s_result.csv")
        self.onnxmodel = os.path.join(self.util.model_path, "yolov5s.onnx")
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
        model = fd.vision.detection.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_ort_cpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_ort_gpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)


    def test_trt(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("images", [1, 3, 320, 320], [1, 3, 640, 640], [1, 3, 1280, 1280])
        model = fd.vision.detection.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_trt", model_name=self.model_name, delta=1e-06, csv_path=self.csv_save_path)

@pytest.mark.skipif(TEST_KUNLUNXIN=="OFF", reason="test kunlunxin.")
class TestYolov5sKunlunXinTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="yolov5s_infer", model_name="yolov5s", csv_path="./infer_result/yolov5s_result.csv")
        self.pdiparams = os.path.join(self.util.model_path, "model.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "model.pdmodel")
        self.image_file_path = os.path.join(self.util.data_path, "val2017_50")
        self.annotation_file_path = os.path.join(self.util.data_path, "annotations/instances_val2017_50.json")
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        pass
    
    def test_KunlunXin(self):
        option = fd.RuntimeOption()
        option.use_kunlunxin()
        model = fd.vision.detection.YOLOv6(self.pdmodel, self.pdiparams, runtime_option=option, model_format=fd.ModelFormat.PADDLE)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path, 0.001, 0.65)
        check_result(result, self.util.ground_truth, case_name="test_KunlunXin", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)


