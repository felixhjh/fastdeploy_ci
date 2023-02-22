from util import *
import fastdeploy as fd
import os
import pytest

TEST_NNADAPTER=os.getenv("TEST_NNADAPTER", "OFF")
@pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
class TestPicoDetTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="picodet_l_320_coco_lcnet", model_name="picodet_l_320_coco_lcnet", csv_path="./infer_result/picodet_result.csv")
        self.pdiparams = os.path.join(self.util.model_path, "model.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "model.pdmodel")
        self.yaml_file = os.path.join(self.util.model_path, "infer_cfg.yml")
        self.image_file_path = os.path.join(self.util.data_path, "val2017_50")
        self.annotation_file_path = os.path.join(self.util.data_path, "annotations/instances_val2017_50.json")
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        pass
    
    def test_openvino_cpu(self):
        self.option.use_openvino_backend()
        self.option.use_cpu()
        model = fd.vision.detection.PicoDet(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_openvino_cpu", model_name=self.model_name, delta=0.001, csv_path=self.csv_save_path)

    def test_paddle_gpu(self):
        self.option.use_paddle_backend()
        self.option.use_gpu()
        model = fd.vision.detection.PicoDet(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_paddle_gpu", model_name=self.model_name, delta=0.04, csv_path=self.csv_save_path)

    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.PicoDet(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_ort_gpu", model_name=self.model_name, delta=0.001, csv_path=self.csv_save_path)


    def test_trt(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.PicoDet(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_trt", model_name=self.model_name, delta=0.001, csv_path=self.csv_save_path)

@pytest.mark.skipif(TEST_NNADAPTER=="OFF", reason="Test NNADAPTER is OFF.")
class TestPicoDetNNADAPTERTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="picodet_l_320_coco_lcnet", model_name="picodet_l_320_coco_lcnet", csv_path="./infer_result/picodet_result.csv")
        self.pdiparams = os.path.join(self.util.model_path, "model.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "model.pdmodel")
        self.yaml_file = os.path.join(self.util.model_path, "infer_cfg.yml")
        self.image_file_path = os.path.join(self.util.data_path, "val2017_50")
        self.annotation_file_path = os.path.join(self.util.data_path, "annotations/instances_val2017_50.json")
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        pass
    
    def test_nnadapter(self):
        getattr(self.option, TEST_NNADAPTER)()
        model = fd.vision.detection.PicoDet(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_nnadapter", model_name=self.model_name, delta=0.001, csv_path=self.csv_save_path)
