from util import *
import fastdeploy as fd
import os
import pytest
TEST_KUNLUNXIN=os.getenv("TEST_KUNLUNXIN","OFF")

class TestPPYoloeTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="ppyoloe_crn_l_300e_coco", model_name="ppyoloe_crn_l_300e", csv_path="./infer_result/ppyoloe_result.csv")
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
    
    @pytest.mark.skipif(TEST_KUNLUNXIN=="OFF", reason="test kunlunxin.")
    def test_kunlunxin(self):
        self.option.use_kunlunxin()
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_kunlunxin", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    @pytest.mark.skipif(TEST_KUNLUNXIN=="ON", reason="test kunlunxin.")
    def test_openvino_cpu(self):
        self.option.use_openvino_backend()
        self.option.use_cpu()
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_openvino_cpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    @pytest.mark.skipif(TEST_KUNLUNXIN=="ON", reason="test kunlunxin.")
    def test_paddle_gpu(self):
        self.option.use_paddle_backend()
        self.option.use_gpu()
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_paddle_gpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    @pytest.mark.skipif(TEST_KUNLUNXIN=="ON", reason="test kunlunxin.")
    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_ort_gpu", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

    @pytest.mark.skipif(TEST_KUNLUNXIN=="ON", reason="test kunlunxin.")
    def test_trt(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_trt", model_name=self.model_name, delta=0, csv_path=self.csv_save_path)

