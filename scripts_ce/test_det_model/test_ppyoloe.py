import sys
sys.path.append("..")
from util import *
import fastdeploy as fd
import os

class TestPPYoloeTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="ppyoloe_crn_l_300e_coco", model_name="ppyoloe_crn_l_300e", csv_path="./infer_result/ppyoloe_result.csv")
        self.pdiparams = os.path.join(self.util.model_path, "model.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "model.pdmodel")
        self.yaml_file = os.path.join(self.util.model_path, "infer_cfg.yml")
        self.image_file_path = os.path.join(self.util.data_path, "val2017")
        self.annotation_file_path = os.path.join(self.util.data_path, "annotations/instances_val2017.json")
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        self.diff = 0.0005
        
    def teardown_method(self):
        pass
    
    def test_ort_cpu(self):
        self.option.use_ort_backend()
        self.option.use_cpu()
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_ort_cpu", model_name=self.model_name, delta=self.diff, csv_path=self.csv_save_path)

    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_ort_gpu", model_name=self.model_name, delta=self.diff, csv_path=self.csv_save_path)


    def test_trt(self):
        self.option.use_trt_backend()
        self.option.use_gpu(0)
        self.option.set_trt_input_shape("image", [1, 3, 640, 640])
        self.option.set_trt_input_shape("scale_factor", [1, 2])
        model = fd.vision.detection.PPYOLOE(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = fd.vision.evaluation.eval_detection(model, self.image_file_path, self.annotation_file_path)
        check_result(result, self.util.ground_truth, case_name="test_trt", model_name=self.model_name, delta=self.diff, csv_path=self.csv_save_path)

