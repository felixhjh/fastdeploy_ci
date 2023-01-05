from util import *
import fastdeploy as fd
import os
import pytest
TEST_NNADAPTER=os.getenv("TEST_NNADAPTER", "OFF")
class CaseBase(object):

    #temp skip ghostnet_x0_5 mobilenetv2_x0_25 shufflenetv2_x2_0
    #cases = ["ghostnet_x0_5", "mobilenetv2_x0_25", "shufflenetv2_x2_0"]
    cases = []

    def set_trt_info(self):
        pass

    def teardown_method(self):
        #print_log(["stderr.log", "stdout.log"], iden="after predict")
        pass
    
    def setup_class(self):
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        self.pdiparams = os.path.join(self.util.model_path, "inference.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "inference.pdmodel")
        self.yaml_file = os.path.join(self.util.model_path, "inference_cls.yaml")
        self.image_file_path = self.util.data_path
        self.label_file_path = os.path.join(self.util.data_path, "val_list_50.txt")
        self.option = fd.RuntimeOption()
        

    def run_predict(self):
        model = fd.vision.classification.PaddleClasModel(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = {}
        tok1_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=1)
        result.update(tok1_result)
        tok5_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=5)
        result.update(tok5_result)
        return result

    @pytest.mark.skipif(TEST_NNADAPTER=="OFF", reason="Test NNADAPTER is OFF.")
    def test_nnadapter(self):
        getattr(self.option, TEST_NNADAPTER)()
        ret = check_result(result, self.util.ground_truth, "test_nnadapter", self.model_name, 0, self.csv_save_path)

    @pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
    def test_ort_cpu(self):
        self.option.use_ort_backend()
        self.option.use_cpu()
        result = self.run_predict()
        ret = check_result(result, self.util.ground_truth, "test_ort_cpu", self.model_name, 0, self.csv_save_path)

    @pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)
        result = self.run_predict()
        check_result(result, self.util.ground_truth, "test_ort_gpu", self.model_name, 0,self.csv_save_path)
    
    @pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
    def test_paddle_cpu_backend(self):
        self.option.use_paddle_backend()
        self.option.use_cpu()
        result = self.run_predict()
        check_result(result, self.util.ground_truth, "test_paddle_cpu_backend", self.model_name, 0,self.csv_save_path)

    @pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
    def test_paddle_gpu_backend(self):
        if self.model_name in CaseBase.cases:
            return
        self.option.use_paddle_backend()
        self.option.use_gpu(0)
        result = self.run_predict()
        check_result(result, self.util.ground_truth, "test_paddle_gpu_backend", self.model_name, 0, self.csv_save_path)

    @pytest.mark.skipif(TEST_NNADAPTER!="OFF", reason="Test NNADAPTER.")
    def test_trt(self):
        self.set_trt_info()
        result = self.run_predict()
        check_result(result, self.util.ground_truth, "test_trt", self.model_name, 0, self.csv_save_path)


