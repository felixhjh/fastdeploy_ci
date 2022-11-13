from util import *
import fastdeploy as fd
import os
class TestPPOCRv3Test(object):
    def setup_class(self):

        ####
        self.util = FastdeployTest(data_dir_name="ICDAR2017", 
            model_dir_name="PPOCRv3_models", 
            model_name="PPOCRv3",
            csv_path="./infer_result/PPOCRv3_result.csv")

        # Det Model
        self.det_model_path = os.path.join(self.util.model_path, "ch_PP-OCRv3_det_infer")  
        self.det_pdiparams = os.path.join(self.det_model_path, "inference.pdiparams")
        self.det_pdmodel = os.path.join(self.det_model_path, "inference.pdmodel")
        # Cls Model
        self.cls_model_path = os.path.join(self.util.model_path, "ch_ppocr_mobile_v2.0_cls_infer") 
        self.cls_pdiparams = os.path.join(self.cls_model_path, "inference.pdiparams")
        self.cls_pdmodel = os.path.join(self.cls_model_path, "inference.pdmodel")

        # Rec Model
        self.rec_model_path = os.path.join(self.util.model_path, "ch_PP-OCRv3_rec_infer") 
        self.rec_pdiparams = os.path.join(self.rec_model_path, "inference.pdiparams")
        self.rec_pdmodel = os.path.join(self.rec_model_path, "inference.pdmodel")
        self.rec_label_file = os.path.join(self.util.model_path, "ppocr_keys_v1.txt")

        self.local_result_path = self.util.ground_truth
        self.image_file_path = os.path.join(self.util.data_path, "ICDAR2017_10")
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        pass
    
    def test_paddle_gpu(self):
        self.option.use_paddle_backend()
        self.option.use_gpu()

        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=self.option)
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=self.option)
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=self.option)
        model = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_paddle_gpu", csv_path=self.csv_save_path)
        

    def test_openvino_cpu(self):
        self.option.use_openvino_backend()
        self.option.use_cpu()

        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=self.option)
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=self.option)
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=self.option)
        model = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_openvino_cpu", csv_path=self.csv_save_path)


    def test_ort_gpu(self):
        self.option.use_ort_backend()
        self.option.use_gpu(0)

        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=self.option)
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=self.option)
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=self.option)
        model = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_ort_gpu", csv_path=self.csv_save_path)

        

