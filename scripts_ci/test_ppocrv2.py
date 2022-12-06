from util import *
import fastdeploy as fd
import os

@pytest.mark.skip(reason="临时跳过")
class FastdeployTestPPOCR(FastdeployTest):
    def __init__(self, data_dir_name: str, model_dir_name: str, model_name: str, url: str ,csv_path="./test.csv"):
        self.py_version = os.environ.get("py_version")
        self.data_path = f"{os.environ.get('DATA_PATH')}/{data_dir_name}/"
        self.model_path = f"{os.environ.get('MODEL_PATH')}/{model_dir_name}/"
        self.model_name = model_name
        self.csv_path = csv_path
        self.check_file_exist(self.csv_path)
        self.ground_truth = self.get_ground_truth_from_url(url)

@pytest.mark.skip(reason="临时跳过")
class TestPPOCRv2Test(object):
    
    def setup_class(self):

        self.util = FastdeployTestPPOCR(data_dir_name="ICDAR2017_10", 
            model_dir_name="PPOCRv2_models", 
            model_name="PPOCRv2",
            url="https://bj.bcebos.com/paddlehub/fastdeploy/PPOCRv2_ICDAR10_BS116.txt",
            csv_path="./infer_result/PPOCRv2_result.csv")

        # Det Model
        self.det_model_path = os.path.join(self.util.model_path, "ch_PP-OCRv2_det_infer")  
        self.det_pdiparams = os.path.join(self.det_model_path, "inference.pdiparams")
        self.det_pdmodel = os.path.join(self.det_model_path, "inference.pdmodel")
        # Cls Model
        self.cls_model_path = os.path.join(self.util.model_path, "ch_ppocr_mobile_v2.0_cls_infer") 
        self.cls_pdiparams = os.path.join(self.cls_model_path, "inference.pdiparams")
        self.cls_pdmodel = os.path.join(self.cls_model_path, "inference.pdmodel")

        # Rec Model
        self.rec_model_path = os.path.join(self.util.model_path, "ch_PP-OCRv2_rec_infer") 
        self.rec_pdiparams = os.path.join(self.rec_model_path, "inference.pdiparams")
        self.rec_pdmodel = os.path.join(self.rec_model_path, "inference.pdmodel")
        self.rec_label_file = os.path.join(self.util.model_path, "ppocr_keys_v1.txt")

        self.local_result_path = self.util.ground_truth
        self.image_file_path = self.util.data_path
        self.option = fd.RuntimeOption()
        self.model_name = self.util.model_name
        self.csv_save_path = self.util.csv_path
        
    def teardown_method(self):
        pass

    def test_ort_cpu(self):
        self.option.use_ort_backend()
        self.option.use_cpu()

        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=self.option)
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=self.option)
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=self.option)
        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_ort_cpu", csv_path=self.csv_save_path)


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
        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_ort_gpu", csv_path=self.csv_save_path)
    
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
        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_paddle_gpu", csv_path=self.csv_save_path)
        
    def test_paddle_cpu(self):
        self.option.use_paddle_backend()
        self.option.use_cpu()

        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=self.option)
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=self.option)
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=self.option)
        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_paddle_cpu", csv_path=self.csv_save_path)

        
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
        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        
        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_openvino_cpu", csv_path=self.csv_save_path)
    

    def test_trt_gpu(self):
        self.option.use_trt_backend()
        self.option.use_gpu()

        # Det
        det_option = self.option
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])
        det_model = fd.vision.ocr.DBDetector(self.det_pdmodel, self.det_pdiparams, runtime_option=det_option)

        # Cls
        cls_option = self.option
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10], [10, 3, 48, 320],
                               [32, 3, 48, 1024])
        cls_model = fd.vision.ocr.Classifier(self.cls_pdmodel, self.cls_pdiparams, runtime_option=cls_option)

        # Rec
        rec_option = self.option
        rec_option.set_trt_input_shape("x", [1, 3, 32, 10], [10, 3, 32, 320],
                               [32, 3, 32, 2304])
        rec_model = fd.vision.ocr.Recognizer(
            self.rec_pdmodel,
            self.rec_pdiparams,
            self.rec_label_file,
            runtime_option=rec_option)
        

        model = fd.vision.ocr.PPOCRv2(det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        model.cls_batch_size = 1
        model.rec_batch_size = 6
        ppocr_diff_check(model, self.image_file_path, self.local_result_path , model_name=self.model_name, case_name="test_trt_gpu", csv_path=self.csv_save_path)
