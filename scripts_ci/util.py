import yaml
import os
import requests

class FastdeployTest(object):
    def __init__(self, data_dir_name: str, model_dir_name: str, model_name: str, url: str ,csv_path="./test.csv"):
        """
        需设置环境变量
        MODEL_PATH: 模型根目录
        DATA_PATH: 数据集根目录
        py_version: python版本 
        """
        self.py_version = os.environ.get("py_version")
        self.data_path = f"{os.environ.get('DATA_PATH')}/{data_dir_name}/"
        self.model_path = f"{os.environ.get('MODEL_PATH')}/{model_dir_name}/"
        self.model_name = model_name
        self.ground_truth = self.get_ground_truth(model_name)
        self.csv_path = csv_path
        self.check_file_exist(self.csv_path)
        

    def check_file_exist(self,csv_path):
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def get_ground_truth(self, model_name):

        f = open('ground_truth.yaml', 'r', encoding="utf-8")
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data[model_name]

    def get_ground_truth_from_url(self, url):
        
        ground_truth_path = "./"
        ground_truth_path = download(url, ground_truth_path)
        
        return ground_truth_path 
        
    @staticmethod
    def redirect_err_out(err="stderr.log", out="stdout.log"):
        import sys
        sys.stdout = open(err, "w")
        sys.stdout = open(out, "w")
            

def check_result(result_data: dict, ground_truth_data: dict, case_name="", model_name="", delta=0, csv_path="./test.csv"):
    for key, result_value in result_data.items():
        if "average_inference_time" in key:
            #write2speedexcel(model_name, case_name, key, result_value, "./infer_result/model_inference_spped.csv")
            continue
        assert key in ground_truth_data, "The key:{} in result_data is not in the ground_truth_data".format(key)
        ground_truth_val = ground_truth_data[key]
        diff = abs(result_value - ground_truth_val)
        print("diff: ", diff)
        write2excel(model_name, case_name, result_value, ground_truth_val, diff, csv_path)
        assert (diff <= delta), "The diff of {} between result_data and ground_truth_data is {} is bigger than {}".format(key, diff, delta)
    return 0


def ppocr_diff_check(ocr_model, image_file_path, local_result_path ,model_name="", case_name="", csv_path='./test.csv'):
    
    # Prepare Images
    img_dir = image_file_path
    imgs_file_lists = []
    if os.path.isdir(img_dir):
        for single_file in os.listdir(img_dir):
            if 'jpg' in single_file:
                file_path = os.path.join(img_dir, single_file)
                if os.path.isfile(file_path):
                    imgs_file_lists.append(file_path)
    imgs_file_lists.sort()

    # Read local result from txt file
    local_result=[]
    with open(local_result_path,'r') as f:
        for line in f:
            local_result.append(list(map(float,line.split(','))))

    # PPOCR Predict
    fd_result=[]
    for idx , image in enumerate(imgs_file_lists):
        
        img = cv2.imread(image)
        result = ocr_model.predict(img)
        #处理结果
        for i in range(len(result.boxes)):
            one_res = result.boxes[i] + [result.rec_scores[i]] + [result.cls_labels[i]] + [result.cls_scores[i]]
            fd_result.append(one_res) 

    # Begin to Diff Compare
    total_num_res = len(local_result)*11
    total_diff_num = 0 

    print("==== Begin to check OCR diff ====")
    for list_local, list_fd in zip (local_result, fd_result):

        for i in range(len(list_local)):
            
            if(i < 8):
                #Det
                diff = list_local[i] - list_fd[i]
                write2excel(model_name, case_name, list_fd[i], list_local[i], diff, csv_path)
                assert(abs(diff) < 1),"Diff exist in Det box result, where is {} - {} .".format(list_local,list_fd)
            elif (i == 8):
                #rec
                diff = round(list_local[i],6) - round(list_fd[i],6)
                write2excel(model_name, case_name, list_fd[i], list_local[i], diff, csv_path)
                assert(abs(diff) < 0.00001),"Diff exist in rec scores result, where is {} - {} .".format(list_local,list_fd)
            elif (i == 9):
                diff = list_local[i] - list_fd[i]
                write2excel(model_name, case_name, list_fd[i], list_local[i], diff, csv_path)
                assert(abs(diff) != 1),"Diff exist in cls label result, where is {} - {} .".format(list_local,list_fd)
            else:
                diff = round(list_local[i],6) - round(list_fd[i],6)
                write2excel(model_name, case_name, list_fd[i], list_local[i], diff, csv_path)
                assert(abs(diff) < 0.00001),"Diff exist in cls score result, where is {} - {} .".format(list_local,list_fd)
    print("==== Finish PPOCR diff check ==== ")
                


def write2excel(model_name, case_name, result_value, ground_truth_val, diff, file_path):
    import csv
    path=file_path
    if not os.path.exists(path):
        with open(path, "w+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(["model_name", "case_name", "infer_result", "ground_truth", "diff"])
        
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        data_row = [model_name, case_name, result_value, ground_truth_val, diff]
        csv_write.writerow(data_row)

def write2speedexcel(model_name, case_name, speed_case_name, speed_value, file_path):
    import csv
    path=file_path
    if not os.path.exists(path):
        with open(path, "w+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(["model_name", "case_name", "speed_case_name", "speed_value"])
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        data_row = [model_name, case_name, speed_case_name, speed_value]
        csv_write.writerow(data_row)
    
def print_log(file_list, iden=""):
    for file in file_list:
        print(f"======================{file} {iden}=====================")
        if os.path.exists(file):
            with open(file, "r") as f:
                print(f.read())
            if file.startswith("log"):
                os.remove(file)
        else:
            print(f"{file} not exist")
        print("======================================================")


def download(url,save_path):
    req = requests.get(url)
    save_path = save_path + url.split('/')[-1]
    
    with open(save_path, 'wb') as f:
        f.write(req.content)
    return save_path 