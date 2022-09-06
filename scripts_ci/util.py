import yaml
import os

class FastdeployTest(object):
    def __init__(self, data_dir_name: str, model_dir_name: str, model_name: str, csv_path="./test.csv"):
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
        if (float(diff) - float(delta)) > 1e-10:
            write2excel(model_name, case_name, result_value, ground_truth_val, diff, csv_path)
        assert (diff <= delta), "The diff of {} between result_data and ground_truth_data is {} is bigger than {}".format(key, diff, delta)
    return 0


def write2excel(model_name, case_name, result_value, ground_truth_val, diff, file_path):
    import csv
    path=file_path
    if not os.path.exists(path):
        with open(path, "w") as f:
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
        with open(path, "a+") as f:
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
