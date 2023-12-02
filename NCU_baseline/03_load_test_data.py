import os
from pprint import pprint as pp
from utils import *

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from train import *
import pandas as pd
first_dataset_doc_path = "../data/First_Phase_Release_Correction/First_Phase_Text_Dataset/"
second_dataset_doc_path = "../data/Second_Phase_Dataset/Second_Phase_Text_Dataset/"
label_path = ["../data/First_Phase_Release_Correction/answer.txt", "../data/Second_Phase_Dataset/answer.txt"]
val_dataset_doc_parh = "../data/valid_dataset/Validation_Release/"
val_label_path = "../data/valid_dataset/answer.txt"

test_dataset_doc_parh = "../data/opendid_test/"
test_label_path = "../data/opendid_test.tsv"
if __name__ == '__main__':
    print("test_dataset_doc_parh ={}".format(test_dataset_doc_parh))
    print("label_path ={}".format(test_label_path))
    print("#### 1 Load Test Dataset")

    test_path = [test_dataset_doc_parh + file_path for file_path in os.listdir(test_dataset_doc_parh)]
    print("num of test_path ={}, ".format(len(test_path)))#1734 #560
    # 已經把第一和第二train資料讀進來
    test_medical_record_dict = {} #x
    test_medical_record_dict = read_text_from_file(test_path)
    # print(test_medical_record_dict)
    test_label_dict = create_label_dict_test(test_label_path)
    print(test_label_dict)