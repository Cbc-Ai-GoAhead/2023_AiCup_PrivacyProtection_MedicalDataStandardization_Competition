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
     #####
    ##  Testing Data
    ####
    print("test_dataset_doc_parh ={}".format(test_dataset_doc_parh))
    print("label_path ={}".format(test_label_path))
    print("#### 1 Load Test Dataset")

    test_path = [test_dataset_doc_parh + file_path for file_path in os.listdir(test_dataset_doc_parh)]
    print("num of test_path ={}, ".format(len(test_path)))#1734 #560
    # 已經把第一和第二train資料讀進來
    test_medical_record_dict = {} #x
    test_medical_record_dict = read_text_from_file(test_path)
    # print(test_medical_record_dict.keys())

    # print(test_medical_record_dict["file47256"])
    test_label_dict = create_label_dict_test(test_label_path)
    # print(test_label_dict['file64764'])
    test_pd = pd.read_csv(test_label_path, names=["fileid","start","context"], dtype=str,sep="\t")
    end_list=[]
    for row in range (len(test_pd)):
        start = test_pd.at[row, 'start']
        context = test_pd.at[row, 'context']
        # print(int(start)+len(context))
        # print(type(int(start)))
        # print(type(len(str(context))))
        end = int(start)+len(str(context))
        end_list.append(end)
    
    test_pd["end"]=end_list
    label_type = ['IDNUM','MEDICALRECORD','SET', 'DATE','ZIP','AGE','OTHER', 'COUNTRY',  'CITY', 'STATE',  'URL', 'TIME', 'DEPARTMENT',  'DOCTOR', 'ROOM', 'PHONE', 'HOSPITAL', 'ORGANIZATION', 'LOCATION-OTHER', 'STREET', 'PATIENT',  'DURATION']
#   need_stip_label = []
    print(len(label_type))
    print(len(end_list))
    print(len(end_list)/len(label_type))
    N = int(len(end_list)//len(label_type))
    # final_list = [copy.copy(e) for _ in range(N) for e in label_type]
    # print(len(final_list))
    # for idx in range (3590):
    #     label_type.extend(label_type)

    print(len(label_type))
    test_pd["label"]=end_list
    test_pd["label"]=label_type*3590
    print(test_pd.head())
    test_pd.to_csv("./answer_time_drop.txt", sep = '\t',columns=["fileid","label","start","end","context"], header=False ,index = None)
    # test_pd