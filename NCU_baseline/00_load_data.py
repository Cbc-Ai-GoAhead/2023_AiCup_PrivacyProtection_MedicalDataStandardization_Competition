import os
from pprint import pprint as pp
from utils import *

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

first_dataset_doc_path = "../data/First_Phase_Release_Correction/First_Phase_Text_Dataset/"
second_dataset_doc_path = "../data/Second_Phase_Dataset/Second_Phase_Text_Dataset/"
label_path = ["../data/First_Phase_Release_Correction/answer.txt", "../data/Second_Phase_Dataset/answer.txt"]
val_dataset_doc_parh = "../data/valid_dataset/Validation_Release/"
val_label_path = "../data/valid_dataset/answer.txt"

if __name__ == '__main__':
    print("first_dataset_path ={}, second_dataset_path={}".format(first_dataset_doc_path, second_dataset_doc_path))
    print("label_path ={}".format(label_path))
    print("#### 1 Load First and Second Dataset")
    first_dataset_path = [first_dataset_doc_path + file_path for file_path in os.listdir(first_dataset_doc_path)]
    second_dataset_path = [second_dataset_doc_path + file_path for file_path in os.listdir(second_dataset_doc_path)]
    print("#### 1 Load train and val path")
    train_path = first_dataset_path + second_dataset_path
    val_path = [val_dataset_doc_parh + file_path for file_path in os.listdir(val_dataset_doc_parh)]
    
    print("num of first_dataset ={}, second_dataset_path={}".format(len(first_dataset_path), len(second_dataset_path)))#1120 #614
    print()
    print("num of train_path ={}, val_path={}".format(len(train_path), len(val_path)))#1734 #560

    print("---------------------")
    #####
    ##  Load train data
    #####
    #load train data from path
    print("#### load train data from path")
    train_medical_record_dict = {} #x
    train_medical_record_dict = read_text_from_file(train_path)
    
    # print("train_medical_record_dict = {}".format(train_medical_record_dict))
    # #load validation data from path
    # print("#### load validation data from path")
    # val_medical_record_dict = {} #x
    # val_medical_record_dict = read_text_from_file(val_path)


    #####
    ##  Load Label
    #####
    # label_path =['../data/First_Phase_Release_Correction/answer.txt', '../data/Second_Phase_Dataset/answer.txt']
    # train_label_dict, train_date_label_dict = create_label_dict(label_path[0])#
    # print(train_label_dict)
    """
    {[['DOCTOR', 18376, 18384, 'I Eifert'], ['TIME', 18412, 18431, '2512-10-20 00:00:00', '2512-10-20T00:00:00'], ['PATIENT', 18443, 18449, 'Bodway']]}
    """
    
    second_dataset_label_dict, second_date_label_dict = create_label_dict(label_path[1])
    train_label_dict.update(second_dataset_label_dict)
    train_date_label_dict.update(second_date_label_dict)
    val_label_dict, val_date_label_dict = create_label_dict(val_label_path)
    # print("val_date_label_dict = {}".format(val_date_label_dict))
    #####
    ##  Check the number of data
    #####
    # #chect the number of data
    ##1734  #1734  #560 #560
    print("num of train medical_data={}, label = {}, val  medical_data={}, label = {}".format(len(list(train_medical_record_dict.keys())),\
    len(list(train_label_dict.keys())), len(list(val_medical_record_dict.keys())), len(list(val_label_dict.keys()))))
    

    