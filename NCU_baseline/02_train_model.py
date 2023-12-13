import os
from pprint import pprint as pp
from utils import *

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from train import *
from dataset_util import *
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
    # 已經把第一和第二train資料讀進來
    train_medical_record_dict = {} #x
    #train_medical_record_dict = read_text_from_file(train_path)
    # 5 reports
    train_medical_record_dict = read_text_from_file(train_path[:2])
    
    print("len train_medical_record_dict = {}".format(len(train_medical_record_dict)))
    # fileid = "file9830"
    # print("train_medical_record_dict = {}".format(train_medical_record_dict[fileid]))
    # print("train_medical_record_dict fileid len= {}".format(len(train_medical_record_dict[fileid])))
    # #load validation data from path
    # print("#### load validation data from path")
    val_medical_record_dict = {} #x
    # val_medical_record_dict = read_text_from_file(val_path)
    val_medical_record_dict = read_text_from_file(val_path[:2])
    print("len val_medical_record_dict = {}".format(len(val_medical_record_dict)))

    #####
    ##  Load Label
    #####
    label_path =['../data/First_Phase_Release_Correction/answer.txt', '../data/Second_Phase_Dataset/answer.txt']
    train_label_dict, train_date_label_dict = create_label_dict(label_path[0])#

    # print(train_label_dict[fileid])

    """
    {[['DOCTOR', 18376, 18384, 'I Eifert'], ['TIME', 18412, 18431, '2512-10-20 00:00:00', '2512-10-20T00:00:00'], ['PATIENT', 18443, 18449, 'Bodway']]}
    """
    
    second_dataset_label_dict, second_date_label_dict = create_label_dict(label_path[1])
    train_label_dict.update(second_dataset_label_dict)
    train_date_label_dict.update(second_date_label_dict)
    val_label_dict, val_date_label_dict = create_label_dict(val_label_path)
    # print("val_date_label_dict = {}".format(val_date_label_dict))

    ####
    ##  Process Text and Label
    ##  create_chunks for sliding window
    ####
    print("train_medical_record_dict.keys() len = {}".format(len(train_medical_record_dict.keys())))
    processed_medical_record_dict, processed_label_dict={}, {}
    for fileid in train_medical_record_dict.keys():
       
        print("Key = {}" .format(fileid))
        text_chunks_dict, label_chunks_dict = create_chunks(fileid, train_medical_record_dict[fileid],train_label_dict[fileid])
        processed_medical_record_dict.update(text_chunks_dict)
        processed_label_dict.update(label_chunks_dict)
    # val data 有需要做 Process 去掉沒用的label嗎
    
    # print(text_chunks)
    # print(label_chunks)
    
    #####
    ##  Check the number of data
    #####
    # #chect the number of data
    ##1734  #1734  #560 #560
    # print("num of train medical_data={}, label = {}, val  medical_data={}, label = {}".format(len(list(train_medical_record_dict.keys())),\
    # len(list(train_label_dict.keys())), len(list(val_medical_record_dict.keys())), len(list(val_label_dict.keys()))))

    print("num of train medical_data={}, label = {}, val  medical_data={}, label = {}".format(len(list(processed_medical_record_dict.keys())),\
    len(list(processed_label_dict.keys())), len(list(val_medical_record_dict.keys())), len(list(val_label_dict.keys()))))
    labels_type = list(set( [label[0] for labels in train_label_dict.values() for label in labels] ))
    labels_type = ["OTHER"] + labels_type
    labels_num = len(labels_type)
    print("labels_type = {}".format(labels_type))
    print("The number of labels labels_num = {}".format(labels_num))

    #####
    ##  Testing  Context and label
    ####
    print("Testing  Context and label")

    ####
    ##  Label to id
    ####


    # print(labels_type)
    # print("The number of labels:", labels_num)

    labels_type_table = {label_name:id for id, label_name in enumerate(labels_type)}
    # label to id
    # print("labels_type_table = {}".format(labels_type_table))


    ####
    ##  Check if there is unknow label in Val Label
    ####

    #check the label_type is enough for validation
    val_labels_type = list(set( [label[0] for labels in val_label_dict.values() for label in labels] ))
    for val_label_type in val_labels_type:
        if val_label_type not in labels_type:
            print("Special label in validation:", val_label_type)
    ####
    ##  Parameter Setting
    ####
    BACH_SIZE = 1

    ####
    ##  Prepare Dataset DataLoader
    ####
    ####
    ##  DATE TIME DURATION SET
    ####
    print("Prepare Dataset DataLoader")

    # processed_medical_record_dict
    # processed_label_dict
    # print(train_medical_record_dict.keys())
    train_id_list = list(processed_medical_record_dict.keys())
    train_medical_record = {sample_id: processed_medical_record_dict[sample_id] for sample_id in train_id_list}
    train_labels = {sample_id: processed_label_dict[sample_id] for sample_id in train_id_list}
    print("----Prepare Dataset DataLoader")

    # print("train_medical_record = {}".format(train_medical_record))
    # print("train_labels = {}".format(train_labels))
    # print("val_id_list===")
    val_id_list = list(val_medical_record_dict.keys())
    val_medical_record = {sample_id: val_medical_record_dict[sample_id] for sample_id in val_id_list}
    val_labels = {sample_id: val_label_dict[sample_id] for sample_id in val_id_list}
    

    print("Display Model------")
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    pretrained_weights = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    # config = AutoConfig.from_pretrained(pretrained_weights, num_labels = labels_num)
    # model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, config)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, num_labels = labels_num)

    # 需要先有模型做斷詞
    train_dataset = Privacy_protection_dataset(train_medical_record, train_labels, tokenizer, labels_type_table, "train")
    val_dataset = Privacy_protection_dataset(val_medical_record, val_labels, tokenizer, labels_type_table, "validation")

    print("---- Dataloader")
    train_dataloader = DataLoader( train_dataset, batch_size = BACH_SIZE, shuffle = True, collate_fn = train_dataset.collate_fn)
    val_dataloader = DataLoader( val_dataset, batch_size = BACH_SIZE, shuffle = False, collate_fn = val_dataset.collate_fn)
    print("----------------print_dataset_loaderstatus")

    print_dataset_loaderstatus(train_dataset, train_dataloader, labels_type_table, BACH_SIZE)

    #####
    ##  Testing Tokenizer  這裡就是在告訴我們 tokenzer完後文本的狀況 
    ## 所以助教給我們程式碼 已經是有修改長度到512 並且文本和label的id有重新修改過
    #####
    

    def post_proxessing(model_result:list):
        #need fix
        return [label.strip() for label in model_result]
    print("----------------print_annotated_medical_report")
    # print_annotated_medical_report(tokenizer, train_dataset, train_medical_record_dict, train_label_dict)

    #####
    ##  Training
    #####
    print("### Train")
    # finetune_model(train_dataloader, val_dataloader, val_dataset)
    
    