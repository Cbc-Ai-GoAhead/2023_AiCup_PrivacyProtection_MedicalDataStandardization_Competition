import os, random
from pprint import pprint as pp
from utils import *
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from bert_model import myModel
from utils import decode_model_result, calculate_batch_score
label_path = ["../data/First_Phase_Release_Correction/answer.txt", "../data/Second_Phase_Dataset/answer.txt"]

testing_dataset_doc_parh = "../data/test_dataset/opendid_test/"
testing_label_path = "../data/test_dataset/opendid_test.tsv"

# 設定隨機種子值，以確保輸出是確定的
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if __name__ == '__main__':
    # read_text_from_file()
    testing_dataset_path = [testing_dataset_doc_parh + file_path for file_path in os.listdir(testing_dataset_doc_parh)]
    print("#### load testing data from path")
    testing_medical_record_dict = {} #x
    testing_medical_record_dict = read_test_text_from_file(testing_dataset_path)

    # print(testing_medical_record_dict.keys())
    test_dict = testing_create_label_dict(testing_label_path)#

    labels_num = 22
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = labels_num)
    ####
    ##  Load Label
    ####
    train_label_dict, train_date_label_dict = create_label_dict(label_path[0])
    second_dataset_label_dict, second_date_label_dict = create_label_dict(label_path[1])
    train_label_dict.update(second_dataset_label_dict)
    #####
    ## 設定label 種類
    #####
    labels_type = list(set( [label[0] for labels in train_label_dict.values() for label in labels] ))
    labels_type_table = {label_name:id for id, label_name in enumerate(labels_type)}
    labels_type = ["OTHER"] + labels_type #add special token [other] in label list
    labels_num = len(labels_type)
    print("labels_type = {}".format(labels_type))
    # test_dataset = Privacy_protection_dataset(testing_medical_record_dict, test_dict, tokenizer, labels_type_table, "test")
    ######
    ##   Testing Dataset
    ######
    testdataset_Privacy_protection_dataset(Dataset):

    #model = myModel()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
     # Put model on device
    model_PATH = "./model/bert-base-cased_9_12_0.5539107501245775dict"
    model_path = "./model/bert-base-cased_9_12_0.5983084987356402"#bert-base-cased_9_12_0.5983084987356402"
    #model.load_state_dict(torch.load(model_PATH))#, map_location=torch.device('cpu')))
    model = torch.load(model_path)
    model = model.to(device)

    model.eval()
    #文本還是需要整個讀進來
    #for i, label in enumerate(test_dict):
    # new_testing_medical_record_dict = sorted(testing_medical_record_dict.items(), key=lambda x:x[1])
    # new_test_dict = sorted(test_dict.items(), key=lambda x:x[1])
    print("test_id_list===")
    test_id_list = list(testing_medical_record_dict.keys())
    test_medical_record = {sample_id: testing_medical_record_dict[sample_id] for sample_id in test_id_list}
    test_labels = {sample_id: test_dict[sample_id] for sample_id in test_id_list}
    # print(test_labels.keys())
    # print(test_labels["file65437"])
    # for sample, label in zip(test_medical_record, test_labels):
    output_string = ""
    for i, sample in enumerate(test_medical_record):
        # print("i={}, sample = {}".format(i, sample))
        context = test_medical_record[sample]
        # print("test_dict[label] = {}".format(testing_medical_record_dict[sample]))
        # print("test_dict[label] = {}".format(test_dict[label]))
        # test_list= test_dict[label]
        print("context ={}".format(context[:513]))
        encodings = tokenizer(sample, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
        print("encoding -----")
        print(encodings)
        
        encodings["input_ids"] = encodings["input_ids"].to(device)
        encodings["attention_mask"] = encodings["attention_mask"].to(device)
        outputs = model(encodings["input_ids"], encodings["attention_mask"])
        print("outputs -----")
        print(outputs)
        break
        model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
        # print("model_predict_table -----")
        # print(model_predict_table)
        model_predict_list = decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table)
        for predict_label_range in model_predict_list:
            predict_label_name, start, end = predict_label_range
            # print("redict_label_name={}, start={}, end={}".format(predict_label_name, start, end))
            predict_str = test_medical_record[sample][start:end]
            # print("redict_label_name={}".format(predict_str))
            # do the postprocessing at here
            sample_result_str = (sample +'\t'+ predict_label_name +'\t'+ str(start) +'\t'+ str(end) +'\t'+ predict_str + "\n")
            # print("sample_result_str={}".format(sample_result_str))
            output_string += sample_result_str
            # print("output_string={}".format(output_string))
    if not os.path.exists("./inference_testing"):
        os.mkdir("./inference_testing")
    with open("./inference_testing/testingset_answer.txt", "w", encoding="utf-8") as f:
        f.write(output_string)
        # for test in test_list:
        #     print(test)
        #     start, context = test
        #     # context 要是tokenizer的結果
        #     print("start={}, context ={}".format(start, context))
        #     encodings = tokenizer(context, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
        #     print("encoding -----")
        #     print(encodings)
        #     encodings["input_ids"] = encodings["input_ids"].to(device)
        #     encodings["attention_mask"] = encodings["attention_mask"].to(device)
        #     outputs = model(encodings["input_ids"], encodings["attention_mask"])
        #     print("outputs -----")
        #     print(outputs)

        #     model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
        #     print("model_predict_table -----")
        #     print(model_predict_table)

        #     model_predict_list = testing_decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table)
        #     print(model_predict_list)

        #     for predict_label_range in model_predict_list:
        #       predict_label_name, start, end = predict_label_range
        #       print("predict_label_name = {}, star={}, end={}".format(predict_label_name, start, end))
        #       predict_str = testing_medical_record_dict[id][start:end]
        #       print("predict_str = {}".format(predict_str))
        #       # do the postprocessing at here
        #       # Predict_str 會抓到 \n 換行符號 要再處理
        #       sample_result_str = (id +'\t'+ predict_label_name +'\t'+ str(start) +'\t'+ str(end) +'\t'+ predict_str + "\n")
        #       print("sample_result_str = {}".format(sample_result_str))
        #       print("output_string = {}".format(output_string))
        #       output_string += sample_result_str
        #     break
        # break
    # for i, sample in enumerate(testing_medical_record_dict):
    #     print("i={}".format(i))
    #     # print("sample = {}".format(testing_medical_record_dict))
    #     # print(testing_medical_record_dict[sample])
    #     break
    
    # print(testing_medical_record_dict["file63794"])
    # output_string = ""
    # for i, sample in enumerate(testing_medical_record_dict):
    #     #model.eval()
    #     print(sample)
    #     print(testing_medical_record_dict[sample])
    #     x, y, id = testing_medical_record_dict[sample]
        
    #     print("x={}, y={}, id ={}".format(x,y,id))
    #     break
        # encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
        # encodings["input_ids"] = encodings["input_ids"].to(device)
        # encodings["attention_mask"] = encodings["attention_mask"].to(device)
        # outputs = model(encodings["input_ids"], encodings["attention_mask"])


    