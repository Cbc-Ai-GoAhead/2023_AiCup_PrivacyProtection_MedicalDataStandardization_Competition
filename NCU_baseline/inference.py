import os
from pprint import pprint as pp
from utils import *

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from bert_model import myModel
from utils import decode_model_result, calculate_batch_score
# os.mkdir("./model/")
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

  # print(label_path)
  print("#### 2 Load Label to solve \ufeff problem")
  #\ufeff problem # 直接讀檔會有 \udeff問題
  with open(label_path[1], "r", encoding="utf-8") as f:
      file_text = f.read()
      #file_text = file_text[1:].strip()
      # file13264
  print("file_text = {}".format(file_text[:50]))

  pp("file_text = {}".format(file_text[:50]))# pp ->可以印出 \t 的資訊

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
  print("#### load validation data from path")
  val_medical_record_dict = {} #x
  val_medical_record_dict = read_text_from_file(val_path)


  #####
  ##  Load Label
  #####
  # label_path =['../data/First_Phase_Release_Correction/answer.txt', '../data/Second_Phase_Dataset/answer.txt']
  train_label_dict, train_date_label_dict = create_label_dict(label_path[0])#
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
  # print(len(list(train_medical_record_dict.keys()))) #1734
  # print(len(list(train_label_dict.keys()))) #1734
  # print(len(list(val_medical_record_dict.keys()))) #560
  # print(len(list(val_label_dict.keys()))) #560

  #####
  ##  Testing  Context and label
  ####
  print("Testing  Context and label")
  # input id (String type)
  #output the medical_record
  # print(train_medical_record_dict["10"])# 印出file_name=10 檔案內容

  # input id (String type)
  # output all labels from medical_record (list type)
  # pp(train_label_dict["10"]) # 印出file_name=10 檔案label

  # {[['file14362', 'TIME', 2410, 2426, '3:10pm on 5/9/16', '2016-09-05T15:10'], [] ]}
  print("## Get label from train_label_dict")
  print("## Get label type") # IDNUM, MEDICALRECORD, DATE, TIME
  # for labels in train_label_dict.values():
  #   #[['IDNUM', 14, 24, '09F016547J'], ['MEDICALRECORD', 25, 35, '091016.NMT'],]
  #   print(labels)
  #   for label in labels:
  #     #['IDNUM', 14, 24, '09F016547J']
  #     print(label)
  #     break
  #   break
  labels_type = list(set( [label[0] for labels in train_label_dict.values() for label in labels] ))
  # print("labels_type = {}".format(labels_type))



  # 原本只有21 個類別
  # 加入 OTHER 總共有 22個類別 去掉 #DATE TIME DURATION SET => 22-4=18
  labels_type = ["OTHER"] + labels_type #add special token [other] in label list
  labels_num = len(labels_type)
  print("labels_type = {}".format(labels_type))
  print("The number of labels labels_num = {}".format(labels_num))

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
  # print(train_medical_record_dict.keys())
  train_id_list = list(train_medical_record_dict.keys())
  train_medical_record = {sample_id: train_medical_record_dict[sample_id] for sample_id in train_id_list}
  train_labels = {sample_id: train_label_dict[sample_id] for sample_id in train_id_list}
  print("----Prepare Dataset DataLoader")
  # print("train_medical_record=  {}".format(train_medical_record))
  # {'file13264': 'SPR no: 42I520757G\nMRN no: 4235207\nSite_name: HORNSBY KU-RING-GAI HOSPITAL\n
  # ,'file23878': 'SPR no: 85A648111M\nMRN no: 850648\nSite_name: LAURA CAMPUS}

  # print("train_labels= {}".format(train_labels))
  # ['DATE', 3401, 3407, '3.4.64', '2064-03-04'], ['DATE', 4118, 4125, '20.4.64', '2064-04-20']]}
  print("val_id_list===")
  val_id_list = list(val_medical_record_dict.keys())
  val_medical_record = {sample_id: val_medical_record_dict[sample_id] for sample_id in val_id_list}
  val_labels = {sample_id: val_label_dict[sample_id] for sample_id in val_id_list}

  # print("val_medical_record= key = {}, value={}".format(val_medical_record.keys()[:10], val_medical_record.values()[:10]))
  # print("val_medical_record= {}".format(val_medical_record[:10]))
  # print("val_labels= {}".format(val_labels[:10]))

  print("Display Model------")
  from transformers import AutoTokenizer, AutoModelForTokenClassification
  model_name = "bert-base-cased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = labels_num)
  # 需要先有模型做斷詞
  train_dataset = Privacy_protection_dataset(train_medical_record, train_labels, tokenizer, labels_type_table, "train")
  val_dataset = Privacy_protection_dataset(val_medical_record, val_labels, tokenizer, labels_type_table, "validation")

  train_dataloader = DataLoader( train_dataset, batch_size = BACH_SIZE, shuffle = True, collate_fn = train_dataset.collate_fn)
  val_dataloader = DataLoader( val_dataset, batch_size = BACH_SIZE, shuffle = False, collate_fn = val_dataset.collate_fn)
  print("----------------")

  #####
  ##  Testing DataSet
  #####
  print(len(train_dataset))
  for sample in train_dataset:
    train_x, train_y,_ = sample
    # print("train_x = {} , train_y={}".format(train_x, train_y))
    # print(train_y)
    break
  print("-----------------")
  # print("DataLoader")
  # print(len(train_dataloader))
  for sample in train_dataloader:
    # batch_medical_record, encodings, batch_labels_tensor, batch_labels
    x_name,train_x, train_y, _ = sample
    # print("x_name = {},".format(x_name))
    # print("train_x = {}, train_y= {}".format(train_x, train_y))
    # print("len train_x = {}, train_y= {}".format(len(train_x), len(train_y)))
    print("-----------------")
    # print(x_name[4440:4448])
    # ['HOSPITAL', ]
    # print(x_name[143:155])
    break
  print("----------------")
  #show the first batch labels embeddings
  print(labels_type_table)
  for i in range(BACH_SIZE):
    print(train_y[i].tolist())
    # 會補成512長度
    print("len train_y[i] ={}".format(len(train_y[i].tolist())))

  #####
  ##  Testing Tokenizer  這裡就是在告訴我們 tokenzer完後文本的狀況 
  ## 所以助教給我們程式碼 已經是有修改長度到512 並且文本和label的id有重新修改過
  #####
  print("--------------------------")
  print("#### Tokenizer")
  #some exist id "10", "11", "12", "file16529"
  id = "file10996"
  print(train_medical_record_dict[id])
  pp(train_label_dict[id])
  print("Number of character in medical_record:", len(train_medical_record_dict[id]))

  example_medical_record = train_medical_record_dict[id]
  example_labels = train_label_dict[id]

  encodings = tokenizer(example_medical_record, padding=True, return_tensors="pt", return_offsets_mapping="True")
  print(encodings.keys())
  #print(encodings["input_ids"])
  #print(encodings["attention_mask"])
  print(encodings["offset_mapping"])
  print(encodings["offset_mapping"].shape)
  #print(tokenizer.decode(encodings["input_ids"][0])) #get the original text

  print(encodings["input_ids"].shape)
  print(encodings["attention_mask"].shape)
  print(len(encodings["offset_mapping"][0]))

  print("### Testing find_token_ids (the funtion in Privacy_protection_dataset)")

  encodeing_start, encodeing_end = train_dataset.find_token_ids(train_label_dict[id][3][1], train_label_dict[id][3][2], encodings["offset_mapping"][0])
  print(encodeing_start, encodeing_end)

  #get the original text
  print(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])) #sometime will error

  decode_start_pos = int(encodings["offset_mapping"][0][encodeing_start][0])
  decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end-1][1])
  print(decode_start_pos, decode_end_pos)
  print(train_medical_record_dict[id][decode_start_pos:decode_end_pos])

  def post_proxessing(model_result:list):
    #need fix
    return [label.strip() for label in model_result]

  print(example_labels)
  position_list = [label[1:3] for label in example_labels]
  expected_result = [label[3] for label in example_labels]
  predict_result = []
  for position in position_list:
    encodeing_start, encodeing_end = train_dataset.find_token_ids(position[0], position[1], encodings["offset_mapping"][0])
    #fix the decode solution at here
    predict_result.append(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])) #sometime will error
  print("predict_result = {}, expected_result={}" .format(predict_result, expected_result))
  print("post_proxessing(predict_result) ={}".format(post_proxessing(predict_result)))
  # print(expected_result)
  print("predict_result == expected_result = {}, post_proxessing(predict_result) == expected_result={}".format(predict_result == expected_result, post_proxessing(predict_result) == expected_result)) # token range clipping problem

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model_PATH = "model/bert-base-cased_0_0.5043912513830731"
  model = torch.load(model_PATH)
  # model = myModel(*args, **kwargs)
  # model.load_state_dict(torch.load(model_PATH))
  model.eval()

 

  # model.state_dict()
  model = model.to(device)
  output_string = ""
  for i, sample in enumerate(val_dataset):
      model.eval()
      x, y, id = sample
      #print(id)
      encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
      encodings["input_ids"] = encodings["input_ids"].to(device)
      encodings["attention_mask"] = encodings["attention_mask"].to(device)
      outputs = model(encodings["input_ids"], encodings["attention_mask"])
      #output = softmax(outputs.logits)
      model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
      #print(model_predict_table)
      model_predict_list = decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table)
      #print(model_predict_list)
      for predict_label_range in model_predict_list:
          predict_label_name, start, end = predict_label_range
          predict_str = val_medical_record_dict[id][start:end]
          # do the postprocessing at here
          sample_result_str = (id +'\t'+ predict_label_name +'\t'+ str(start) +'\t'+ str(end) +'\t'+ predict_str + "\n")
          output_string += sample_result_str
      #print(y)
  if not os.path.exists("./inference_testing"):
      os.mkdir("./inference_testing")
  with open("./inference_testing/answer.txt", "w", encoding="utf-8") as f:
      f.write(output_string)


  print("### Other")
  for i, sample in enumerate(val_dataset):
    model.eval()
    x, y, id = sample
    print(id)
    encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
    encodings["input_ids"] = encodings["input_ids"].to(device)
    encodings["attention_mask"] = encodings["attention_mask"].to(device)
    outputs = model(encodings["input_ids"], encodings["attention_mask"])
    #output = softmax(outputs.logits)
    model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
    #print(model_predict_table)
    print(decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table))
    print(y)
    break