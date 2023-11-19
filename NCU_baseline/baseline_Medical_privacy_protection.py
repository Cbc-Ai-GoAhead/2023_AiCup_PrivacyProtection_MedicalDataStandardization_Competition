import os
from pprint import pprint as pp
# os.mkdir("./model/")

first_dataset_doc_path = "../data/First_Phase_Release_Correction/First_Phase_Text_Dataset/"
second_dataset_doc_path = "../data/Second_Phase_Dataset/Second_Phase_Text_Dataset/"
label_path = ["../data/First_Phase_Release_Correction/answer.txt", "../data/Second_Phase_Dataset/answer.txt"]
val_dataset_doc_parh = "../data/valid_dataset/Validation_Release/"
val_label_path = "../data/valid_dataset/answer.txt"

first_dataset_path = [first_dataset_doc_path + file_path for file_path in os.listdir(first_dataset_doc_path)]
second_dataset_path = [second_dataset_doc_path + file_path for file_path in os.listdir(second_dataset_doc_path)]
train_path = first_dataset_path + second_dataset_path
val_path = [val_dataset_doc_parh + file_path for file_path in os.listdir(val_dataset_doc_parh)]

print(len(first_dataset_path)) #1120
print(len(second_dataset_path)) #614
print()
print(len(train_path)) #1734
print(len(val_path)) #560

#\ufeff problem
with open(label_path[1], "r", encoding="utf-8") as f:
    file_text = f.read()
    #file_text = file_text[1:].strip()
print(file_text[:50])
pp(file_text[:50])

def create_label_dict(label_path):
  label_dict = {} #y
  with open(label_path, "r", encoding="utf-8") as f:
    file_text = f.read()
    file_text = file_text.strip("\ufeff").strip() #train file didn't remove this head
  for line in file_text.split("\n"):
    sample = line.split("\t") #(id, label, start, end, query) or (id, label, start, end, query, time_org, timefix)
    sample[2], sample[3] = (int(sample[2]), int(sample[3])) #start, end = (int(start), int(end))
    #print(sample)
    if sample[0] not in label_dict.keys():
      label_dict[sample[0]] = [sample[1:]]
    else:
      label_dict[sample[0]].append(sample[1:])
  #print(label_dict)
  return label_dict

#load train data from path
train_medical_record_dict = {} #x
for data_path in train_path:
  id = data_path.split("/")[-1].split(".txt")[0]
  with open(data_path, "r", encoding="utf-8") as f:
    file_text = f.read()
    train_medical_record_dict[id] = file_text

#load validation data from path
val_medical_record_dict = {} #x
for data_path in val_path:
  id = data_path.split("/")[-1].split(".txt")[0]
  with open(data_path, "r", encoding="utf-8") as f:
    file_text = f.read()
    val_medical_record_dict[id] = file_text


train_label_dict = create_label_dict(label_path[0])
second_dataset_label_dict = create_label_dict(label_path[1])
train_label_dict.update(second_dataset_label_dict)
val_label_dict = create_label_dict(val_label_path)


#chect the number of data
print(len(list(train_medical_record_dict.keys()))) #1734
print(len(list(train_label_dict.keys()))) #1734
print(len(list(val_medical_record_dict.keys()))) #560
print(len(list(val_label_dict.keys()))) #560

# input id (String type)
#output the medical_record
print(train_medical_record_dict["10"])

# input id (String type)
# output all labels from medical_record (list type)
pp(train_label_dict["10"])

labels_type = list(set( [label[0] for labels in train_label_dict.values() for label in labels] ))
labels_type = ["OTHER"] + labels_type #add special token [other] in label list
labels_num = len(labels_type)

print(labels_type)
print("The number of labels:", labels_num)

labels_type_table = {label_name:id for id, label_name in enumerate(labels_type)}
print(labels_type_table)




#check the label_type is enough for validation
val_labels_type = list(set( [label[0] for labels in val_label_dict.values() for label in labels] ))
for val_label_type in val_labels_type:
  if val_label_type not in labels_type:
    print("Special label in validation:", val_label_type)

print("#### Load pre trained!!")
#Loading time depend on your network speed
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = labels_num)

print(type(tokenizer))
print(type(model))
print(model)


print("Self define Model")
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.droupout = nn.Dropout(p=0.1, inplace=False)
        self.fc = nn.Linear(768, 22)


    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.shape)
        output = self.droupout(output.last_hidden_state)
        out = self.fc(output)

        return out
# model = myModel()
# print(model)

BACH_SIZE = 1
# #TRAIN_RATIO = 0.9
# LEARNING_RATE = 1e-4
# EPOCH = 1


import torch
from torch.utils.data import Dataset, DataLoader
class Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      if mode == "train" or mode == "validation":
        self.id_list = list(medical_record_dict.keys())
        self.x = list(medical_record_dict.values())
        self.y = [[labels[:3] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]

  def __getitem__(self, index):
      return self.x[index], self.y[index], self.id_list[index]

  def __len__(self):
      return len(self.x)

  def find_token_ids(self, label_start, label_end, offset_mapping):
    """find the correct labels ids after tokenizer"""
    encodeing_start = float("inf") #max
    encodeing_end = 0
    for token_id, token_range in enumerate(offset_mapping):
      token_start, token_end = token_range
      #if token range one side out of label range, still take the token
      if token_start == 0 and token_end == 0: #special tocken
        continue
      if label_start<token_end and label_end>token_start:
        if token_id<encodeing_start:
          encodeing_start = token_id
        encodeing_end = token_id+1
    return encodeing_start, encodeing_end

  def encode_labels_position(self, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):
      encodeing_labels = []
      for label in sample_labels:
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)
        encodeing_labels.append([label[0], encodeing_start, encodeing_end])
      batch_encodeing_labels.append(encodeing_labels)
    return batch_encodeing_labels

  def create_labels_tensor(self, batch_shape:list, batch_labels_position_encoded:list):
    if batch_shape[-1]> 4096:
      batch_shape[-1] = 4096
    labels_tensor = torch.zeros(batch_shape)

    for sample_id in range(batch_shape[0]):
      for label in batch_labels_position_encoded[sample_id]:
        label_id = self.labels_type_table[label[0]]
        start = label[1]
        end = label[2]
        if start >= 4096: continue
        elif end >= 4096: end = 4096
        labels_tensor[sample_id][start:end] = label_id
    return labels_tensor

  def collate_fn(self, batch_items:list):
    """the calculation process in dataloader iteration"""
    batch_medical_record = [sample[0] for sample in batch_items]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True

    batch_labels_position_encoded = self.encode_labels_position(batch_labels, encodings["offset_mapping"])
    #print(encodings["offset_mapping"])
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)
    return encodings, batch_labels_tensor, batch_labels

train_id_list = list(train_medical_record_dict.keys())
train_medical_record = {sample_id: train_medical_record_dict[sample_id] for sample_id in train_id_list}
train_labels = {sample_id: train_label_dict[sample_id] for sample_id in train_id_list}

val_id_list = list(val_medical_record_dict.keys())
val_medical_record = {sample_id: val_medical_record_dict[sample_id] for sample_id in val_id_list}
val_labels = {sample_id: val_label_dict[sample_id] for sample_id in val_id_list}

train_dataset = Privacy_protection_dataset(train_medical_record, train_labels, tokenizer, labels_type_table, "train")
val_dataset = Privacy_protection_dataset(val_medical_record, val_labels, tokenizer, labels_type_table, "validation")


train_dataloader = DataLoader( train_dataset, batch_size = BACH_SIZE, shuffle = True, collate_fn = train_dataset.collate_fn)
val_dataloader = DataLoader( val_dataset, batch_size = BACH_SIZE, shuffle = False, collate_fn = val_dataset.collate_fn)

print(len(train_dataset))
for sample in train_dataset:
  train_x, train_y,_ = sample
  #print(train_x)
  print(train_y)
  break

print(len(train_dataloader))
for sample in train_dataloader:
  train_x, train_y, _ = sample
  #print(train_x)
  print(train_y)
  break

#show the first batch labels embeddings
print(labels_type_table)
for i in range(BACH_SIZE):
  print(train_y[i].tolist())

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
print("encodeing_start={}, encodeing_end={}" .format(encodeing_start, encodeing_end))

#get the original text
print(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])) #sometime will error
# 7303827. KMG
print("encodings[input_ids][0] = {}" .format(encodings["input_ids"][0])) #是 文檔內容的 tensor_id
print("tokenizer.decode-1 = {}" .format(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])))
decode_start_pos = int(encodings["offset_mapping"][0][encodeing_start][0])
decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end-1][1])

print("offset_mapping decode")
# print(encodings["offset_mapping"])
# tensor([[[   0,    0],
#          [   0,    2],
#          [   2,    3],
#          ...,
#          [1903, 1906],
#          [1906, 1909],
#          [   0,    0]]])
print(encodings["offset_mapping"][0][encodeing_start][0])#第65個的第0個值149
print(encodings["offset_mapping"][0][encodeing_start][1])#第65個的第0個值151
print(encodings["offset_mapping"][0][encodeing_start-1][1])#第64個值的第1一個值147

# 7303827.KMG
# s
# 7303827.KMG
# 1  7303827.KMG
print("Start END")
print(train_medical_record_dict[id][int(encodings["offset_mapping"][0][encodeing_start][0]):int(encodings["offset_mapping"][0][encodeing_end][1])])
print("Start END-1")
print(train_medical_record_dict[id][int(encodings["offset_mapping"][0][encodeing_start][0]):int(encodings["offset_mapping"][0][encodeing_end-1][1])])
print("Start-1 END-1")
print(train_medical_record_dict[id][int(encodings["offset_mapping"][0][encodeing_start-1][0]):int(encodings["offset_mapping"][0][encodeing_end-1][1])])
print("Doing Format offset_mapping")
print("decode_start_pos={}, decode_end_pos={}" .format(decode_start_pos, decode_end_pos))
print("train_medical_record_dict[id][decode_start_pos:decode_end_pos]= ")
#7303827.KMG 空格會被刪去
# 重新修正 tokenizer後的位置
print("encodeing_end-1 =")
print(train_medical_record_dict[id][decode_start_pos:decode_end_pos])
print("encodeing_end  =")
print(train_medical_record_dict[id][decode_start_pos:decode_end_pos+1])



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
print("predict_result = {}" .format(predict_result))
print("expected_result= {}".format(expected_result))
print("post_proxessing(predict_result) ={}".format(post_proxessing(predict_result)))
# print(expected_result)
print("predict_result == expected_result = {}, post_proxessing(predict_result) == expected_result={}".format(predict_result == expected_result, post_proxessing(predict_result) == expected_result)) # token range clipping problem

#you need to do post proxessing after useing the model to solve the problem

# print(post_proxessing(predict_result) == expected_result)

#show the samples in train still need preprocess

error = 0
error_list = []
for id in train_dataset.id_list:

  # example_medical_record = medical_record_dict[id]
  example_medical_record = train_medical_record_dict[id]
  example_labels = train_label_dict[id]
  position_list = [label[1:3] for label in example_labels]
  expected_result = [label[3] for label in example_labels]
  predict_result = []
  print("position_list={} expected_result={}".format(position_list, expected_result))
  break
#   encodings = tokenizer(example_medical_record, padding=True, return_tensors="pt", return_offsets_mapping="True")
#   for position in position_list:
#     encodeing_start, encodeing_end = train_dataset.find_token_ids(position[0], position[1], encodings["offset_mapping"][0])
#     predict_result.append(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])) #get the original text
#   if post_proxessing(predict_result) != expected_result:
#     error_list.append(id)
#     error += 1

# if error == 0:
#   print("pass the test")
# else:
#   print("error list:", error_list)



# print("### Train")
# # Load the TensorBoard notebook extension
# # %load_ext tensorboard
# # # Clear any logs from previous runs
# # !rm -rf ./logs/

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# from tqdm import tqdm
# from torch.optim import AdamW
# from torch.nn import CrossEntropyLoss
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = model.to(device) # Put model on device
# optim = AdamW(model.parameters(), lr = LEARNING_RATE)
# loss_fct = CrossEntropyLoss()

# def decode_model_result(model_predict_table, offsets_mapping, labels_type_table):
#   model_predict_list = model_predict_table.tolist()
#   id_to_label = {id:label for label, id in labels_type_table.items()}
#   predict_y = []
#   pre_label_id = 0
#   for position_id, label_id in enumerate(model_predict_list):
#     if label_id!=0:
#       if pre_label_id!=label_id:
#         start = int(offsets_mapping[position_id][0])
#       end = int(offsets_mapping[position_id][1])
#     if pre_label_id!=label_id and pre_label_id!=0:
#       predict_y.append([id_to_label[pre_label_id], start, end])
#     pre_label_id = label_id
#   if pre_label_id!=0:
#     predict_y.append([id_to_label[pre_label_id], start, end])
#   return predict_y

# def calculate_batch_score(batch_labels, model_predict_tables, offset_mappings, labels_type_table):
#     score_table = {"TP":0, "FP":0, "TN":0}
#     batch_size = model_predict_tables.shape[0]
#     for batch_id in range(batch_size):
#         smaple_prediction = decode_model_result(model_predict_tables[batch_id], offset_mappings[batch_id], labels_type_table)
#         smaple_ground_truth = batch_labels[batch_id]
#         #print(smaple_prediction)
#         #print(smaple_ground_truth)
#         # do the post_processing at here
#         # calculeate TP, TN, FP
#         smaple_ground_truth = set([tuple(token) for token in smaple_ground_truth])
#         smaple_prediction = set([tuple(token) for token in smaple_prediction])
#         score_table["TP"] += len( smaple_ground_truth & smaple_prediction)
#         score_table["TN"] += len( smaple_ground_truth - smaple_prediction)
#         score_table["FP"] += len( smaple_prediction - smaple_ground_truth)
#     if (score_table["TP"] + score_table["FP"])==0 or (score_table["TP"] + score_table["TN"])==0:
#       return 0, 0, 0

#     Precision = score_table["TP"] / (score_table["TP"] + score_table["FP"])
#     Recall = score_table["TP"] / (score_table["TP"] + score_table["TN"])
#     if(Precision + Recall) ==0:
#       return 0, 0, 0

#     F1_score = 2 * (Precision * Recall) / (Precision + Recall)
#     return Precision, Recall, F1_score

# train_step = 0
# val_step = 0
# # one epoch about 8 minutes for T4 on Colab
# # 5G memory needed when BACH_SIZE = 1
# for epoch in range(EPOCH):
#   model.train()
#   for batch_x, batch_y, batch_labels in train_dataloader:
#     train_step += 1
#     optim.zero_grad()
#     batch_x["input_ids"] = batch_x["input_ids"].to(device)
#     batch_x["attention_mask"] = batch_x["attention_mask"].to(device)
#     batch_y = batch_y.long().to(device)
#     outputs = model(batch_x["input_ids"], batch_x["attention_mask"])
#     #print(batch_y.shape)
#     train_loss = loss_fct(outputs.transpose(-1, -2), batch_y)
#     #print("train_loss", train_loss)
#     writer.add_scalar('Loss/train', train_loss, train_step)

#     # calculate loss
#     train_loss.backward()
#     # update parameters
#     optim.step()

#   model.eval()
#   sum_val_F1 = 0
#   for batch_x, batch_y, batch_labels in val_dataloader:
#     val_step += 1
#     optim.zero_grad()
#     batch_x["input_ids"] = batch_x["input_ids"].to(device)
#     batch_x["attention_mask"] = batch_x["attention_mask"].to(device)
#     batch_y = batch_y.long().to(device)
#     outputs = model(batch_x["input_ids"], batch_x["attention_mask"])
#     model_predict_tables = torch.argmax(outputs, dim=-1, keepdim=True)
#     model_predict_tables = model_predict_tables.squeeze(-1)
#     P, R, F1 = calculate_batch_score(batch_labels, model_predict_tables, batch_x["offset_mapping"], labels_type_table)
#     if val_step%50==0:
#       print("val_F1", F1)
#     val_loss = loss_fct(outputs.transpose(-1, -2), batch_y)
#     sum_val_F1 += float(F1)
#     #print("val_loss", val_loss)
#     writer.add_scalar('Loss/val', val_loss, val_step)
#     writer.add_scalar('F1/val', F1, val_step)
#   torch.save(model.state_dict(), "./model/" + "bert-base-cased"+"_"+str(epoch)+"_"+str(sum_val_F1/len(val_dataloader)))
# writer.close()





# for i, sample in enumerate(val_dataloader):
#   model.eval()
#   encodings, y, batch_labels = sample
#   batch_size = encodings["input_ids"].shape[0]
#   #print(i)
#   #encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
#   encodings["input_ids"] = encodings["input_ids"].to(device)
#   encodings["attention_mask"] = encodings["attention_mask"].to(device)
#   outputs = model(encodings["input_ids"], encodings["attention_mask"])
#   #output = softmax(outputs.logits)
#   #print(outputs.logits.shape)
#   model_predict_tables = torch.argmax(outputs, dim=-1, keepdim=True)
#   #if batch_size==1:
#   #  model_predict_table.unsqueeze(0)
#   model_predict_tables = model_predict_tables.squeeze(-1)

#   P, R, F1 = calculate_batch_score(batch_labels, model_predict_tables, encodings["offset_mapping"], labels_type_table)
#   print("Precision: %.2f, Recall %.2f, F1 score: %.2f" %(P, R, F1))
#   if i==20:
#     break

# output_string = ""
# for i, sample in enumerate(val_dataset):
#     model.eval()
#     x, y, id = sample
#     #print(id)
#     encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
#     encodings["input_ids"] = encodings["input_ids"].to(device)
#     encodings["attention_mask"] = encodings["attention_mask"].to(device)
#     outputs = model(encodings["input_ids"], encodings["attention_mask"])
#     #output = softmax(outputs.logits)
#     model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
#     #print(model_predict_table)
#     model_predict_list = decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table)
#     #print(model_predict_list)
#     for predict_label_range in model_predict_list:
#         predict_label_name, start, end = predict_label_range
#         predict_str = val_medical_record_dict[id][start:end]
#         # do the postprocessing at here
#         sample_result_str = (id +'\t'+ predict_label_name +'\t'+ str(start) +'\t'+ str(end) +'\t'+ predict_str + "\n")
#         output_string += sample_result_str
#     #print(y)
# if not os.path.exists("./submission"):
#     os.mkdir("./submission")
# with open("./submission/answer.txt", "w", encoding="utf-8") as f:
#     f.write(output_string)


# print("### Other")
# for i, sample in enumerate(val_dataset):
#   model.eval()
#   x, y, id = sample
#   print(id)
#   encodings = tokenizer(x, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True")
#   encodings["input_ids"] = encodings["input_ids"].to(device)
#   encodings["attention_mask"] = encodings["attention_mask"].to(device)
#   outputs = model(encodings["input_ids"], encodings["attention_mask"])
#   #output = softmax(outputs.logits)
#   model_predict_table = torch.argmax(outputs.squeeze(), dim=-1)
#   #print(model_predict_table)
#   print(decode_model_result(model_predict_table, encodings["offset_mapping"][0], labels_type_table))
#   print(y)
#   break