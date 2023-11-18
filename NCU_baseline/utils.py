def read_text_from_file(file_path):
  medical_record_dict ={}
  for data_path in file_path:
    # print("data_path = {}" .format(data_path))
    file_id = data_path.split("/")[-1].split(".txt")[0]
    # print("file_id = {}" .format(file_id))
    
    with open(data_path, "r", encoding="utf-8") as f:
      file_text = f.read()
      # 文本一列一列讀進來
      medical_record_dict[file_id] = file_text
      # print(train_medical_record_dict[file_id] )

  return medical_record_dict

def create_label_dict(label_path):
  label_dict = {} #y
  with open(label_path, "r", encoding="utf-8") as f:
    file_text = f.read()
    file_text = file_text.strip("\ufeff").strip() #train file didn't remove this head
  for line in file_text.split("\n"):
    sample = line.split("\t") #(id, label, start, end, query) or (id, label, start, end, query, time_org, timefix)
    sample[2], sample[3] = (int(sample[2]), int(sample[3])) #start, end = (int(start), int(end))
    # print(sample)
    """
    ['file1436', 'TIME', 3651, 3670, '2761-04-09 00:00:00', '2761-04-09T00:00:00']
    ['file1436', 'PATIENT', 3682, 3695, 'ELLIS-GEFFERS']
    ['file14362', 'IDNUM', 8, 18, '86L006749H']
    """
    if sample[0] not in label_dict.keys():
      label_dict[sample[0]] = [sample[1:]]
    else:
      label_dict[sample[0]].append(sample[1:])
  #print(label_dict)
  return label_dict

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
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):#offset_mapping用意是?
      encodeing_labels = []
      for label in sample_labels:
        # tokenizer後 id 要重新排?
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)#label 的位置也要做position encoding
        encodeing_labels.append([label[0], encodeing_start, encodeing_end])
      batch_encodeing_labels.append(encodeing_labels)
    return batch_encodeing_labels

  def create_labels_tensor(self, batch_shape:list, batch_labels_position_encoded:list):
    if batch_shape[-1]> 4096:
      batch_shape[-1] = 4096
    labels_tensor = torch.zeros(batch_shape)
    # print("---in create_labels_tensor")
    # print("batch_shape = {}" .format(batch_shape))#(1, 512) # Bert只能有512個token
    # print("batch_shape[1] = {}" .format(batch_shape[1]))
    for sample_id in range(batch_shape[0]):# 取出一個batch 內的值
      # print("sample_id = {}".format(sample_id))
      for label in batch_labels_position_encoded[sample_id]:
        # print("label ={}".format(label))
        #label =['MEDICALRECORD', 1, 8]
        #label =['PATIENT', 8, 17]
        label_id = self.labels_type_table[label[0]]
        # print("label_id = {}".format(label_id))
        start = label[1] # 取出Position encode編碼過後的位置
        end = label[2]
        if start >= 4096: continue
        elif end >= 4096: end = 4096
        labels_tensor[sample_id][start:end] = label_id#使用2D 儲存smapleid 0第0筆(MEDICALRECORD), 他起始位置與結束位置 ,與 label轉換成的id
        # print("sample_id = {}, start={}, end={}, labels_tensor={}" .format(sample_id, start, end, labels_tensor[sample_id][start:end]))
        """
        b="D.O.B:  09/08/2957"
        >>> b[8:18]
        '09/08/2957'
        """
        """
        會有 null值
        label =['DOCTOR', inf, 0]
        label_id = 1
        label =['DATE', inf, 0]

        """
    return labels_tensor

  def collate_fn(self, batch_items:list):
    """the calculation process in dataloader iteration"""
    # print("batch_items =")
    # print(batch_items)
    batch_medical_record = [sample[0] for sample in batch_items] #(id, label, start, end, query) or (id, label, start, end, query, time_org, timefix)
    # sample 0: 第id的文本, sample 1: label, sample 2: start_positioin, sample 3: End_position, sample 4: query
    # sample  0 取出文本 ('\nEpisode No:  62E239483S\n621239.MVH\n\n
    # sample  1 : [['IDNUM', 14, 24], ['MEDICALRECORD', 25, 35]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]

    # 文本丟入 encoding進行編碼 進行斷詞
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    batch_labels_position_encoded = self.encode_labels_position(batch_labels, encodings["offset_mapping"])
    #print(encodings["offset_mapping"])
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels