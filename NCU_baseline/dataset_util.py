import torch
from torch.utils.data import Dataset, DataLoader

class Longformer_Privacy_protection_dataset(Dataset):
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

  def encode_labels_position(self, batch_medical_record,batch_id_list, encodings, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):#offset_mapping用意是?
      encodeing_labels = []
      # print("sample_labels={}, sample_offsets ={}".format(sample_labels, sample_offsets))
      for label in sample_labels:
        # tokenizer後 id 要重新排?
        # print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
        #['DATE', -3828, -3820] label位置會變成負的
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)#label 的位置也要做position encoding
        
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # find token id 會出錯
        # 所以要做 offsetmapping
        # print("------")
        # label 1 label2會變成負數
        # print("label = {}, label[1]={}, label[2]={}, sample_offsets={}".format(label, label[1], label[2], sample_offsets))
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # 會有 inf 造成錯誤
        # print("offset_mapping[0] = {}".format(offset_mapping[0]))
        # print("offset_mapping[0][encodeing_start][0] = {}".format(offset_mapping[0][encodeing_start][0]))
        # decode_start_pos = int(offset_mapping[0][encodeing_start][0])
        # decode_end_pos = int(offset_mapping[0][encodeing_end-1][1])
        # decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end][1])
        # print(decode_start_pos, decode_end_pos)
        # print("batch_id_list[0] ={}".format(batch_id_list[0]))
        # print("batch_medical_record[int(batch_id_list[0])] = {}".format(batch_medical_record))
        # batch_szie=1只有一個值
        # print("batch_medical_record[0][decode_start_pos:decode_end_pos]={}".format(batch_medical_record[0][decode_start_pos:decode_end_pos]))

        encodeing_labels.append([label[0], encodeing_start, encodeing_end])
        # encodeing_labels.append([label[0], decode_start_pos, decode_end_pos])
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
    # print("#####batch_medical_record ={}".format(batch_medical_record))
    # sample 0: 第id的文本, sample 1: label, sample 2: start_positioin, sample 3: End_position, sample 4: query
    # sample  0 取出文本 ('\nEpisode No:  62E239483S\n621239.MVH\n\n
    # sample  1 : [['IDNUM', 14, 24], ['MEDICALRECORD', 25, 35]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    # print("batch_labels = {} , batch_id_list={}".format(batch_labels, batch_id_list))
    # print("batch_medical_record len = {}".format(len(batch_medical_record[0])))
    # print("batch_labels len = {}".format(len(batch_labels[0])))
    # 文本丟入 encoding進行編碼 進行斷詞#512
    #encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, max_length=256,return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, max_length=128,return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # >22G
    #encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True,max_length=2048,return_tensors="pt", return_offsets_mapping="True") # truncation=True

    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True,max_length=1024,return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # truncation=True 代表有斷詞
    # print("#####encodings  ={}".format(encodings['input_ids'][0]))
    # print("#####encodings  len={}".format(len(encodings['input_ids'][0])))
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    batch_labels_position_encoded = self.encode_labels_position(batch_medical_record,batch_id_list, encodings, batch_labels, encodings["offset_mapping"])
    # print("batch_labels_position_encoded ={}".format(batch_labels_position_encoded))
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels

    
class Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      if mode == "train" or mode == "validation":
        self.id_list = list(medical_record_dict.keys())
        self.x = list(medical_record_dict.values())
        # self.y = [[labels[:3] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]
        self.y = [[labels[:4] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]

  def __getitem__(self, index):
      return self.x[index], self.y[index], self.id_list[index]

  def __len__(self):
      return len(self.x)
  def label_encoding():
      tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
  def find_token_ids(self, label_start, label_end, offset_mapping):
    """find the correct labels ids after tokenizer"""
    # 使用 id 原始的位置來對應 [50, 53] 不是使用轉成 token的id 來尋找位置
    encodeing_start = float("inf") #max
    encodeing_end = 0
    # print("find_token_ids")
    for token_id, token_range in enumerate(offset_mapping):
      
      token_start, token_end = token_range
      # print("token id ={}" .format(token_id))
      # print("token_start={}, token_end={}" .format(token_start, token_end))
      #if token range one side out of label range, still take the token
      if token_start == 0 and token_end == 0: #special tocken
        continue
      if label_start<token_end and label_end>token_start:
        
          # token_id 代表第幾的字
        if token_id<encodeing_start:
          encodeing_start = token_id
        encodeing_end = token_id+1
    # 會取出在第幾個 token id 
    # print("--Final token_id ={}, token_id_start={}, token_id_end={}" .format(token_id, encodeing_start, encodeing_end))
    return encodeing_start, encodeing_end

  def encode_labels_position(self, batch_medical_record,batch_id_list, encodings, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    # print("batch_lables ={}".format(batch_lables))
    # print("offset mapping ={}".format(offset_mapping))
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):#offset_mapping用意是?
      encodeing_labels = []
      # print("sample_labels={}, sample_offsets ={}".format(sample_labels, sample_offsets))
      for label in sample_labels:
        # tokenizer後 id 要重新排?
        # print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
        #['DATE', -3828, -3820] label位置會變成負的
        
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)#label 的位置也要做position encoding
        # 找到 token id  他有個字的起始位置
        # token id =163
        # token_start=388, token_end=390
        # token id =164
        # token_start=390, token_end=392
        # token id =165
        # token_start=392, token_end=394
        # token id =166
        # token_start=394, token_end=395
        # token id =167
        # token_start=395, token_end=396
        # token id =168
        # token_start=397, token_end=398
          
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # find token id 會出錯
        # 所以要做 offsetmapping
        # print("------")
        # label 1 label2會變成負數
        # print("label = {}, label[1]={}, label[2]={}, sample_offsets={}".format(label, label[1], label[2], sample_offsets))
        # print("label = {}, label[1]={}, label[2]={} ".format(label, label[1], label[2] ))
        # print("token_start={}, token_end={}".format(encodeing_start, encodeing_end))
        # 會有 inf 造成錯誤
        # print("offset_mapping[0] = {}".format(offset_mapping[0]))
          
        # print("offset_mapping[0][encodeing_start][0] = {}".format(offset_mapping[0][encodeing_start][0]))
        # print("offset_mapping[0][encodeing_start][0] = {}".format(offset_mapping[0][encodeing_end-1][1]))
        # if encodeing_start=="inf":
        #      print("--encodeing inf")
        #      print("start = {}, end={}".format(encodeing_start, encodeing_end-1))
        #      print("label = {}" .format(label))
        #      print("batch_medical_record = {}".format(batch_medical_record))
        #      print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
        try:
            #
            # decode_start_pos = int(offset_mapping[0][encodeing_start][0])
            # #比如 4 到 11 取道第10個的結尾
            # decode_end_pos = int(offset_mapping[0][encodeing_end-1][1])
            # 有時候 token id 數量 會超過512 所以要把超過位置的 label 去除 使用 except 來處理
            #直接取用
            #encodeing_labels.append([label[0], decode_start_pos, decode_end_pos])
            encodeing_labels.append([label[0], encodeing_start, encodeing_end])
        except:
            print("label position out of toke id 512 start = {}, end={}".format(encodeing_start, encodeing_end-1))
            print("label = {}" .format(label))
            print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
       
        # decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end][1])
        # print("decode tokenid context start position ={}, end position ={}".format(decode_start_pos, decode_end_pos))
        # print("batch_id_list[0] ={}".format(batch_id_list[0]))
        # print("batch_medical_record[int(batch_id_list[0])] = {}".format(batch_medical_record))
        # batch_szie=1只有一個值
        # print("batch_medical_record[0][decode_start_pos:decode_end_pos]={}".format(batch_medical_record[0][decode_start_pos:decode_end_pos]))

        #encodeing_labels.append([label[0], encodeing_start, encodeing_end])
        
      batch_encodeing_labels.append(encodeing_labels)
    return batch_encodeing_labels

  def create_labels_tensor(self, batch_shape:list, batch_labels_position_encoded:list):
    # print("-----Create label Tensor")
    if batch_shape[-1]> 4096:
      batch_shape[-1] = 4096
    labels_tensor = torch.zeros(batch_shape)
    # print("---in create_labels_tensor")
    # print("batch_shape = {}" .format(batch_shape))#(1, 512) # Bert只能有512個token
    # print("batch_shape[1] = {}" .format(batch_shape[1]))
    for sample_id in range(batch_shape[0]):# 取出一個batch 內的值
      # print("sample_id = {}".format(sample_id)) # batch_id 的編號
      for label in batch_labels_position_encoded[sample_id]:
        # print("label ={}".format(label))
        #label =['MEDICALRECORD', 1, 8]
        #label =['PATIENT', 8, 17]
        label_id = self.labels_type_table[label[0]]
        # print("label_id = {}".format(label_id))
        # 應該是要給 context編碼過的id
        # print("label_id = {}, label[0]={}".format(label_id, label[0]))
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
    # print("--------------Collate fn")
    # print("batch_items ={}".format(batch_items))
    
    # print(batch_items)
    batch_medical_record = [sample[0] for sample in batch_items] #(id, label, start, end, query) or (id, label, start, end, query, time_org, timefix)
    # print("batch_items context len ={}".format(len(batch_medical_record[0])))
    # print("#####batch_medical_record ={}".format(batch_medical_record))
    # sample 0: 第id的文本, sample 1: label, sample 2: start_positioin, sample 3: End_position, sample 4: query
    # sample  0 取出文本 ('\nEpisode No:  62E239483S\n621239.MVH\n\n
    # batch_labels = [[['DOCTOR', 41, 43], ['DOCTOR', 44, 46], ['DATE', 50, 56], ['DOCTOR', 88, 99]]] , batch_id_list=['109_0_4']
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    # sample  1 : [['IDNUM', 14, 24], ['MEDICALRECORD', 25, 35]
    # print("batch_labels = {} , batch_id_list={}".format(batch_labels, batch_id_list))
    # 文本丟入 encoding進行編碼 進行斷詞
    # print("batch_medical_record = {}" .format(batch_medical_record))
    # print("len batch_medical_record = {}" .format(len(batch_medical_record)))
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # 這邊有把truncation　去掉
    # encodings = self.tokenizer(batch_medical_record, padding=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # truncation=True 代表有斷詞
    # print("#####encodings ={}".format(encodings))
    # print("record len = {}, encoding len = {}".format(len(batch_medical_record[0]), len(encodings["input_ids"][0])))
    # result = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0], skip_special_tokens=False)
    # print("result len ={}".format(len(result)))
    # print(result)
    # print("------- end ecnoding batchitem")
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    # !!!!batch_label 作encoding 去找出tokeinzer 後文本的位置, 因為 tokenizer完 文本也會被編碼
    # encodings : 文本編碼後的內容
    # offsetmapping 就是重新組成字串後文字所在的位置, 只有CLS 和 SEP 是 [0,0]
    # \n 會被轉成?
    batch_labels_position_encoded = self.encode_labels_position(batch_medical_record,batch_id_list, encodings, batch_labels, encodings["offset_mapping"])
    # print("batch_labels_position_encoded ={}".format(batch_labels_position_encoded))
    #print(encodings["offset_mapping"])
    # print("batch_labels_position_encoded={}".format(batch_labels_position_encoded)) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    # 回傳 label id 的起始到結束位置
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels

class testdataset_Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      
      self.id_list = list(medical_record_dict.keys())
      self.x = list(medical_record_dict.values())
      self.y = [[labels[:3] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]

  def __getitem__(self, index):
      return self.x[index], self.y[index], self.id_list[index]
      # return index,index,index

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

  def encode_labels_position(self, batch_medical_record,batch_id_list, encodings, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):#offset_mapping用意是?
      encodeing_labels = []
      # print("sample_labels={}, sample_offsets ={}".format(sample_labels, sample_offsets))
      for label in sample_labels:
        # tokenizer後 id 要重新排?
        # print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
        #['DATE', -3828, -3820] label位置會變成負的
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)#label 的位置也要做position encoding
        
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # find token id 會出錯
        # 所以要做 offsetmapping
        # print("------")
        # label 1 label2會變成負數
        # print("label = {}, label[1]={}, label[2]={}, sample_offsets={}".format(label, label[1], label[2], sample_offsets))
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # 會有 inf 造成錯誤
        # print("offset_mapping[0] = {}".format(offset_mapping[0]))
        # print("offset_mapping[0][encodeing_start][0] = {}".format(offset_mapping[0][encodeing_start][0]))
        decode_start_pos = int(offset_mapping[0][encodeing_start][0])
        decode_end_pos = int(offset_mapping[0][encodeing_end-1][1])
        # decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end][1])
        # print(decode_start_pos, decode_end_pos)
        # print("batch_id_list[0] ={}".format(batch_id_list[0]))
        # print("batch_medical_record[int(batch_id_list[0])] = {}".format(batch_medical_record))
        # batch_szie=1只有一個值
        # print("batch_medical_record[0][decode_start_pos:decode_end_pos]={}".format(batch_medical_record[0][decode_start_pos:decode_end_pos]))

        #encodeing_labels.append([label[0], encodeing_start, encodeing_end])
        encodeing_labels.append([label[0], decode_start_pos, decode_end_pos])
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
        # print("label_id = {}, label[0]={}".format(label_id, label[0]))
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
    # print("#####batch_medical_record ={}".format(batch_medical_record))
    # sample 0: 第id的文本, sample 1: label, sample 2: start_positioin, sample 3: End_position, sample 4: query
    # sample  0 取出文本 ('\nEpisode No:  62E239483S\n621239.MVH\n\n
    # sample  1 : [['IDNUM', 14, 24], ['MEDICALRECORD', 25, 35]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    # print("batch_labels = {} , batch_id_list={}".format(batch_labels, batch_id_list))
    # 文本丟入 encoding進行編碼 進行斷詞
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True,max_length=1024 , return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # truncation=True 代表有斷詞
    # print("#####encodings ={}".format(encodings))
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    batch_labels_position_encoded = self.encode_labels_position(batch_medical_record,batch_id_list, encodings, batch_labels, encodings["offset_mapping"])
    #print(encodings["offset_mapping"])
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels

class Long_Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.start = 0
      self.end = len(self.texts)
      self.batch_size = batch_size
      self.max_sub_sentence_len = max_sub_sentence_len
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      self.visit_order = [i for i in range(self.end)]

      if shuffle:
        random.shuffle(self.visit_order)

      if mode == "train" or mode == "validation":
        self.id_list = list(medical_record_dict.keys())
        self.x = list(medical_record_dict.values())
        self.y = [[labels[:3] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]

  def __getitem__(self, index):
      return self.x[index], self.y[index], self.id_list[index]

  def __len__(self):
      return len(self.x)
  def __split_long_text(self, text: str) -> list:
      """
      用於迭代器返回數據樣本時將文本進行切割 切隔成數個

      Args:
          text (str): 長文本, e.g. -> "NCU真偉大"
      
      Returns:
          [list] -> ["NCU", "真偉大"]（假設self.max_sub_sentence_len = 3）
      """
      sub_texts, start, length = [], 0, len(text)
      while start < length:
          sub_texts.append(text[start: start + self.max_sub_sentence_len])
          start += self.max_sub_sentence_len
      return sub_texts
  def __next__(self) -> dict:
    """
    迭代器，每次return sub數據，長文本會切割成若個短句。

    Raises:
        StopIteration: [description]

    Returns:
        [dict] -> {
            'text': [sub_sentence 1, sub_sentence 2, ...],
            'label': 1
        }
    """
    if self.start < self.end:
        ret = self.start
        batch_end = ret + self.batch_size
        self.start += self.batch_size
        currents = self.visit_order[ret: batch_end]
        return {'text': [self.__split_long_text(self.texts[c]) for c in currents], 'label': [int(self.labels[c]) for c in currents]}
    else:
        self.start = 0
        raise StopIteration
  def __iter__(self):
        return self
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
    return encodeing_sencode_labels_positiontart, encodeing_end

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
    # 4096用意是？
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
    # truncation=True 代表有斷詞
      
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    batch_labels_position_encoded = self.encode_labels_position(batch_labels, encodings["offset_mapping"])
    #print(encodings["offset_mapping"])
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels
class ori_Privacy_protection_dataset(Dataset):
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

  def encode_labels_position(self, batch_medical_record,batch_id_list, encodings, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):#offset_mapping用意是?
      encodeing_labels = []
      # print("sample_labels={}, sample_offsets ={}".format(sample_labels, sample_offsets))
      for label in sample_labels:
        # tokenizer後 id 要重新排?
        # print("label[1]={}, label[2]={}, sample_offsets={}".format(label[1], label[2], sample_offsets))
        #['DATE', -3828, -3820] label位置會變成負的
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)#label 的位置也要做position encoding
        
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # find token id 會出錯
        # 所以要做 offsetmapping
        # print("------")
        # label 1 label2會變成負數
        # print("label = {}, label[1]={}, label[2]={}, sample_offsets={}".format(label, label[1], label[2], sample_offsets))
        # print("encodeing_start={}, encodeing_end={}".format(encodeing_start, encodeing_end))
        # 會有 inf 造成錯誤
        # print("offset_mapping[0] = {}".format(offset_mapping[0]))
        # print("offset_mapping[0][encodeing_start][0] = {}".format(offset_mapping[0][encodeing_start][0]))
        # decode_start_pos = int(offset_mapping[0][encodeing_start][0])
        # decode_end_pos = int(offset_mapping[0][encodeing_end-1][1])
        # decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end][1])
        # print(decode_start_pos, decode_end_pos)
        # print("batch_id_list[0] ={}".format(batch_id_list[0]))
        # print("batch_medical_record[int(batch_id_list[0])] = {}".format(batch_medical_record))
        # batch_szie=1只有一個值
        # print("batch_medical_record[0][decode_start_pos:decode_end_pos]={}".format(batch_medical_record[0][decode_start_pos:decode_end_pos]))

        encodeing_labels.append([label[0], encodeing_start, encodeing_end])
        # encodeing_labels.append([label[0], decode_start_pos, decode_end_pos])
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
    # print("#####batch_medical_record ={}".format(batch_medical_record))
    # sample 0: 第id的文本, sample 1: label, sample 2: start_positioin, sample 3: End_position, sample 4: query
    # sample  0 取出文本 ('\nEpisode No:  62E239483S\n621239.MVH\n\n
    # sample  1 : [['IDNUM', 14, 24], ['MEDICALRECORD', 25, 35]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    # print("batch_labels = {} , batch_id_list={}".format(batch_labels, batch_id_list))
    # print("batch_medical_record len = {}".format(len(batch_medical_record[0])))
    # print("batch_labels len = {}".format(len(batch_labels[0])))
    # 文本丟入 encoding進行編碼 進行斷詞#512
    #encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, max_length=256,return_tensors="pt", return_offsets_mapping="True") # truncation=True
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, max_length=128,return_tensors="pt", return_offsets_mapping="True") # truncation=True
    # truncation=True 代表有斷詞
    # print("#####encodings ={}".format(encodings))
    # encode label
    # 丟入 bert
    # 第0句話輸出的token id =0 第1句話 id =1, padding mask =1 不是padding的id都是1
    batch_labels_position_encoded = self.encode_labels_position(batch_medical_record,batch_id_list, encodings, batch_labels, encodings["offset_mapping"])
    # print("batch_labels_position_encoded ={}".format(batch_labels_position_encoded))
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)# 對應到 label id
    return batch_medical_record, encodings, batch_labels_tensor, batch_labels

