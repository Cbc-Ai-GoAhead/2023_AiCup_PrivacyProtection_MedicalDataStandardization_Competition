#https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/
#https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
#https://www.zhihu.com/question/32745078
#https://zhuanlan.zhihu.com/p/504204038
from pprint import pprint as pp
#https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/
#https://blog.csdn.net/weixin_42223207/article/details/119336324
def read_text_from_file(file_path):
  medical_record_dict ={}
  for data_path in file_path:
    # print("data_path = {}" .format(data_path))
    file_id = data_path.split("/")[-1].split(".txt")[0]
    # print("file_id = {}" .format(file_id))
    
    with open(data_path, "r", encoding="utf-8") as f:
      file_text = f.read()
      # file_text = f.read().splitlines()
      # 文本直接整個讀進來
      # print("file txt =")
      # pp(file_text)
      medical_record_dict[file_id] = file_text
      # print(train_medical_record_dict[file_id] )
    # break

  return medical_record_dict

def read_test_text_from_file(file_path):
  medical_record_dict ={}
  for data_path in file_path:
    # print("data_path = {}" .format(data_path))
    file_id = data_path.split("/")[-1].split(".txt")[0]
    # print("file_id = {}" .format(file_id))
    
    with open(data_path, "r", encoding="utf-8") as f:
      file_text = f.read()
      # file_text = f.read().splitlines()
      # 文本直接整個讀進來
      # print("file txt =")
      # pp(file_text)
      medical_record_dict[file_id] = file_text
      # print(train_medical_record_dict[file_id] )

  return medical_record_dict
def testing_create_label_dict(label_path):

  label_dict = {} #y
  date_label_dict = {} #DATE TIME DURATION SET
  with open(label_path, "r", encoding="utf-8") as f:
    file_text = f.read()
    file_text = file_text.strip("\ufeff").strip() #train file didn't remove this head
  for line in file_text.split("\n"):
    sample = line.split("\t") #(id, label, start, end, query) or (id, label, start, end, query, time_org, timefix)
    # print("sample ={}".format(sample))
    sample[1] = int(sample[1])
    # sample[2], sample[3] = (int(sample[2]), int(sample[3])) #start, end = (int(start), int(end))


    # print(sample)
    """
    ['file1436', 'TIME', 3651, 3670, '2761-04-09 00:00:00', '2761-04-09T00:00:00']
    ['file1436', 'PATIENT', 3682, 3695, 'ELLIS-GEFFERS']
    ['file14362', 'IDNUM', 8, 18, '86L006749H']
    """
    # sample[0] is filename
    # print(sample[0])
    if sample[0] not in label_dict.keys():
      #DATE TIME DURATION SET
      # if sample[1] == ('DATE' or "TIME" or "DURATIOM" or "SET"):
      #   date_label_dict[sample[0]] = [sample[1:]]

      label_dict[sample[0]] = [sample[1:]]
        
      
      # print(label_dict)
    else:
      # if sample[1] == ('DATE' or "TIME" or "DURATIOM" or "SET"):
      #   date_label_dict[sample[0]] = [sample[1:]]
      label_dict[sample[0]].append(sample[1:]) # 組成group list
        
      # 144': [['IDNUM', 13, 23, '77H941695D'], ['MEDICALRECORD', 24, 34, '772941.RZP'],]
    # print(label_dict)
  return label_dict#, date_label_dict
    
def create_label_dict(label_path):
  label_dict = {} #y
  date_label_dict = {} #DATE TIME DURATION SET
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
    
    # sample[0] is filename
    # print(sample[0])
    if sample[0] not in label_dict.keys():
      #DATE TIME DURATION SET
      if sample[1] == ('DATE' or "TIME" or "DURATIOM" or "SET"):
        date_label_dict[sample[0]] = [sample[1:]]
      label_dict[sample[0]] = [sample[1:]]
        
      
      # print(label_dict)
    else:
      if sample[1] == ('DATE' or "TIME" or "DURATIOM" or "SET"):
        date_label_dict[sample[0]] = [sample[1:]]
      label_dict[sample[0]].append(sample[1:]) # 組成group list
        
      # 144': [['IDNUM', 13, 23, '77H941695D'], ['MEDICALRECORD', 24, 34, '772941.RZP'],]
    # print(label_dict)
  return label_dict, date_label_dict
def extract_date_lable(train_label_dict, train_id_list):
  print("extract_date_lable")
  for sample_id in train_id_list:
    sample_id_lablel_group_list = train_label_dict[sample_id]
    print(sample_id_lablel_group_list)
    for sample_id_list in sample_id_lablel_group_list:
      print(sample_id_list)
      break
    break
  print("----------extract_date_lable")
def testing_decode_model_result(model_predict_table, offsets_mapping, labels_type_table):
  model_predict_list = model_predict_table.tolist()
  print("----------------------------------------------------")
  print("model_predict_list = {}".format(model_predict_list))
  id_to_label = {id:label for label, id in labels_type_table.items()}
  predict_y = []
  pre_label_id = 0
  for position_id, label_id in enumerate(model_predict_list):
    print("position_id={}, label_id={}".format(position_id, label_id))
    if label_id!=0:
      if pre_label_id!=label_id:
        print("offsets_mapping[position_id][0] = {}".format(offsets_mapping[position_id][0]))
        start = int(offsets_mapping[position_id][0])
      print("offsets_mapping[position_id][1] = {}".format(offsets_mapping[position_id][1]))
      end = int(offsets_mapping[position_id][1])
    if pre_label_id!=label_id and pre_label_id!=0:
      predict_y.append([id_to_label[pre_label_id], start, end])
    pre_label_id = label_id
  if pre_label_id!=0:
    predict_y.append([id_to_label[pre_label_id], start, end])
  return predict_y
    
def decode_model_result(model_predict_table, offsets_mapping, labels_type_table):
  model_predict_list = model_predict_table.tolist()
  id_to_label = {id:label for label, id in labels_type_table.items()}
  predict_y = []
  pre_label_id = 0
  for position_id, label_id in enumerate(model_predict_list):
    if label_id!=0:
      if pre_label_id!=label_id:
        start = int(offsets_mapping[position_id][0])
      end = int(offsets_mapping[position_id][1])
    if pre_label_id!=label_id and pre_label_id!=0:
      predict_y.append([id_to_label[pre_label_id], start, end])
    pre_label_id = label_id
  if pre_label_id!=0:
    predict_y.append([id_to_label[pre_label_id], start, end])
  return predict_y

def calculate_batch_score(batch_labels, model_predict_tables, offset_mappings, labels_type_table):
    score_table = {"TP":0, "FP":0, "TN":0}
    batch_size = model_predict_tables.shape[0]
    for batch_id in range(batch_size):
        smaple_prediction = decode_model_result(model_predict_tables[batch_id], offset_mappings[batch_id], labels_type_table)
        smaple_ground_truth = batch_labels[batch_id]
        #print(smaple_prediction)
        #print(smaple_ground_truth)
        # do the post_processing at here
        # calculeate TP, TN, FP
        smaple_ground_truth = set([tuple(token) for token in smaple_ground_truth])
        smaple_prediction = set([tuple(token) for token in smaple_prediction])
        score_table["TP"] += len( smaple_ground_truth & smaple_prediction)
        score_table["TN"] += len( smaple_ground_truth - smaple_prediction)
        score_table["FP"] += len( smaple_prediction - smaple_ground_truth)
    if (score_table["TP"] + score_table["FP"])==0 or (score_table["TP"] + score_table["TN"])==0:
      return 0, 0, 0

    Precision = score_table["TP"] / (score_table["TP"] + score_table["FP"])
    Recall = score_table["TP"] / (score_table["TP"] + score_table["TN"])
    if(Precision + Recall) ==0:
      return 0, 0, 0

    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    return Precision, Recall, F1_score




import torch
from torch.utils.data import Dataset, DataLoader
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
    # 文本丟入 encoding進行編碼 進行斷詞
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
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
class testdataset_Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      
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
    # 文本丟入 encoding進行編碼 進行斷詞
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
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
    # 文本丟入 encoding進行編碼 進行斷詞
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True
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
####
## Decode
####

def decode_model_result(model_predict_table, offsets_mapping, labels_type_table):
    model_predict_list = model_predict_table.tolist()
    id_to_label = {id:label for label, id in labels_type_table.items()}
    predict_y = []
    pre_label_id = 0
    for position_id, label_id in enumerate(model_predict_list):
      if label_id!=0:
        if pre_label_id!=label_id:
          start = int(offsets_mapping[position_id][0])
        end = int(offsets_mapping[position_id][1])
      if pre_label_id!=label_id and pre_label_id!=0:
        predict_y.append([id_to_label[pre_label_id], start, end])
      pre_label_id = label_id
    if pre_label_id!=0:
      predict_y.append([id_to_label[pre_label_id], start, end])
    return predict_y

def calculate_batch_score(batch_labels, model_predict_tables, offset_mappings, labels_type_table):
  score_table = {"TP":0, "FP":0, "TN":0}
  batch_size = model_predict_tables.shape[0]
  for batch_id in range(batch_size):
      smaple_prediction = decode_model_result(model_predict_tables[batch_id], offset_mappings[batch_id], labels_type_table)
      smaple_ground_truth = batch_labels[batch_id]
      #print(smaple_prediction)
      #print(smaple_ground_truth)
      # do the post_processing at here
      # calculeate TP, TN, FP
      smaple_ground_truth = set([tuple(token) for token in smaple_ground_truth])
      smaple_prediction = set([tuple(token) for token in smaple_prediction])
      score_table["TP"] += len( smaple_ground_truth & smaple_prediction)
      score_table["TN"] += len( smaple_ground_truth - smaple_prediction)
      score_table["FP"] += len( smaple_prediction - smaple_ground_truth)
  if (score_table["TP"] + score_table["FP"])==0 or (score_table["TP"] + score_table["TN"])==0:
    return 0, 0, 0

  Precision = score_table["TP"] / (score_table["TP"] + score_table["FP"])
  Recall = score_table["TP"] / (score_table["TP"] + score_table["TN"])
  if(Precision + Recall) ==0:
    return 0, 0, 0

  F1_score = 2 * (Precision * Recall) / (Precision + Recall)
  return Precision, Recall, F1_score

def print_dataset_loaderstatus(train_dataset, train_dataloader, labels_type_table, BACH_SIZE):
    #####
    ##  Testing DataSet
    #####
    print(len(train_dataset))
    for sample in train_dataset:
        train_x, train_y,_ = sample
        # print("train_x = {} , train_y={}".format(train_x, train_y))
        # print(train_y)
        break
    print("-----------------train_dataset")
    # print("DataLoader")
    # print(len(train_dataloader))
    for sample in train_dataloader:
        # batch_medical_record, encodings, batch_labels_tensor, batch_labels
        print("sample = {}".format(sample))
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
def print_annotated_medical_report(tokenizer,train_dataset, train_medical_record_dict, train_label_dict):
    '''
    測試讀取的 medical report 內容
    #　可以在考慮把＼ｎ去掉  和 \t
    output : 全部的 sequence pairs
    '''
    print("--------------------------")
    print("#### Tokenizer")
    #some exist id "10", "11", "12", "file16529"
    id = "file10996"#"file9830"
    print(train_medical_record_dict[id])
    pp(train_label_dict[id])
    print("Number of character in medical_record:", len(train_medical_record_dict[id]))

    example_medical_record = train_medical_record_dict[id]
    example_labels = train_label_dict[id]
    encodings = tokenizer(example_medical_record, padding=True, return_tensors="pt", return_offsets_mapping="True")

    print("not truncation len= {}".format(len(encodings.input_ids)))
    encodings = tokenizer(example_medical_record, padding=True, truncation=True,return_tensors="pt", return_offsets_mapping="True")
    print("truncation len= {}".format(len(encodings.input_ids)))
    print(encodings.keys())
    #print(encodings["input_ids"])
    #print(encodings["attention_mask"])
    print("encodings[offset_mapping] = {} ".format(encodings["offset_mapping"]))
    print("encodings[offset_mapping] shape= {} ".format(encodings["offset_mapping"].shape))
    #print(tokenizer.decode(encodings["input_ids"][0])) #get the original text


    print("encodings[input_ids].shape = {} ".format(encodings["input_ids"].shape))
    print("encodings[attention_mask]. shape= {} ".format(encodings["attention_mask"].shape))
    print("len(encodings[offset_mapping][0])= {} ".format(len(encodings["offset_mapping"][0])))
    # print(encodings["input_ids"].shape)
    # print(encodings["attention_mask"].shape)
    # print(len(encodings["offset_mapping"][0]))

    print("### Testing find_token_ids (the funtion in Privacy_protection_dataset)")

    print("train_label_dict[id][3][0]={}, train_label_dict[id][3][1]={}, train_label_dict[id][3][2]={}" .format(train_label_dict[id][3][0], train_label_dict[id][3][1], train_label_dict[id][3][2]))
    encodeing_start, encodeing_end = train_dataset.find_token_ids(train_label_dict[id][3][1], train_label_dict[id][3][2], encodings["offset_mapping"][0])
    print("encodeing_start={} encodeing_end={}".format(encodeing_start, encodeing_end))

    #get the original text
    #
    print(tokenizer.decode(encodings["input_ids"][0][encodeing_start:encodeing_end])) #sometime will error
    # 有時候 encode mapping會錯誤
    print("---Offsetmapping")
    decode_start_pos = int(encodings["offset_mapping"][0][encodeing_start][0])
    decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end-1][1])
    # decode_end_pos = int(encodings["offset_mapping"][0][encodeing_end][1])
    print(decode_start_pos, decode_end_pos)
    print(train_medical_record_dict[id][decode_start_pos:decode_end_pos])
