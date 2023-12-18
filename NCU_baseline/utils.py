#https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/
#https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
#https://www.zhihu.com/question/32745078
#https://zhuanlan.zhihu.com/p/504204038
from pprint import pprint as pp
#https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/
#https://blog.csdn.net/weixin_42223207/article/details/119336324
def reposition(preserve_label_list_group, conetext_start_position):
  #file_id start end value
  processed_preserve_label_list_group = []
  for id_list in preserve_label_list_group:
    value = id_list[3]
    #tmp_val.append(value) # context的內容
    
    print("conetext_start_position = {}".format(conetext_start_position))
    start = id_list[1]- conetext_start_position
    end = id_list[2]- conetext_start_position

    # reinsert to id_list
    id_list[1] = start
    id_list[2] = end
    print("start = {} end={}".format(start, end))
    processed_preserve_label_list_group.append(id_list)
  return processed_preserve_label_list_group
def testing_find_label_value_in_text(text_list, label_list_group, conetext_start_position,conetext_end_position):#label_value_to_text):
  l = []
  # print(t)
  # print(label_value_to_text)
  # processd_label_list_group =[]
  tmp_val = []
  # check label poistion if outlier then remove
  preserve_label_list_group=[]
  for id_list in label_list_group:
    start = id_list[1]
    end = id_list[1]
    if (start>=conetext_start_position) and (end<=conetext_end_position):
      preserve_label_list_group.append(id_list)
    # if start 

  #file_id start end value
  # for id_list in preserve_label_list_group:
  #   value = id_list[3]
  #   tmp_val.append(value) # context的內容
  #   processd_label_list_group.append(id_list)
  #   print("conetext_start_position = {}".format(conetext_start_position))
  #   start = id_list[1]- conetext_start_position
  #   end = id_list[2]- conetext_start_position

  #   # reinsert to id_list
  #   id_list[1] = start
  #   id_list[2] = end
  #   print("start = {} end={}".format(start, end))
  preserve_label_list_group = reposition(preserve_label_list_group, conetext_start_position)
  #如果文本 只有 other的類別要去除
  #先用label 的值來找context有沒有符合
  num_of_label_in_context=0
  for id_list in preserve_label_list_group:
    value = id_list[3]
    if(text_list[-1].find(value)==-1):
      continue
    else:
      num_of_label_in_context+=1
  return num_of_label_in_context, preserve_label_list_group
  for id_list in label_list_group:
    val = id_list[3]
    # 這裡有bug 會尋找到文本的內容
    # 會另外找到文本後半段不相關的內容
    # print("text type={}".format(type(text_list[-1])))
    # print("text[-1]={}".format(text_list[-1]))
    if(text_list[-1].find(val)==-1):
        # l.append("")
        continue
        # return "", l
    else:#有找到文本
        #l.append(label)
        l.append(val) # 先append val 應該是要appendlabel
        # return val, l
        tmp_val.append(val) # context的內容
        processd_label_list_group.append(id_list)
        start = id_list[1]- conetext_start_position
        end = id_list[2]- conetext_start_position
        # print("start = {}, end ={}".format(start, end))
        if start>0:# end >0
          # print("text = {}".format(t))
          #每次進來一個文本
          # print("t[start:end] = {}, val ={}".format(t[0][start:end], val))
          if text_list[0][start:end] == val:

            l.append(val) # 先append val 應該是要appendlabel
            # return val, l
            tmp_val.append(val)
            processd_label_list_group.append(id_list)
  # 如果get val有值就更新 processed_label_dict
  return tmp_val, processd_label_list_group

def find_label_value_in_text(t, label_list_group, new_position):#label_value_to_text):
  l = []
  # print(t)
  # print(label_value_to_text)
  processd_label_list_group =[]
  tmp_val = []
  #file_id start end value
  for id_list in label_list_group:
    val = id_list[3]
    # 這裡有bug 會尋找到文本的內容
    # 會另外找到文本後半段不相關的內容
    if(t[-1].find(val)==-1):
        # l.append("")
        continue
        # return "", l
    else:#有找到文本
        #l.append(label)
        l.append(val) # 先append val 應該是要appendlabel
        # return val, l
        tmp_val.append(val) # context的內容
        processd_label_list_group.append(id_list)
        start = id_list[1]- new_position
        end = id_list[2]- new_position
        # print("start = {}, end ={}".format(start, end))
        if start>0:# end >0
          # print("text = {}".format(t))
          #每次進來一個文本
          # print("t[start:end] = {}, val ={}".format(t[0][start:end], val))
          if t[0][start:end] == val:

            l.append(val) # 先append val 應該是要appendlabel
            # return val, l
            tmp_val.append(val)
            processd_label_list_group.append(id_list)
  # 如果get val有值就更新 processed_label_dict
  return tmp_val, processd_label_list_group

def create_chunks(fileid, text,label_list_group):
  WINDOW_LENGTH = 510
  STRIDE_LENGTH = 510
  chunk_num = 0
  processd_label_list_group=[]


  processed_medical_record_dict = {}
  processed_label_dict = {}

  l = []
  p = 0
  # print("text[p:p+WINDOW_LENGTH] ={}".format(text[p:p+WINDOW_LENGTH]))
  # t.append(text[p:p+WINDOW_LENGTH])
  # print("p = {} p+WINDOW_LENGTH={}".format(p,p+WINDOW_LENGTH))
  # print("t[-1] = {}" .format(t[-1]))
  # print(label[0])
  # print(t[-1].find(label))

  print("-------")
  print("text length = {}".format(len(text)))
  # print(t)
  # print(label_value_to_text)
  # get_val_list, processd_label_list_group= find_label_value_in_text(t, label_list_group)#label_value_to_text)
  # 計算切幾個文本
  print("num of segmentation = {}".format(len(text)//WINDOW_LENGTH))
  #
  num_segment = len(text)//WINDOW_LENGTH
  max_length = num_segment*WINDOW_LENGTH
  remove_over_max_length_label_list_group = []
  for id_list in label_list_group:# 去掉最後超出範圍的, 或是最後一個要由後往前取
    start = id_list[1]
    end = id_list[2]
    print("max_length ={}, start={}, end={}".format(max_length, start, end))
    if (int(start) < max_length) and (int(end)< max_length):#前一份文本
      remove_over_max_length_label_list_group.append(id_list)
  print("ori lablel len ={}, remove label len = {}".format(label_list_group, remove_over_max_length_label_list_group))
  while p+WINDOW_LENGTH < len(text):# 會捨棄最後的文本
    conetext_start_position = chunk_num*WINDOW_LENGTH
    conetext_end_position = (chunk_num+1)*WINDOW_LENGTH
    # get_val_list, processd_label_list_group= find_label_value_in_text(t, label_list_group,new_position)#label_value_to_text)
    # get_val_list, processd_label_list_group= testing_find_label_value_in_text([text[p:p+WINDOW_LENGTH]], remove_over_max_length_label_list_group,conetext_start_position,conetext_end_position)#label_value_to_text)
    num_of_label_in_context, processd_label_list_group= testing_find_label_value_in_text([text[p:p+WINDOW_LENGTH]], remove_over_max_length_label_list_group,conetext_start_position,conetext_end_position)#label_value_to_text)
    # print("chunk_num = {}".format(chunk_num))
    # print("get_val_list= {}, o_val_list={}".format(get_val_list, processd_label_list_group))

    #if len(get_val_list)!=0: #去掉沒有類別的值
    if num_of_label_in_context!=0: #去掉沒有類別的值
      
      fileid+="_"+str(chunk_num)
      print("fileid = {}".format(fileid))
      print("p:p+WINDOW_LENGTH] = {} {}".format(p, p+WINDOW_LENGTH))
      
      processed_medical_record_dict[fileid]=text[p:p+WINDOW_LENGTH]
      print("processed_medical_record_dict[fileid] = {}".format(processed_medical_record_dict[fileid]))
      print("len processed_medical_record_dict = {}" .format(len(processed_medical_record_dict[fileid])))
      processed_label_dict[fileid]=processd_label_list_group
    chunk_num+=1 # 向後位移
    p += STRIDE_LENGTH
      # for processed_label_list in processd_label_list_group:
      #   processed_label_list[1] = processed_label_list[1]-
      # l.extend(o_val_list)
    # print("t={}" .format(t))
    # print("l={}" .format(l))

  print("---Processed Medical Report={}".format(processed_medical_record_dict))
  print("---Processed label Report={}".format(processed_label_dict))
  # print("---Processed Medical Report={}".format(processed_medical_record_dict))
  # print("---Processed label Report={}".format(processed_label_dict))
  return processed_medical_record_dict,processed_label_dict





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

def print_dataset_loaderstatus(train_dataset, train_dataloader,tokenizer, labels_type_table, BACH_SIZE):
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
        # print("sample = {}".format(sample))
        x_name,train_x, train_y, y_label = sample
        print("x_name = {},".format(x_name))
        print("y_label = {},".format(y_label))
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
        # print(train_x)
        # print(tokenizer.convert_ids_to_tokens(train_x["input_ids"].cpu().detach().numpy()))
        print("type = {}".format(train_x["input_ids"]))
        print("type = {}".format(train_x["input_ids"].tolist()))
        print(tokenizer.convert_ids_to_tokens(train_x["input_ids"].tolist()[0]))
        print("len train_x ={}".format(len(train_x)))
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
