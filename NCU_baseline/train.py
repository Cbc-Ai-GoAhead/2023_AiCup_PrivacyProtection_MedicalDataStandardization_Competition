from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from bert_model import myModel
def evaluate_total_f1(model, val_dataloader, sum_val_F1):
  
  sum_val_F1 = 0
  for batch_x_name, batch_x, batch_y, batch_labels in val_dataloader:
    val_step += 1
    optim.zero_grad()
    batch_x["input_ids"] = batch_x["input_ids"].to(device)
    batch_x["attention_mask"] = batch_x["attention_mask"].to(device)
    batch_y = batch_y.long().to(device)
    outputs = model(batch_x["input_ids"], batch_x["attention_mask"])
    model_predict_tables = torch.argmax(outputs, dim=-1, keepdim=True)
    model_predict_tables = model_predict_tables.squeeze(-1)
    P, R, F1 = calculate_batch_score(batch_labels, model_predict_tables, batch_x["offset_mapping"], labels_type_table)
    if val_step%50==0:
      print("val_F1", F1)
    val_loss = loss_fct(outputs.transpose(-1, -2), batch_y)
    sum_val_F1 += float(F1)
    #print("val_loss", val_loss)
    writer.add_scalar('Loss/val', val_loss, val_step)
    writer.add_scalar('F1/val', F1, val_step)

  # model.save_pretrained(output_dir)
  # output_dir= output_dir+"_token"
  # tokenizer.save_pretrained(output_dir)
  output_dir = "./model_pre/" + "bert-base-cased"+"_"+str(epoch)+"_"+str(BACH_SIZE)+"_"+str(sum_val_F1/len(val_dataloader))
  torch.save(model.state_dict(), output_dir+"dict")
  torch.save(model, output_dir)
  # model.save_model(output_dir)
  # model.save_pretrained(output_dir)

  if sum_val_F1 > base_f1_score:
    base_f1_score = sum_val_F1
    output_dir = "./model_pre/best/" + "best_bert-base-cased"+"_"+str(epoch)+"_"+str(BACH_SIZE)+"_"+str(sum_val_F1/len(val_dataloader))
    # model.save_pretrained(output_dir)
    # output_dir= output_dir+"_token"
    # tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir+"dict")
    torch.save(model, output_dir)
    # model.save_model(output_dir)
    # model.save_pretrained(output_dir)
    decode_model(model, val_dataset)

  # 第2種 inference的方法
  # for i, sample in enumerate(val_dataloader):
    
  #   batch_x_name, encodings, y, batch_labels = sample
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
def decode_model(model, val_dataset):
  #####
  ### Inference
  #####
  
  
  output_string = ""
  for i, sample in enumerate(val_dataset):
    
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
        # Predict_str 會抓到 \n 換行符號 要再處理
        sample_result_str = (id +'\t'+ predict_label_name +'\t'+ str(start) +'\t'+ str(end) +'\t'+ predict_str + "\n")
        output_string += sample_result_str
    #print(y)# y 有跟原本一樣嗎？
  exp_path = "./submission"+"_"+str(epoch)+"_"+str(BACH_SIZE)
  if not os.path.exists(exp_path):
    os.mkdir(exp_path)
  answer_path = exp_path+"/"+"answer.txt"
  with open(answer_path, "w", encoding="utf-8") as f:
    f.write(output_string)
  

def finetune_model(train_dataloader, val_dataloader, val_dataset):
  ####
  ##  Init model
  ####
  model = myModel()
  print(model)
  
  ####
  ##  Hyperameter 
  ####
  BACH_SIZE = 12#1
  #TRAIN_RATIO = 0.9
  LEARNING_RATE = 1e-4
  EPOCH = 1
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = model.to(device) # Put model on device
  optim = AdamW(model.parameters(), lr = LEARNING_RATE)
  loss_fct = CrossEntropyLoss()

  train_step = 0
  val_step = 0
  # one epoch about 8 minutes for T4 on Colab
  # 5G memory needed when BACH_SIZE = 1
  base_f1_score = 0.00
  for epoch in range(EPOCH):
    model.train()
    # x_name,train_x, train_y, _
    for batch_x_name, batch_x, batch_y, batch_labels in train_dataloader:
      train_step += 1
      optim.zero_grad()
      batch_x["input_ids"] = batch_x["input_ids"].to(device)
      batch_x["attention_mask"] = batch_x["attention_mask"].to(device)
      batch_y = batch_y.long().to(device)
      outputs = model(batch_x["input_ids"], batch_x["attention_mask"])
      #print(batch_y.shape)
      train_loss = loss_fct(outputs.transpose(-1, -2), batch_y)
      #print("train_loss", train_loss)
      writer.add_scalar('Loss/train', train_loss, train_step)
    
      # calculate loss
      train_loss.backward()
      # update parameters
      optim.step()
    
    
    

    model.eval()
    evaluate_total_f1(model, val_dataloader, base_f1_score, writer)
    writer.close()
    decode_model(model, )
    
    
    
    
    
    