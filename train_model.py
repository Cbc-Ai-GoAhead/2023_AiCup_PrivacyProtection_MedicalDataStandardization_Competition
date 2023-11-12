import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import os, sys
import numpy as np
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_epoch_model(optimizer, model, bucket_train_dataloader):
    # each epoch init
    total_loss = 0
    for step, (seqs, labels, masks) in enumerate(tqdm(bucket_train_dataloader)):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss

    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))
    print(sample_text(model, tokenizer, text=predict_text))
    torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_Finial.pt'))
    if avg_train_loss < min_loss:
        min_loss = avg_train_loss
        torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_best.pt'))
    return None
def train_model(optimizer, model, bucket_train_dataloader, device, model_name, num_epochs=10000):
    #need tensorboard?
    #writer = SummaryWriter('logs/{}'.format(model_tag))
    model_dir = f"{model_name}"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model = model.to(device)
    min_loss = 9999
    predict_text = special_tokens_dict['bos_token'] + "MANILDRA  NSW  2865"

    # Iterate through each epoch and call our train_epoch function
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        model.train()

        
        predictions, true_labels = [], []
        # epoch_loss = train_epoch_lstm(optimizer, model, loader)

        total_loss = train_epoch_model(optimizer, model, bucket_train_dataloader)#train_epoch(train_loader,model, args.lr,optimizer, device)


        avg_train_loss = total_loss / len(bucket_train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        print(sample_text(model, tokenizer, text=predict_text))
        torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_Finial.pt'))
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_best.pt'))

            
def sample_text(model, tokenizer, text, n_words=100):
    '''
    input : model, tokenizer, text(句子 string), n_words(生成字數限制)
    output : 模型預測結果 (string)
    '''
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None

    with torch.no_grad():
        for _ in range(n_words):
            out = model(inputs, past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            text.append(inputs.item())
            if tokenizer.decode(inputs.item()) == eos:
                break

    return tokenizer.decode(text)
def get_anno_format(sentence , infos , boundary):
    '''
    input : 句子(string), 模型預測phi資訊(string), 上個句子結尾索引(integer)
    將模型輸出的 phi 資訊對應 sentence
    儲存於字典並append於list
    output : 儲存phi字典的list
    '''
    anno_list = []
    lines = infos.split("\n")
    # 創建一個字典，用來存儲PHI信息的對應
    normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
    phi_dict = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            phi_dict[parts[0]] = parts[1]
    for phi_key, phi_value in phi_dict.items():
        normalize_time = None
        if phi_key in normalize_keys:
            if '=>' in phi_value:
                temp_phi_values = phi_value.split('=>')
                phi_value = temp_phi_values[0]
                normalize_time = temp_phi_values[-1]
            else:
                normalize_time = phi_value
        if phi_value not in sentence or len(phi_value) < 1:
            continue
        st_idx = sentence.find(phi_value)
        ed_idx = st_idx + len(phi_value)
        item_dict = {
                    'phi' : phi_key,
                    'st_idx' : st_idx + boundary,
                    'ed_idx' : ed_idx + boundary,
                    'entity' : phi_value,
        }
        if normalize_time is not None:
            item_dict['normalize_time'] = normalize_time
        anno_list.append(item_dict)
    return anno_list

def predict_sent(sents):
    '''
    input : 一篇病理報告全部的句子(list)
    output : 上傳格式的 phi 資訊
    '''
    boundary = 0
    annotations = []

    for sent in sents:
        decode_phase = sample_text(model, tokenizer, text=special_tokens_dict['bos_token'] + sent , n_words=200)
        if special_tokens_dict['sep_token'] in decode_phase:
            try:
                _ , phi_infos = decode_phase.split(special_tokens_dict['sep_token'])
            except:
                continue

            if "PHI:Null" not in phi_infos:
                annotation = get_anno_format(sent , phi_infos , boundary)
                annotations.extend(annotation)
        boundary += len(sent)
    return annotations

def predict_file(txts , write_file):
    '''
    寫出上傳格式的資訊
    '''
    with open(write_file , 'w' , encoding='utf-8') as fw:
        for txt in (txts):
            test_sents = read_file(txt)
            anno_infos = predict_sent(test_sents)
            txt_name = txt.split('\\')[-1].replace('.txt' , '')
            for anno_info in anno_infos:
                fw.write(txt_name + '\t')
                fw.write(f"{anno_info['phi']}\t")
                fw.write(f"{anno_info['st_idx']}\t")
                fw.write(f"{anno_info['ed_idx']}\t")
                if anno_info['phi'] in ['DATE' , "TIME" , "DURATION" , "SET"]:
                    fw.write(f"{anno_info['entity']}\t")
                    if anno_info["normalize_time"] != "":
                        fw.write(f"{anno_info['normalize_time']}\n")
                    else:
                        fw.write(f"{anno_info['entity']}\n")
                else:
                    fw.write(f"{anno_info['entity']}\n")