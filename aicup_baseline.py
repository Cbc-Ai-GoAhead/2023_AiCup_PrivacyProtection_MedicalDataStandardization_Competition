#Merge Task https://colab.research.google.com/drive/1f0460USzaUbpxwNjLJtYVKm9USP0Z_EK#scrollTo=hUSHfpdtx4a5
#Phase3 https://colab.research.google.com/drive/1IxOdC7jROq18OfEiDRMttLV4pk62UOV6#scrollTo=65hmv62Pw9Y7

import os

import numpy as np
from tqdm import tqdm, trange
from torch.optim import AdamW

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset
import multiprocessing

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_torch_seed()

def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}

def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案
    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)

    bounary , item_idx , temp_seq , seq_pairs = 0 , 0 , "" , []

    for w_idx, word in enumerate(article):
        if word == '\n':
            new_line_idx = w_idx + 1
            if temp_seq == "":
                temp_seq = "PHI:Null"

            seq_pair = special_tokens_dict['bos_token'] + article[bounary:new_line_idx] + special_tokens_dict['sep_token'] + temp_seq + special_tokens_dict['eos_token']
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\n"
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    return seq_pairs

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder, num_processes=4):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())

    pool = multiprocessing.Pool(num_processes)
    print("processing each medical file")
    results = pool.starmap(process_medical_report, [(txt_name, medical_report_folder, annos_dict, special_tokens_dict) for txt_name in txt_names])

    seq_pairs = [pair for result in results for pair in result]

    pool.close()
    pool.join()
    print("All medical file done")
    return seq_pairs

anno_info_path = "data/First_Phase_Release_Correction/answer.txt"
report_folder = "data/First_Phase_Release_Correction/First_Phase_Text_Dataset"
train_seq_pairs = generate_annotated_medical_report_parallel(anno_info_path, report_folder, num_processes=4)

idx = 10

print(f"input : \n{train_seq_pairs[idx]}")

class GPTDataset(Dataset):
    '''
    繼承torch.Dataset
    '''
    def __init__(self,seq_paris, tokenizer , special_tokens_dict , pad_idx , mode = 'train'):
        self.seq_paris = seq_paris
        self.tokenizer = tokenizer
        self.special_tokens_dict = special_tokens_dict
        self.pad_idx = pad_idx
        self.mode = mode

    def __len__(self):
        return len(self.seq_paris)

    def __getitem__(self, index):
        return self.seq_paris[index]

    def pad_sequence(self , non_pad_token , non_pad_label , non_pad_attn):
        '''
        input : token ids, labels, attention masks
        將每個向量 padding 之後組成矩陣
        output : pad token ids, pad labels, pad attention masks
        '''
        max_size = max([len(ele) for ele in non_pad_token])
        pad_batch1 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_token])
        pad_batch2 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_label])
        pad_batch3 = torch.stack([torch.cat([t, torch.LongTensor([0] * (max_size - len(t)))]) for t in non_pad_attn])
        return pad_batch1 , pad_batch2 , pad_batch3

    def collate_batch(self , datasets):
        '''
        input : token ids
        Dataloader 呼叫時的函式
        回傳批次的torch data
        output : pad token ids, pad labels, pad attention masks
        '''
        tokens_list , labels_list , attention_mask_list = [] , [] , []
        for dataset in datasets:
            tokenizer.padding_side = 'left'
            encoded_seq = tokenizer(dataset, padding=True)
            indexed_tks = encoded_seq["input_ids"]
            attention_mask = encoded_seq["attention_mask"]

            tokens_list.append(torch.tensor(indexed_tks))
            labels_list.append(torch.tensor(indexed_tks))
            attention_mask_list.append(torch.tensor(attention_mask))
        return self.pad_sequence(tokens_list , labels_list , attention_mask_list)

BATCH_SIZE = 12

plm = "EleutherAI/pythia-70m" ## 變換 model 的地方
tokenizer = AutoTokenizer.from_pretrained(plm)

tokenizer.add_special_tokens(special_tokens_dict)
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tr_dataset = GPTDataset(train_seq_pairs,
                        tokenizer,
                        special_tokens_dict,
                        PAD_IDX)

bucket_train_dataloader = DataLoader(tr_dataset,
                                    batch_size=BATCH_SIZE,
                                    collate_fn=tr_dataset.collate_batch)

model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))

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

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01}
]
# 優化器
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=1e-4  #學習率
)
# 迭代次數
epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型儲存資料夾名稱
model_name = "1110_50e_5"
# 模型儲存路徑
model_dir = f"output/models/{model_name}"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model = model.to(device)
min_loss = 9999

predict_text = special_tokens_dict['bos_token'] + "MANILDRA  NSW  2865"

# 模型訓練開始
for _ in range(epochs):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

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
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))
    print(sample_text(model, tokenizer, text=predict_text))
    torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_Finial.pt'))
    if avg_train_loss < min_loss:
        min_loss = avg_train_loss
        torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_best.pt'))

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
test_phase_path = r'data/valid_dataset/Validation_Release'
test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))

write_file = "/output/submission.txt"
print(test_txts)
predict_file(test_txts , write_file)