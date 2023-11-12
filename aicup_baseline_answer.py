# https://colab.research.google.com/drive/1IxOdC7jROq18OfEiDRMttLV4pk62UOV6#scrollTo=0DMccN-wNdlQ
# from datasets import load_dataset, Features, Value
# !pip install datasets
# 是在使用 驗證集做調整嗎?

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
# 這次先不做驗證集 訓練
# dataset = load_dataset("csv", data_files="data/opendid_set1.tsv", delimiter='\t',
#                        features = Features({
#                               'fid': Value('string'), 'idx': Value('int64'),
#                               'content': Value('string'), 'label': Value('string')}),
#                               column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
# print("#### data_information")
# print(dataset['train'][0])
# print(dataset['train'][1])
# print(dataset['train'][7])


from tqdm.notebook import tqdm
from islab.aicup import aicup_predict
# !pip install islab-opendeid
import io
BATCH_SIZE = 32
from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files="data/opendid_valid.tsv", delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])
valid_list

# from tqdm.notebook import tqdm
# from islab.aicup import aicup_predict
# import io
# BATCH_SIZE = 32
#
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}

plm = "EleutherAI/pythia-70m" ## 變換 model 的地方
tokenizer = AutoTokenizer.from_pretrained(plm)

tokenizer.add_special_tokens(special_tokens_dict)
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# tr_dataset = GPTDataset(train_seq_pairs,
#                         tokenizer,
#                         special_tokens_dict,
#                         PAD_IDX)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))

model.load_state_dict(torch.load("output/models/1016_5e_5/GPT_best.pt"))
# model.eval()
# model.state_dict()
model = model.to(device)

with open("./answer.txt",'w',encoding='utf8') as f:
#with io.open("answer.txt",'w',encoding='utf8') as f:
    # for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
    #     with torch.no_grad():
    #         seeds = valid_list[i:i+BATCH_SIZE]
    #         outputs = aicup_predict(model, tokenizer, input=seeds)
    #         for o in outputs:
    #             f.write(o)
    #             f.write('\n')

    for i in range(0, len(valid_list), BATCH_SIZE):
        with torch.no_grad():
            seeds = valid_list[i:i+BATCH_SIZE]
            outputs = aicup_predict(model, tokenizer, input=seeds)
            for o in outputs:
                f.write(o)
                f.write('\n')