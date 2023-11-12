import os

import numpy as np

from torch.optim import AdamW

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup


import matplotlib.pyplot as plt
from torch.nn import functional as F


# self define module
from util import *
from dataset import GPTDataset
from train_model import *
# import logger
# Tag
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}
"""
module_name = "Not Implement"
procedure_name = "ai_main"

time_str = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
logObj = log_class.Log('./logs', module_name)
logObj.set_name( module_name + "[" + procedure_name + "]"  )
experiment_name = "Training Model"
logObj.info('Experiments: ' + experiment_name)
"""
if __name__ == '__main__':
    set_torch_seed()
    ####
    ##  Input Data
    ####
    logObj.info('input DataSet')
    print("### input DataSet")
    anno_info_path = "data/First_Phase_ReleaseCorrection/answer.txt"
    report_folder = "data/First_Phase_ReleaseCorrection/First_Phase_Text_Dataset"
    train_seq_pairs = generate_annotated_medical_report_parallel(anno_info_path, report_folder, num_processes=4)
    idx = 100

    print(f"input : \n{train_seq_pairs[idx]}")

    ####
    ##  Build Model
    ####
    # logging.info("Model & Dataloader 宣告")
    print("### Model & Dataloader 宣告")
    BATCH_SIZE = 12
    
    # down load model
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

    print("### Hyperparameters")

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
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("### Start Training")
    # 模型儲存資料夾名稱
    model_name = "exp/{}" .format("1016_5e_5")

    #train(optimizer, model, bucket_train_dataloader, dev_dataloader, train_ix_to_tag, dev_ix_to_tag,num_epochs=10000):
    train_model(optimizer, model, bucket_train_dataloader, device, model_name, num_epochs=epochs)

    test_phase_path = r'data/First_Phase_ReleaseCorrection/Validation_Release'
    test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))

    #write_file = "/content/submission.txt"
    write_file = "./submission.txt"
    print(test_txts)
    # 會再呼叫sample_text
    predict_file(test_txts , write_file)


