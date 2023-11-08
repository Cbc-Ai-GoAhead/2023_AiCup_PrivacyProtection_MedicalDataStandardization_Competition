import os
import random
import numpy as np
import torch
import multiprocessing
# Tag
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}
#process
def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

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
