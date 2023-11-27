from bert_model import myModel
import torch
model_PATH = "./model/bert-base-cased_9_12_0.5983084987356402"


device = 'cpu'
# vgg = models.vgg16().to(device)
fine_tune_model = torch.load(model_PATH, map_location=torch.device('cpu'))
# fine_tune_model = model.to(device)
# for name, param in fine_tune_model.named_parameters():
#     if param.requires_grad:
#         print( name, param.data)
print("---fine tune model")
print(fine_tune_model)
for name, param in fine_tune_model.named_parameters():
    if param.requires_grad:
        print(name)
    #fc.weight
    #fc.bias
    if name == "fc.weight":
        print(param.data)
    if name == "fc.bias":
        print(param.data)
# print(fine_tune_model.layername.weight.shape)
# for its in fine_tune_model.items():
#     print(its)
# for param in fine_tune_model.parameters():
#     print(param.data)
    
# print(fine_tune_model.keys())

print("-------Load pretrained Bert model")
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
model_name = "bert-base-cased"
labels_num = 22
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_auto = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = labels_num)
print("---tokenizer model")
print(tokenizer)

print("---model_auto")
print(model_auto)
for name, param in model_auto.named_parameters():
    if param.requires_grad:
        print(name)
    #classifier.weight
    #classifier.bias
    if name == "classifier.weight":
        print(param.data)
    if name == "classifier.bias":
        print(param.data)
    
# print(model_auto.layername.weight.shape)
# for param in model_auto.parameters():
#     print(param.data)
# print(model_auto.keys())