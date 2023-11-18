print("#### Load pre trained!!")
#Loading time depend on your network speed
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.droupout = nn.Dropout(p=0.1, inplace=False)
        self.fc = nn.Linear(768, 22)


    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.shape)
        output = self.droupout(output.last_hidden_state)
        out = self.fc(output)

        return out

if __name__ == '__main__':
    model_name = "bert-base-cased"
    labels_num = 22
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = labels_num)

    print(type(tokenizer))
    print(type(model))
    print(model)


    print("Self define Model")
    model = myModel()
    print(model)