print("#### Load pre trained!!")
#Loading time depend on your network speed
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
# from peft import LoraConfig, TaskType

# from peft import LoraConfig, get_peft_model, TaskType

class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()
        self.num_labels = 22
        self.bert = AutoModel.from_pretrained('bert-base-cased', cache_dir="./checkpoint/")
        self.droupout = nn.Dropout(p=0.1, inplace=False)
        self.fc = nn.Linear(768, 22)

        # self.init_weights()
    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.shape)
        output = self.droupout(output.last_hidden_state)
        out = self.fc(output)

        return out
class myLongModel(torch.nn.Module):

    def __init__(self):

        super(myLongModel, self).__init__()
        self.num_labels = 22
        self.bert = AutoModel.from_pretrained('allenai/longformer-base-4096', cache_dir="./checkpoint/")
        self.droupout = nn.Dropout(p=0.1, inplace=False)
        self.fc = nn.Linear(768, 22)

        # self.init_weights()
    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.shape)
        output = self.droupout(output.last_hidden_state)
        out = self.fc(output)

        return out
if __name__ == '__main__':
    # model_name = "bert-base-cased"
    labels_num = 22
    #Method 1 load pretrained weight
    pretrained_weights = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, cache_dir="./checkpoint/")
    # config 用來修改模型參數的
    """
    config = AutoConfig.from_pretrained(pretrained_weights, num_labels = labels_num)
    print(config)
    """
    model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, cache_dir="./checkpoint/",num_labels = labels_num)
    print(model)
    #model.save_pretrained("./model/bert_save_testing")




    # token完
    # bert 預訓練模型
    # Method 2 Lora
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, num_labels = labels_num)
    peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
    )#target_modules=["key", "query", "value"]

    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    model.save_pretrained("./model/bert_save_testing")
    """
    # print(type(tokenizer))
    # print(type(model))
    # print(model)

    #才丟入自定義的模型
    print("Self define Model")
    # LORA_R = 16  # 設定LORA（Layer-wise Random Attention）的R值 Set LORA R value
    # LORA_ALPHA = 16  # 設定LORA的Alpha值 Set LORA Alpha value
    # LORA_DROPOUT = 0.05  # 設定LORA的Dropout率 Set LORA dropout value
   
    # lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1)
    # model = myModel()
    # model = get_peft_model(model, lora_config)

    
    print(model)