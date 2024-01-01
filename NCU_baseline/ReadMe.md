0101
training
 00_medical_main.py  00_sliding_window_medical_main.py
inference
  05_testing_dataset_slidwindow.py
  11_time_regulation_group.py
11/23
新增time_regulation 對時間進行正規化
Bert 對 Duration和 Set沒有預測出來
value會有特殊字元 / ?

Bert 預測到下一列會多輸出換行
例子1
file4096    DEPARTMENT  575.0   583.0   PaLMS
33              
例子2
file2179    HOSPITAL    287.0   320.0   T
PAMBULA DISTRICT HOSPITAL HOSPI        
file21392   HOSPITAL    305 325 T
GREENWICH HOSPITAL
file13206   HOSPITAL    296 319 T
BOOLEROO CAMPUS HOSPI     
換行要處理掉 目前先用 手動修改

模型使用 save_dict沒辦法讀取
只能使用torch.save
但是會每次都預測不一樣
Hugging face 要使用 save_pretrianed?
11/26
新增第6個欄位
需要對預測的值 進行 offsetmapping

安裝lora遇到問題pip install peft
ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.

We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.

torchvision 0.11.0+cu111 requires torch==1.10.0+cu111, but you'll have torch 2.1.1 which is incompatible.
torchaudio 0.10.0+cu111 requires torch==1.10.0, but you'll have torch 2.1.1 which is incompatible.

trainer.save_model("path/to/model")
AutoModelForSequenceClassification.from_pretrained("path/to/model")
https://stackoverflow.com/questions/72108945/saving-finetuned-model-locally


改成用 if best 就作evaluation

docker run -it --gpus '"device=0"' --shm-size=64G --net=host  -v /home/cbc/2023_projects/NCU_NLP/NLP-1:/NLP   -v /etc/localtime:/etc/localtime:ro  --name aicup_hw2 sed_core:latest /bin/bash
pip install peft
pip show typing-extensions
pip install typing-extensions==4.3.0
------
model.save_pretrained("./model/bert_save_testing")
ImportError: cannot import name 'soft_unicode' from 'markupsafe' (/opt/conda/lib/python3.8/site-packages/markupsafe/__init__.py)
pip install markupsafe==2.0.1
------
pytorch >1.10.1 ,GPU driver be too old
lora's pytorch version is> 1.2.0
