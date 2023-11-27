# 2023_AiCup_PrivacyProtection_MedicalDataStandardization_Competition
模型位置~/.cache/huggingface/hub

上傳檔案的格式必須為純文字檔案，並以 Tab 鍵將上述欄位進行分隔，並且以 UTF-8 編碼方式存檔，存檔檔名必須是 answer.txt。 最後壓縮成 ZIP 壓縮檔後透過本頁頁籤 Submit / View Results 中的上傳介面上傳。下面提供一個上傳的壓縮檔案內容範例：


docker run -it --gpus "device=0:2" --shm-size=64G --net=host  -v /home/cbc/2023_projects/NCU_NLP/NLP-1:/NLP   -v /etc/localtime:/etc/localtime:ro  --name nlp_hw1 sed_core:latest /bin/bash
docker run -it --gpus '"device=1"' --shm-size=64G --net=host  -v /home/cbc/2023_projects/NCU_NLP/NLP-1:/NLP   -v /etc/localtime:/etc/localtime:ro  --name aicup_hw2 sed_core:latest /bin/bash

11/25
輸出的answer.txt 必須要有5個欄位, 後放多tab , 線上系統不會計算到
1037    IDNUM   13  23  44N0614707  
如果是時間正規化必須要有六個欄位

DATE    145 154 14/5/1998   1998-05-14