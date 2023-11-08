# 2023_AiCup_PrivacyProtection_MedicalDataStandardization_Competition
20231105
aicup_sample_code.py can run batch_size =4, batch_size >4 out of cuda memory
RuntimeError: CUDA out of memory. Tried to allocate 1.20 GiB (GPU 0; 10.76 GiB total capacity; 8.43 GiB already allocated; 1.03 GiB free; 8.76 GiB reserved in total by PyTorch)

split aicup_sample_code.py to different module
Initial Repository
	write module

counter problems
1. cant log information
2. Traceback (most recent call last):
  File "ai_main.py", line 89, in <module>
    train_model(optimizer, model, bucket_train_dataloader, device, model_name, num_epochs=epochs)
  File "/dataset/NLP/aicup/train_model.py", line 59, in train_model
    total_loss = train_epoch_model(optimizer, model, bucket_train_dataloader, device)#train_epoch(train_loader,model, args.lr,optimizer, device)
  File "/dataset/NLP/aicup/train_model.py", line 18, in train_epoch_model
    for step, (seqs, labels, masks) in enumerate(tqdm(bucket_train_dataloader)):
  File "/opt/conda/lib/python3.8/site-packages/tqdm/std.py", line 1171, in __iter__
    for obj in iterable:
TypeError: 'torch.device' object is not iterable


