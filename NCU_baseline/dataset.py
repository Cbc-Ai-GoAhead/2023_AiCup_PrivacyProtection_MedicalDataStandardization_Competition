import torch
from torch.utils.data import Dataset, DataLoader
class Privacy_protection_dataset(Dataset):
  def __init__(self, medical_record_dict:dict, medical_record_labels:dict, tokenizer, labels_type_table:dict, mode:str):
      self.labels_type_table = labels_type_table
      self.tokenizer = tokenizer
      if mode == "train" or mode == "validation":
        self.id_list = list(medical_record_dict.keys())
        self.x = list(medical_record_dict.values())
        self.y = [[labels[:3] for labels in medical_record_labels[sample_id]] for sample_id in self.id_list]

  def __getitem__(self, index):
      return self.x[index], self.y[index], self.id_list[index]

  def __len__(self):
      return len(self.x)

  def find_token_ids(self, label_start, label_end, offset_mapping):
    """find the correct labels ids after tokenizer"""
    encodeing_start = float("inf") #max
    encodeing_end = 0
    for token_id, token_range in enumerate(offset_mapping):
      token_start, token_end = token_range
      #if token range one side out of label range, still take the token
      if token_start == 0 and token_end == 0: #special tocken
        continue
      if label_start<token_end and label_end>token_start:
        if token_id<encodeing_start:
          encodeing_start = token_id
        encodeing_end = token_id+1
    return encodeing_start, encodeing_end

  def encode_labels_position(self, batch_lables:list, offset_mapping:list):
    """ encode the batch_lables's position"""
    batch_encodeing_labels = []
    for sample_labels, sample_offsets in zip(batch_lables, offset_mapping):
      encodeing_labels = []
      for label in sample_labels:
        encodeing_start, encodeing_end = self.find_token_ids(label[1], label[2], sample_offsets)
        encodeing_labels.append([label[0], encodeing_start, encodeing_end])
      batch_encodeing_labels.append(encodeing_labels)
    return batch_encodeing_labels

  def create_labels_tensor(self, batch_shape:list, batch_labels_position_encoded:list):
    if batch_shape[-1]> 4096:
      batch_shape[-1] = 4096
    labels_tensor = torch.zeros(batch_shape)

    for sample_id in range(batch_shape[0]):
      for label in batch_labels_position_encoded[sample_id]:
        label_id = self.labels_type_table[label[0]]
        start = label[1]
        end = label[2]
        if start >= 4096: continue
        elif end >= 4096: end = 4096
        labels_tensor[sample_id][start:end] = label_id
    return labels_tensor

  def collate_fn(self, batch_items:list):
    """the calculation process in dataloader iteration"""
    batch_medical_record = [sample[0] for sample in batch_items]
    batch_labels = [sample[1] for sample in batch_items]
    batch_id_list = [sample[2] for sample in batch_items]
    encodings = self.tokenizer(batch_medical_record, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping="True") # truncation=True

    batch_labels_position_encoded = self.encode_labels_position(batch_labels, encodings["offset_mapping"])
    #print(encodings["offset_mapping"])
    #print(batch_labels_position_encoded) #show the labels after position encoding
    batch_labels_tensor = self.create_labels_tensor(encodings["input_ids"].shape, batch_labels_position_encoded)
    return encodings, batch_labels_tensor, batch_labels