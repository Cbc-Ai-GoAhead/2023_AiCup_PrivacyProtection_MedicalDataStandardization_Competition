from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files="/content/drive/MyDrive/aicup/PublicDataset_phase3/opendid_valid.tsv", delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])
valid_list