# from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
pretrained_weights = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

# config = AutoConfig.from_pretrained(pretrained_weights, num_labels = labels_num)
# model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, config)
labels_num = 22
model = AutoModelForTokenClassification.from_pretrained(pretrained_weights, num_labels = labels_num)

# tokenizer = AutoTokenizer.from_pretrained('roberta-large', do_lower_case=True)

example = "This is a tokenization example"

encodings = tokenizer(example, padding=True, return_offsets_mapping="True")#, return_tensors="pt"
print(encodings)
# print(encodings["input_ids"].convert_tokens_to_string())
print(encodings["input_ids"])
print(tokenizer.decode(encodings["input_ids"]))
'''[Output]
[CLS] how's everything going? [SEP]
'''

# 純粹去對 vocab 做轉換
print("tpye = {}".format(type(encodings["input_ids"])))
print(tokenizer.convert_ids_to_tokens(encodings["input_ids"]))

result = tokenizer.convert_ids_to_tokens(encodings["input_ids"], skip_special_tokens=False)
print(result)
# desired_output = []
# for word_id in encoded.word_ids():
#     if word_id is not None:
#         start, end = encoded.word_to_tokens(word_id)
#         if start == end - 1:
#             tokens = [start]
#         else:
#             tokens = [start, end-1]
#         if len(desired_output) == 0 or desired_output[-1] != tokens:
#             desired_output.append(tokens)
# print(desired_output)