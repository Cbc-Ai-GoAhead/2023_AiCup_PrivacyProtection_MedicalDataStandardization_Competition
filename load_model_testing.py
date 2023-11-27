from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
####
##  Special Token定義
####
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}

plm = "EleutherAI/pythia-70m" ## 變換 model 的地方
tokenizer = AutoTokenizer.from_pretrained(plm)

model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))