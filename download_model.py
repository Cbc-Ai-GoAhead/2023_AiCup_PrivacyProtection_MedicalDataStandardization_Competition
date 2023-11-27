from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
plm = "EleutherAI/pythia-70m" ## 變換 model 的地方
tokenizer = AutoTokenizer.from_pretrained(plm)

# tokenizer.add_special_tokens(special_tokens_dict)
# PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)