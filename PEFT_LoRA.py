from transformers import AutoTokenizer

model_id = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

'''
loaded a model distillBert : best for text embedding and classification tasks
you can change the model to say gpt 2 for text generation tasks
you can also change the model to any other model from huggingface hub
'''
from datasets import load_dataset

dataset = load_dataset("ag_news", split="train[:5%]") 
print(type(dataset))

# def tokenize_fn(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# tokenized = dataset.map(tokenize_fn, batched=True)
