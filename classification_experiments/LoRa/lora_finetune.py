
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from unsloth import FastLanguageModel
import torch
from peft import PeftModel
import os
import pandas as pd

# os.chdir('..')
# os.chdir('..')

TRAIN_DF = pd.read_csv("Data/TRAIN_DF.csv", index_col=0)
TRAIN_DF.rename(columns={'Response': 'Explanations'}, inplace=True)

# TEST_DF = pd.read_csv("Data/TEST_DF.csv", index_col=0)
# Example: Assuming df is your pandas DataFrame
# df = pd.DataFrame(...)

# Convert pandas DataFrame to Hugging Face Dataset


if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available")


model_parameters = {
   'model_name' : 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
   'model_dtype' : None ,
   'model_load_in_4bit' : True,
   'model_max_seq_length':2048
}

# model, tokenizer = FastLanguageModel.from_pretrained("llama_2_13b_lora_18k")

# model.save_pretrained_merged("full_model", tokenizer, save_method = "merged_16bit",)
model, tokenizer = FastLanguageModel.from_pretrained(
   model_name = model_parameters['model_name'], #"llama_2_13b_lora",
   # model_name = model_parameters['model_name'],
   max_seq_length = model_parameters['model_max_seq_length'],
   dtype = model_parameters['model_dtype'],
   load_in_4bit = model_parameters['model_load_in_4bit'],
   # device="cuda:1"
   # cache_folder='/media/data/hugging_face_cache'
)
# model = PeftModel.from_pretrained(model, "llama_2_13b_lora_18k")
# model.save_pretrained_merged("full_model", tokenizer, save_method = "merged_16bit",)
# print("model saved")
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     AutoTokenizer,
#     TrainingArguments,
# )

# model_name = "unsloth/llama-2-13b-bnb-4bit"
# import torch

# available_devices = list(range(1, 6))  # CUDA:1 to CUDA:5
# device_map = {f"model.{i}": device for i, device in enumerate(available_devices)}
# print(device_map)
# model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         load_in_4bit=True,
#         device_map="auto",
#         # use_auth_token=True,
#         # cache_folder='/media/data/hugging_face_cache'
#     )
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


lora_parameters = {
   'lora_r': 16,
   'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
   'lora_alpha': 16,
   'lora_dropout': 0,
   'lora_bias': "none",
   'lora_use_gradient_checkpointing': "unsloth",
   'lora_random_state': 42,
}


model = FastLanguageModel.get_peft_model(
   model,
   r = lora_parameters['lora_r'],
   target_modules = lora_parameters['target_modules'],
   lora_alpha = lora_parameters['lora_alpha'],
   lora_dropout = lora_parameters['lora_dropout'],
   bias = lora_parameters['lora_bias'],
   use_gradient_checkpointing =    lora_parameters['lora_use_gradient_checkpointing'],
   random_state = lora_parameters['lora_random_state'],
)



prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Lets classify the following text for hate speech based on 'Human' text

Instruction:
You are given Userinput as Human, and your job is to classify it as hatespeech, offensive, or normal based. You are given Explanation to why its classified as.

if its hatespeech your reply will be: Assistant: "hatespeech"
if its not hatespeech your reply will be : Assistant: "normal"
if its offensive and not hatespeech, your reply will be : Assistant: "offensive"
Here is the query

Human: {tweet_text}
Explanation: {Explanations}
Assistant: {label}
Key_Features: {key_features}"""

# TRAIN_DF = pd.read_csv("TRAIN_DF.csv", index_col=0)
# Assuming TRAIN_DF is already loaded as a DataFrame
TRAIN_DF['instruction_text'] = TRAIN_DF.apply(
    lambda row: (
        f"""{prompt.format(
            tweet_text=row['tweet_text'], 
            Explanations=row['Explanations'], 
            label=row['label'], 
            key_features=row['key_features']
        )}"""
    ),
    axis=1
)

dataset = Dataset.from_pandas(TRAIN_DF)
dataset = dataset.train_test_split(test_size=0.2)



training_arguments = {
   # Tracking parameters
   'eval_strategy' : "steps",
   'eval_steps': 100,
   'logging_strategy' : "steps",
   'logging_steps': 5,
   'save_strategy' : "epoch",

   # Training parameters
   'per_device_train_batch_size' : 40,
   'num_train_epochs' : 3,
   'optim' : "adamw_8bit",
   'fp16' : not is_bfloat16_supported(),
   'bf16' : is_bfloat16_supported(),
   'warmup_steps' : 100,
   'learning_rate' : 2e-4,
   'lr_scheduler_type' : "cosine",
   'weight_decay' : 0.01,

   'seed' : 3407,
   'output_dir' : "outputs",

}


trainer = SFTTrainer(
   model = model,
   tokenizer = tokenizer,
   train_dataset = dataset['train'],
   eval_dataset = dataset['test'],
   dataset_text_field = "instruction_text",
   max_seq_length = model_parameters['model_max_seq_length'],
   dataset_num_proc = 2,
   packing = False,
   args = TrainingArguments(
	**training_arguments
   ),
)

trainer.model.print_trainable_parameters()
trainer.train()


new_model_local = "llama_2_13b_lora_18k2"
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)