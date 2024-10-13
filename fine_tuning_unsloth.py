from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

import subprocess
from torch import __version__ as torch_version
from packaging.version import Version as V

def install_packages():
    # Determine the version of xformers to install based on torch version
    xformers = "xformers==0.0.27" if V(torch_version) < V("2.4.0") else "xformers"
    
    # Packages to install
    packages = [xformers, "trl", "peft", "accelerate", "bitsandbytes", "triton"]
    
    # Create the pip install command
    command = ["pip", "install", "--no-deps"] + packages
    
    # Run the pip install command
    subprocess.check_call(command)

if __name__ == "__main__":
    install_packages()


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "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"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0, #To be as fast as possible
    target_modules =['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj', "o_proj", "gate_proj"],
    use_rslora=True, # vizualisation ? r/alpha ?
    use_gradient_checkpointing="unsloth", # faster checkpointing,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral",
)

dataset = load_dataset('json', data_files='formatted_conversations.jsonl') 

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset_train = dataset['train'].map(formatting_prompts_func, batched = True,)

print("PANDAS DF :", dataset_train.to_pandas())

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    dataset_text_field = 'text',
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()