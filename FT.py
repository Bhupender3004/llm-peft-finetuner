import os
import re
import pandas as pd

from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer
from tqdm import tqdm
from huggingface_hub import notebook_login
from typing import List, Tuple, Union
from datasets import DatasetDict
from datasets import Dataset 

def get_qlora_configs(load_in_4bit: bool,
                      bnb_4bit_use_double_quant: bool,
                      bnb_4bit_quant_type: str,
                      bnb_4bit_compute_dtype: torch.dtype,
                      r: int,
                      lora_alpha: int,
                      target_modules: Union[List[str],str],
                      lora_dropout: float,
                      bias: str,
                      task_type: str) -> Tuple[BitsAndBytesConfig, LoraConfig]:
    """
    Create the configurations for use QLoRA thechniques

    Args:
        load_in_4bit (bool): This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        bnb_4bit_use_double_quant (bool): This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_type (str): This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_compute_dtype (torch.dtype): This sets the computational type which might be different than the input time. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        r (int): Lora attention dimension.
        lora_alpha (int): The alpha parameter for Lora scaling.
        target_modules (Union[List[str],str]): The names of the modules to apply Lora to.
        lora_dropout (float): The dropout probability for Lora layers.
        bias (str): Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        task_type (str): The task type for the model.

    Returns:
        Tuple[BitsAndBytesConfig, LoraConfig]: The configuration for BitsAndBytes and Lora.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    lora_config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    bnb_config, lora_config


def load_model_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer from the HuggingFace model hub using quantization.

    Args:
        model_name (str): The name of the model.
        bnb_config (BitsAndBytesConfig): The quantization configuration of BitsAndBytes.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer.
    """


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        trust_remote_code=True,
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,use_auth_token = True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False, mask_token="[MASK]")

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt(example: str) -> str:
    """"
    Format the prompt for the model.

    Args:
        example (str): The example.

    Returns:
        str: The formatted prompt.
    """

    final_text  = """### Instructions:
    Your task is to convert a question into a SQL query, given a Postgres database schema.
    Adhere to these rules:
    - **Deliberately go through the question and database schema word by word** to appropriately answer the question
    - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    - When creating a ratio, always cast the numerator as float

    ### Input:
    Generate a SQL query that answers the question `{question}`.
    This query will run on a database whose schema is represented in this string:
    {context}

    ### Response:
    {answer}
    ### End
    """.format(question = example['input'], context = example["instruction"], answer = example["output"])

    example["text"] = final_text

    return example


def get_max_length(model: str, max_length_default_value: int = 1024) -> int:
    """
    Get the maximum length of the model.

    Args:
        model (str): The model name.
        max_length_default_value (int): The default value for the maximum length.

    Returns:
        int: The maximum length of the model.
    """

    max_length = None

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            return max_length

    return max_length_default_value


def tokenize_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def formatting_prompts_func(examples):
    final_text  = """### Instructions:
    Your task is to convert a question into a SQL query, given a Postgres database schema.
    Adhere to these rules:
    - **Deliberately go through the question and database schema word by word** to appropriately answer the question
    - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    - When creating a ratio, always cast the numerator as float

    ### Input:
    Generate a SQL query that answers the question `{question}`.
    This query will run on a database whose schema is represented in this string:
    {context}

    ### Response:
    {answer}
    ### End
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = final_text.format(question=input, context=instruction, answer=output)
        # print(text)
        texts.append(text)
        # break
        
    return { "text" : texts}


def preprocess_dataset(tokenizer: AutoTokenizer,
                       max_length: int,
                       seed: int,
                       columns_to_remove: List[str],
                       dataset: DatasetDict) -> DatasetDict:
    """
    Preprocess the dataset for training.

    Args:
        tokenizer (AutoTokenizer): The tokenizer.
        max_length (int): The maximum length of the model.
        seed (int): The seed for shuffling the dataset.
        columns_to_remove (List[str]): The columns to remove from the dataset.
        dataset (DatasetDict): The Hugging face dataset.

    Returns:
        DatasetDict: The preprocessed dataset.
    """
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    _preprocessing_function = partial(tokenize_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = columns_to_remove,
    )

    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    dataset = dataset.shuffle(seed = seed)

    return dataset


def find_all_linear_names(model: AutoModelForCausalLM) -> list:
    """
    Find modules to apply LoRA to.

    Args:
        model (AutoModelForCausalLM): The model that will be fine-tuned.

    Returns:
        list: List with the modules names that we are going to apply LoRA
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_name = names[0] if len(names) == 1 else names[-1]
            if name != 'lm_head':
              lora_module_names.add(lora_module_name)

    print(f"LoRA module names: {list(lora_module_names)}")

    return list(lora_module_names)


def create_peft_config(r: int, lora_alpha: int, target_modules, lora_dropout: float, bias: str, task_type: str) -> LoraConfig:
    """
    Create the Parameter Efficient Fine-Tuning configuration.

    Args:
        r (int): Lora attention dimension.
        lora_alpha (int): The alpha parameter for Lora scaling.
        target_modules (_type_): _description_
        lora_dropout (float): The dropout probability for Lora layers.
        Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        task_type (str): The task type for the model.

    Returns:
        LoraConfig: _description_
    """
    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config


def preprare_model_for_fine_tune(model: AutoModelForCausalLM,
                                 lora_r: int,
                                 lora_alpha: int,
                                 lora_dropout: float,
                                 bias: str,
                                 task_type: str) -> AutoModelForCausalLM:
    """
    Prepares the model for fine-tuning.

    Args:
        model (AutoModelForCausalLM): The model that will be fine-tuned.
        lora_r (int): Lora attention dimension.
        lora_alpha (int): The alpha parameter for Lora scaling.
        lora_dropout (float): The dropout probability for Lora layers.
        Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        task_type (str): The task type for the model.

    Returns:
        AutoModelForCausalLM: The model prepared for fine-tuning.
    """
    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)

    model.config.use_cache = False

    return model


def free_memory(model: AutoModelForCausalLM, trainer: Trainer) -> None:
    """
    Free memory for merging weights

    Args:
        model (AutoModelForCausalLM): Pre-trained Hugging Face model
        trainer (Trainer): Trainer
    """
    pass

    del model
    del trainer
    torch.cuda.empty_cache()


def save_metrics(train_result, trainer: Trainer) -> None:
    """
    Save the metrics.
    """

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)


def save_model(model: AutoModelForCausalLM, output_dir: str) -> None:
    """
    Save the model.

    Args:
        model (AutoModelForCausalLM): The model.
        output_dir (str): The output directory.
    """

    os.makedirs(output_dir, exist_ok = True)
    model.save_pretrained(output_dir)
    print(f"Model saved in {output_dir}")


def fine_tune(model: AutoModelForCausalLM, trainer: Trainer, output_dir: str) -> None:
    """
    Fine-tune the model.

    Args:
        model (AutoModelForCausalLM): The model to fine-tune.
        trainer (Trainer): The trainer with the training configuration.
        output_dir (str): The output directory to save the model.
    """

    print("Training...")

    train_result = trainer.train()

    save_metrics(train_result, trainer)
    save_model(trainer.model, output_dir)
    # free_memory(model, trainer)

def print_trainable_parameters(model, use_4bit = False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

################################################################################
# transformers parameters
################################################################################

# The pre-trained model from the Hugging Face Hub to load and fine-tune (string)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading (bool)
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization) (bool)
bnb_4bit_use_double_quant = False

# Quantization type (fp4 or nf4) (string)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
compute_dtype = getattr(torch, "float16")
bnb_4bit_compute_dtype = compute_dtype

################################################################################
# QLoRA parameters
################################################################################

# Number of examples to train (int)
number_of_training_examples = 1000

# Number of examples to use to validate (int)
number_of_valid_examples = 200

# Dataset Name (string)
dataset_name = "b-mc2/sql-create-context"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension (int)
lora_r = 32

# Alpha parameter for LoRA scaling (int)
lora_alpha = 64

# Dropout probability for LoRA layers (float)
lora_dropout = 0.1

# Bias (string)
bias = "none"

# Task type (string)
task_type = "CAUSAL_LM"

# Random seed (int)
seed = 33

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored (string)
output_dir = "/data-disk/llm_models/FineTune_mistral_DW_217_KG_2000_v3/"

# Batch size per GPU for training (int)
per_device_train_batch_size = 4

# Number of update steps to accumulate the gradients for (int)
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer) (float)
learning_rate = 5e-5

# Optimizer to use (string)
optim = "paged_adamw_32bit"

# Number of training epochs (int)
num_train_epochs = 1

# Linear warmup steps from 0 to learning_rate (int)
warmup_steps = 10

# Enable fp16/bf16 training (set bf16 to True with an A100) (bool)
fp16 = True

# Log every X updates steps (int)
logging_steps = 100


bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

model, tokenizer = load_model_tokenizer(model_name, bnb_config)

################################################################################
# Preparing the DataSET For the Finetunning 
################################################################################
train_df = pd.read_csv("/home/bhupender.sharma/llm_finetunning/data/train/combined_train_2000_217_v2.csv")
train_dataset = Dataset.from_pandas(train_df)
print(f"The Shape of the TRAINING ------Data {train_df.shape}")

eval_df = pd.read_csv("/home/bhupender.sharma/llm_finetunning/data/test/combined_test_200_28_v2.csv")
eval_dataset = Dataset.from_pandas(eval_df)
print(f"The Shape of the TESTING------- Data {eval_df.shape}")

max_length = get_max_length(model)
columns_to_remove = ['output', 'input', 'instruction', 'Comment - ishpreet/vijay', 'query owner' ]
train_dataset = preprocess_dataset(tokenizer, max_length, seed, columns_to_remove, train_dataset)
eval_dataset = preprocess_dataset(tokenizer, max_length, seed, columns_to_remove, eval_dataset)


print_trainable_parameters(model)

model = preprare_model_for_fine_tune(model,
                                     lora_r,
                                     lora_alpha,
                                     lora_dropout,
                                     bias,
                                     task_type)

# Training parameters
trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset=eval_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        per_device_eval_batch_size= 4,
        gradient_accumulation_steps = 4,
        warmup_steps = warmup_steps,
        learning_rate = learning_rate,
        fp16 = fp16,
        logging_steps = 50,
        output_dir = output_dir,
        optim = optim,
        num_train_epochs=num_train_epochs,
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = 'True',
        group_by_length=True,
        split_batches=True,
    ),
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
)

fine_tune(model, trainer, output_dir)


################################################################################
# Saving The mdoel into The Directory 
################################################################################
# Load fine-tuned weights
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = "auto", torch_dtype = torch.bfloat16)
# Merge the LoRA layers with the base model
model = model.merge_and_unload()

# Save fine-tuned model at a new location
output_merged_dir = "/data-disk/llm_models/FineTune_mistral_DW_217_KG_2000_v3/saved_model"
os.makedirs(output_merged_dir, exist_ok = True)
model.save_pretrained(output_merged_dir, safe_serialization = False)

# Save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)