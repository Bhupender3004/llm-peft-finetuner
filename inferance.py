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

compute_dtype = getattr(torch, "float16")

def print_extracted_answer(raw_answer: str) -> None:
    """
    Print the extracted answer from the model.
    If the model does not extract the answer, print the raw_answer.

    Args:
        raw_answer (str): The raw answer from the model.
    """
    pattern = r'### Response:\s*([\S\s]*?)\s*### End:*'

    # Use re.search to find the match
    match = re.search(pattern, raw_answer)

    # Check if a match was found
    if match:
        # Extract the desired text (group 1 in the match object)
        extracted_text = match.group(1).strip()  # Remove leading/trailing white spaces
        print(f'Model Answer: {extracted_text}')
    else:
        print("No match found.")
        print(raw_answer)

    return extracted_text


def get_context_question_answer_from_index(valid_ds: DatasetDict, index: int) -> Tuple[str, str, str]:
    """
    Get the context, question, and answer from the dataset.

    Args:
        valid_ds (DatasetDict): The validation dataset.
        index (int): The index of the dataset.

    Returns:
        Tuple[str, str, str]: The context, question, and answer.
    """
    question = valid_ds[index]['input']
    context = valid_ds[index]['instruction']
    answer = valid_ds[index]['output']

    return context, question, answer


def print_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, context: str, answer: str = '') -> None:
    """
    Print the inference from the model.

    Args:
        model (AutoModelForCausalLM): Fine-tuned model.
        tokenizer (AutoTokenizer): Tokenizer.
        question (str): The natural language question.
        context (str): The database schema.
        answer (str): The query answer.
    """

    print(f'question: {question}')
    print(f'context: {context}')
    print(f'answer: {answer}\n')

    message = f'''
    ### Instructions:
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
    '''
    inputs = tokenizer(message, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=2000,do_sample=True,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
        pad_token_id=eval_tokenizer.eos_token_id
    ).to('cuda')
    output = tokenizer.batch_decode(outputs)[0]
    print(output)
    print_extracted_answer(output)

context = """CREATE TABLE transaction_1804 (
	date DATE, --Date when the transaction was made by the user/customer in the format YYYY-MM-DD
	order_id REAL, --Unique id for each transaction
  	user_id INTEGER, --Unique id of the user/customer, can be used as a primary key to join with other tables
	transaction_type TEXT, --Type of transaction made (e.g., buy/sell)
	status_of_transaction BOOLEAN, --Current status of the transaction, where TRUE represnts a successful transaction, FALSE represents a failed/unsettled/error tracnsaction
	transaction_error_type TEXT, --Type of error caused in the transaction (e.g., Err: Settlement Failure, Err: Insuffcient Funds)
  	average_price FLOAT, --Average price of the stock during the transaction
	total_order_amount FLOAT, --Total amount of the order
	quantity FLOAT, --Quantity of stock involved in the transaction
	revenue_fee FLOAT, --revenue generated with each transaction.
	good_faith_violation BIT, --Indicator for Good Faith Violation (GFV) violation, where TRUE represents a violation.
	pattern_day_trading_violation BIT, --Indicator for Pattern Day Trading (PDT) violation, where TRUE represents a violation.
	stock_industry TEXT, --Industry of the stock involved
	stock_company_name TEXT, --Name of the company associated with the stock
	product_type TEXT, --type of product for which the transaction is made e.g., equity
	buy_quantity REAL, --Quantity of stocks bought in the transaction
	sell_quantity REAL, --Quantity of stocks sold in the transaction
	subscription_plan TEXT, --Subscription plan of the user/customer
	customer_segment TEXT --Segment of the customer/user (e.g., High Growth diversified investor)
);
CREATE TABLE customer_profile_1804 (
	user_id TEXT PRIMARY_KEY, --Unique id of the user/customer
	user_name TEXT, --Username of the user/customer
	user_country TEXT, --Country of residence of the user/customer
	first_name TEXT, --First name of the user/customer
	last_name TEXT, --Last name of the user/customer
	account_opening_date DATE, --Date when the account was opened
	partner_id INTEGER, --Partner identifier associated with the user/customer, (e.g., 1, 2, etc)
	phone INTEGER, --Phone number of the user/customer
	account_management_type TEXT, --Type of account management (e.g., self, managed)
	account_trading_type TEXT, --Type of trading account (e.g., cash)
	user_city TEXT, --city of residence of the user/customer (e.g., new york), all the user_cities are in lower case
	user_state TEXT, --state of residence of the user/customer (e.g., new york, florida), the user_state is in lower case
	subscription_plan TEXT, --Subscription plan of the user/customer
	customer_segment TEXT --Segment of the customer/user (e.g., Low Value Infrequent Traders)

);
CREATE TABLE revenue_1804 (
  	user_id TEXT, --Unique id of the user/customer
	date DATE, --Date for which the revenue information is recorded
	revenue_last_week FLOAT, --Revenue generated by the user/customer in the last week
	revenue_last_month FLOAT, --Revenue generated by the user/customer in the last month
	revenue_last_quarter FLOAT, --Revenue generated by the user/customer in the last quarter
	revenue_last_year FLOAT, -- Revenue generated by the user/customer in the last year
	revenue_till_date FLOAT --Total revenue generated by the user/customer till the recorded date
);"""



bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

base_model_id = 'mistralai/Mistral-7B-Instruct-v0.2'

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token
ft_model = PeftModel.from_pretrained(base_model, "/data-disk/llm_models/FineTune_mistral_DW_217_KG_2000_v3",torch_dtype=torch.float16,is_trainable=False)

question = "How are energy stocks performing as compared to financials stocks for investors in our books?"

print_inference(model=ft_model, tokenizer=eval_tokenizer, context=context, question=question)

