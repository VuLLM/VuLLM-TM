from arrow import get
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, AutoTokenizer, 
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments)
from peft import PeftModel, PeftConfig
import pandas as pd
import torch
from datasets import Dataset
from pynvml import *
sys.path.append('/sise/home/urizlo/VuLLM')
# from Injection_model import Prepare_dataset_with_unique_lines
from utils import Custom_trainer
from huggingface_hub import login
from tqdm import tqdm
from collections import defaultdict
import json
from dotenv import load_dotenv
import argparse


def process_c_functions(df, column_name):
    """
    Process C functions in a DataFrame column by splitting them into lines and
    handling special cases where lines contain only "{" or "}" characters.

    Args:
        df (pandas.DataFrame): The DataFrame containing the C functions.
        column_name (str): The name of the column containing the C functions.

    Returns:
        pandas.DataFrame: The DataFrame with the processed C functions.
    """
    for index, row in df.iterrows():
        lines = row[column_name].split('\n')  # Split the function into lines
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ["{", "}"] and i > 0:  # Check if line contains only "{" or "}"
                # Append the character to the end of the previous line
                processed_lines[-1] = processed_lines[-1] + " " + line
            else:
                processed_lines.append(lines[i])  # Add the line as it is
            i += 1
        # Update the DataFrame with the processed function
        df.at[index, column_name] = '\n'.join(processed_lines)
    return df



def add_locaitions(path_local, df, col):
    df_local = pd.read_csv(path_local)
    df_local = df_local.fillna("")
    df[col] = df[col] + "<|endoftext|>" + "\n" + df_local[f'local_res']
    return df



def append_spaces_suffix_to_duplicates(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Appends spaces as suffix to duplicate lines in the specified column of the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column containing the function code.

    Returns:
        pd.DataFrame: The modified DataFrame with spaces appended to duplicate lines.
    """
    modified_data = data.copy()
    for i, row in modified_data.iterrows():
        function_code = row[column]
        lines = function_code.split('\n')
        modified_lines = []
        line_counts = defaultdict(int)  # Start all values equal to 1
        for line in lines:
            if line_counts[line.strip()] > 0:
                spaces = " " * line_counts[line.strip()]
                modified_lines.append(f"{line}{spaces}")
            else:
                modified_lines.append(line)
            line_counts[line.strip()] += 1
        modified_data.at[i, column] = '\n'.join(modified_lines)
    return modified_data




def prepare_data(path_nonvul, path_local, tokenizer):
    # data = []
    # with open(path_nonvul, "r") as file:
    #     for line in file:
    #         json_obj = json.loads(line)
    #         data.append(json_obj)
    df = pd.read_csv(path_nonvul)
    df = df.fillna("")
    df = process_c_functions(df, 'nonvul')
    df = append_spaces_suffix_to_duplicates(df, 'nonvul')
    df = add_locaitions(path_local, df, 'nonvul')
    tokenized_func1 = tokenizer(df['nonvul'].tolist(), return_tensors='pt', padding='max_length', truncation=True)
    tokenized_func1 = Dataset.from_dict(tokenized_func1)
    return tokenized_func1



def get_model_tokenizer(injection_model_id):
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    # config = PeftConfig.from_pretrained(injection_model_id)
    localization_model = AutoModelForSeq2SeqLM.from_pretrained(injection_model_id, 
                                                return_dict=True,
                                                torch_dtype=torch.float16, 
                                                device_map='auto',
                                                trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(injection_model_id)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # Load the Lora model
    # localization_model = PeftModel.from_pretrained(localization_model, injection_model_id).to(device)

    localization_model.config.max_length=359
    localization_model.config.use_cache=False
    localization_model.config.decoder_start_token_id = tokenizer.bos_token_id  # Replace tokenizer.cls_token_id with the appropriate token ID
    localization_model.config.pad_token_id = 50256
    localization_model.config.decoder.pad_token_id = localization_model.config.decoder.eos_token_id
    localization_model.config.encoder.pad_token_id = localization_model.config.encoder.eos_token_id
    localization_model.config.encoder.max_length = 359
    localization_model.config.decoder.max_length = 359
    return localization_model, tokenizer



def get_trainer(model, tokenizer, tokenized_local_test):
    checkpoint = "Salesforce/codet5p-6b"
    training_args = Seq2SeqTrainingArguments(
    output_dir="saved_models/injection_model",
    per_device_eval_batch_size=1,
    tf32=True,
    remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    trainer = Custom_trainer.CodeT5pTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_local_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer



def get_prediction(trainer, tokenizer):
    local_res = []
    # Get the total number of batches for the progress bar
    total_batches = len(trainer.get_eval_dataloader())
    # Use tqdm to create a progress bar
    for b in tqdm(trainer.get_eval_dataloader(), total=total_batches, desc="Evaluating"):
        out = trainer.model.generate(**b, max_length=359)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        local_res.append(text)
    return local_res


def main(path_testset, model_huggingface_path, path_res_local, output_dir):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    load_dotenv()
    # TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    # login(token=TOKEN)
    model, tokenizer = get_model_tokenizer(model_huggingface_path)
    tokenized_func1 = prepare_data(path_testset, path_res_local, tokenizer)
    trainer = get_trainer(model, tokenizer, tokenized_func1)
    local_res = get_prediction(trainer, tokenizer)
    df = pd.DataFrame(local_res, columns=['func_inj'])
    df.to_csv(output_dir, index=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_testset', type=str, default='saved_models', help='Path to test set csv file.')
    parser.add_argument('--model_huggingface_path', type=str, default='urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate', help='Path to load lora adapters form huggingface')
    parser.add_argument('--path_res_local', type=str, default='connected_models/localization_results/vulgen_res_loc.csv', help='Path to localization results of the testset')
    parser.add_argument('--output_dir', type=str, default=False, help='Path where to save the loclization csv file')
    args = parser.parse_args()
    main(args.path_testset, args.model_huggingface_path, args.path_res_local, args.output_dir)