from arrow import get
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, AutoTokenizer, 
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments)
from peft import PeftModel, PeftConfig
import pandas as pd
import torch
from datasets import Dataset
from pynvml import *
sys.path.append('/sise/home/urizlo/VuLLM')
from localization_model.CodeT5p import Prepare_data_original
from utils import Custom_trainer
from huggingface_hub import login
from tqdm import tqdm
from collections import defaultdict
import pickle
import hashlib
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



def get_model_tokenizer(localization_model_id):
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    # config = PeftConfig.from_pretrained(localization_model_id)
    localization_model = AutoModelForSeq2SeqLM.from_pretrained(localization_model_id, 
                                                return_dict=True,
                                                torch_dtype=torch.float16, 
                                                trust_remote_code=True,
                                                device_map='auto'
                                                )
    tokenizer = AutoTokenizer.from_pretrained(localization_model_id)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # Load the Lora model
    # localization_model = PeftModel.from_pretrained(localization_model, localization_model_id).to(device)

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



def delete_pre_spaces(s):
    """
    Removes leading spaces from each line in the input string and returns the cleaned string.

    Args:
        s (str): The input string.

    Returns:
        str: The cleaned string with leading spaces removed from each line.
    """
    if not type(s) == float:
        lines = s.split('\n')
        cleaned_lines = [line.lstrip() for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line != ""]
        result = '\n'.join(cleaned_lines)
        return result
    else:
        return s



def drop_duplicates(df):
    function_groups = {}

    for i in range(len(df)):
        nonvul = df['nonvul'].iloc[i]
        lines_after_fix = df['vul'].iloc[i]
        # Split the file content into functions (assuming functions are well-defined)
        row = nonvul + lines_after_fix  # Change this based on your function definitions
        function_hash = hashlib.sha256(row.encode()).hexdigest()
        if function_hash not in function_groups:
            function_groups[function_hash] = []
        function_groups[function_hash].append(i)

    indexes_to_drop = [index for index_list in function_groups.values() if len(index_list) > 1 for index in index_list[1:]]
    df = df.drop(indexes_to_drop)
    df = df.reset_index(drop=True)
    return df



def get_accurecy(local_res_path, path_dataset, all_vulgen=False):
    """
    Calculate the accuracy of vulnerability detection.

    Args:
        vuls (list): List of generated vulnerabilities.
        vul_funcs (DataFrame): DataFrame containing vulnerability functions.

    Returns:
        list: List of indices of vulnerabilities that were not detected accurately.
    """
    df = pd.read_csv(local_res_path)
    local_res = df['res_loc'].tolist()
    data = pd.read_csv(path_dataset)
    if all_vulgen:
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        data = data.drop(indexes_to_delete)
        data = data.reset_index(drop=True)
        data = drop_duplicates(data)

    local_truth = data['lines_after_fix'].tolist()
    count = 0
    not_good = []
    for i in range(len(local_res)):
        x = delete_pre_spaces(local_res[i])
        y = delete_pre_spaces(local_truth[i]) 
        if x == y:
            count += 1
        else:
            not_good.append(i)
    print(count/len(local_res))
    return not_good




def main(path_testset, model_huggingface_path, all_vulgen, output_dir):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    load_dotenv()
    # TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    # login(token=TOKEN)
    model, tokenizer = get_model_tokenizer(model_huggingface_path)
    tokenized_local_test = Prepare_data_original.create_testset(tokenizer, with_spaces=True, path_testset=path_testset, is_vulgen=all_vulgen)
    trainer = get_trainer(model, tokenizer, tokenized_local_test)
    local_res = get_prediction(trainer, tokenizer)
    df = pd.DataFrame(local_res, columns=['res_loc'])
    df.to_csv(output_dir, index=True)
    not_good = get_accurecy(output_dir, path_testset, all_vulgen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv', help='Path to test set csv file.')
    parser.add_argument('--model_huggingface_path', type=str, default='urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate', help='Path to load lora adapters form huggingface')
    parser.add_argument('--all_vulgen', type=bool, default=False, help='Is the test set is the whole test set of VulGen?')
    parser.add_argument('--output_dir', type=str, default=False, help='Path to where to save the loclization csv file')
    args = parser.parse_args()
    main(args.path_testset, args.model_huggingface_path, args.all_vulgen, args.output_dir)