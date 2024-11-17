from arrow import get
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel, PeftConfig
import pandas as pd
import torch
from datasets import Dataset
from pynvml import *
sys.path.append('/sise/home/urizlo/VuLLM')
from localization_model.CodeT5p import Prepare_data_original
from utils import Nxcode_7B, Custom_SFTTrainer
from huggingface_hub import login
from tqdm import tqdm
from collections import defaultdict
import pickle
import hashlib
from dotenv import load_dotenv
import argparse
import evaluate
import numpy as np
import pickle

localization_predicts = []

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



def get_trainer(model, tokenizer, tokenized_local_test):
    
    def generate_prompt(sample, return_response=True):
        return sample['prompt']
    # config evaluation metrics
    metric = evaluate.load("sacrebleu")
    google_bleu = evaluate.load("google_bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        gen_len_list = []

        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Convert preds to tensor if it's a NumPy array
        if isinstance(preds, np.ndarray):
            preds = torch.tensor(preds)
        decoded_preds_original = tokenizer.batch_decode(preds, skip_special_tokens=True)
        localization_predicts.append(decoded_preds_original)

        # Ensure labels are in numpy array format
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds_original, decoded_labels)
        decoded_labels = decoded_labels[0]
        gen_len_list.append([len(tokenizer.encode(pred)) for pred in decoded_preds][0])

        # SacreBleu
        results = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"sacreBleu": results["score"]}

        # GoogleBleu
        results = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["googleBleu"] = results["google_bleu"]

        # Accuracy
        count = 0
        for p, l in zip(decoded_preds, decoded_labels):
            if p == l:
                count += 1
        total_tokens = len(decoded_labels)
        accuracy = count / total_tokens
        result['eval_accuracy'] = accuracy

        # Generation length
        if gen_len_list:
            result["gen_len"] = round(np.mean(gen_len_list), 4)

        result = {k: round(v, 4) for k, v in result.items()}
        # print("Computed metrics:", result)
        return result
    # create trainer object
    training_args = TrainingArguments(
        output_dir="output_dir",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        weight_decay=0.005,
        num_train_epochs=10,
        # predict_with_generate=True,
        bf16=True,
        tf32=True,
        bf16_full_eval=True,
        eval_accumulation_steps=1,
        label_names = ["labels"],
        do_train=True,
        do_eval=True,
        logging_strategy='epoch',
        # generation_max_length=810,
        # generation_num_beams=1,
        dataloader_num_workers=4,
        # warmup_steps=57000,
        report_to="neptune",
        # report_to="none",
        lr_scheduler_type='linear',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    response_template = "Instruction:\n"
    max_seq_length = 1400
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = Custom_SFTTrainer.Custom_SFTTrainer(
        model=model,
        args=training_args,
        dataset_batch_size=1,
        # train_dataset=train,
        data_collator=data_collator,
        eval_dataset=tokenized_local_test,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        formatting_func=generate_prompt,
        # preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    return trainer



def get_prediction(trainer, tokenizer):
    local_res = []
    # Get the total number of batches for the progress bar
    # total_batches = len(trainer.get_eval_dataloader())
    # # Use tqdm to create a progress bar
    # for b in tqdm(trainer.get_eval_dataloader(), total=total_batches, desc="Evaluating"):
    #     out = trainer.model.generate(**b, max_length=359)
    #     text = tokenizer.decode(out[0], skip_special_tokens=True)
    #     local_res.append(text)
    trainer.evaluate()
    local_res = localization_predicts
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
    model, tokenizer = Nxcode_7B.create_model_and_tokenizer_one_GPU(model_huggingface_path)
    test = Prepare_data_original.create_testset_for_only_decoder(tokenizer, with_spaces=True, path_testset=path_testset, is_vulgen=all_vulgen)
    # test = test[:10]
    eos = tokenizer.eos_token
    test['prompt'] = test.apply(lambda row: f"""function:\n{row['nonvul']}\nInstruction:\n{row['lines_after_fix']}{eos}""", axis=1)
    prompt_lengths = test['prompt'].apply(lambda x: len(tokenizer(x, truncation=True, padding=False)['input_ids'])).tolist()
    counter = {index: length for index, length in enumerate(prompt_lengths)}
    counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    indexes_to_delete = [index for index, length in counter.items() if length > 1400]
    # Save indexes as pickle
    with open('connected_models/indexes_to_delete_from_test_codeqwen.pkl', 'wb') as f:
        pickle.dump(indexes_to_delete, f)
    test = test.drop(indexes_to_delete)
    test= Dataset.from_pandas(test)
    
    max_seq_length = 1400

    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length

    test = test.filter(filter_long_samples)
    trainer = get_trainer(model, tokenizer, test)
    local_res = get_prediction(trainer, tokenizer)
    df = pd.DataFrame(local_res, columns=['res_loc'])
    df.to_csv(output_dir, index=True)
    # not_good = get_accurecy(output_dir, path_testset, all_vulgen)

if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv', help='Path to test set csv file.')
    parser.add_argument('--model_huggingface_path', type=str, default='urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate', help='Path to load lora adapters form huggingface')
    parser.add_argument('--all_vulgen', type=bool, default=False, help='Is the test set is the whole test set of VulGen?')
    parser.add_argument('--output_dir', type=str, default=False, help='Path to where to save the loclization csv file')
    args = parser.parse_args()
    main(args.path_testset, args.model_huggingface_path, args.all_vulgen, args.output_dir)