from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import numpy as np
import neptune
import evaluate
import os
# import ..utils
import sys
sys.path.append('/sise/home/urizlo/VuLLM')
from utils import Nxcode_7B, Create_lora_starCoder, Custom_SFTTrainer
from Injection_model import Prepare_dataset_with_unique_lines
import argparse
from dotenv import load_dotenv
from datasets import Dataset



def main(path_trainset, path_testset, is_vulgen, output_dir, learning_rate, per_device_train_batch_size, num_train_epochs, generation_num_beams):    
    checkpoint = "Qwen/CodeQwen1.5-7B"
    load_dotenv()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_seq_length = 1450

    model, tokenizer = Nxcode_7B.create_model_and_tokenizer(checkpoint)
    eos = tokenizer.eos_token
    # read and tokenized data
    train, test = Prepare_dataset_with_unique_lines.create_datasets_for_only_decoder(tokenizer, path_trainset, path_testset, is_vulgen=is_vulgen)
    train['prompt'] = train.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    test['prompt'] = test.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    train = Dataset.from_pandas(train)
    test= Dataset.from_pandas(test)
    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length

    # Apply the filter
    train = train.filter(filter_long_samples)
    test = test.filter(filter_long_samples)
    # create lora adaptors
    model = Create_lora_starCoder.create_lora(model, rank=32, dropout=0.05)

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
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Ensure labels are in numpy array format
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
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

    # config env varibles
    os.environ["NEPTUNE_API_TOKEN"] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4Y2VlNTFhZC1hODJkLTQ4NzItOTE0MS0yZmNkNWY3ZWE0MTEifQ=='
    os.environ["NEPTUNE_PROJECT"] = 'zlotman/Localization-model'
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        weight_decay=0.005,
        num_train_epochs=num_train_epochs,
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
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = Custom_SFTTrainer.Custom_SFTTrainer(
        model=model,
        args=training_args,
        dataset_batch_size=1,
        train_dataset=train,
        data_collator=data_collator,
        eval_dataset=test,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        formatting_func=generate_prompt,
        # preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_trainset', type=str, default='Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv', help='Path to trainset csv file')
    parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv', help='Path to testset csv file')
    parser.add_argument('--is_vulgen', type=bool, default=False, help='Is trainset and test are from vulgen dataset?')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Output directory for the saved model')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size_per_device', type=int, default=1, help='Batch size per device')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--generation_num_beams', type=int, default=1, help='Number of beams for generation')
    args = parser.parse_args()
    main(args.path_trainset, args.path_testset, args.is_vulgen ,args.output_dir, args.learning_rate, args.batch_size_per_device, args.epochs, args.generation_num_beams)