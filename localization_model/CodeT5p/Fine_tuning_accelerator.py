import transformers
from transformers import (Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
import torch
import numpy as np
import bitsandbytes as bnb
import neptune
from accelerate import Accelerator
import evaluate
import sys
import os
sys.path.append('/sise/home/urizlo/VuLLM')
from utils import Custom_trainer, CodeT5p_6B, Create_lora
import Prepare_data_original
from dotenv import load_dotenv
import argparse



def main(path_trainset, path_testset, is_vulgen, output_dir, learning_rate, per_device_train_batch_size, num_train_epochs, generation_num_beams):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    load_dotenv()
    accelerator = Accelerator()

    # get model and tokenizer and desgin architecture of the model
    checkpoint = "Salesforce/codet5p-6b"
    # peft_model_id = "urizlo/CodeT5-2B-lora-dropout-r32-batch2-6GPUs-lr1e-5-epochs25-allLinearsAdaptors"
    model, tokenizer = CodeT5p_6B.create_model_and_tokenizer(checkpoint, multi_gpu=True)

    # read and tokenized data
    tokenized_train, tokenized_test = Prepare_data_original.create_datasets(tokenizer, with_spaces=True, path_trainset=path_trainset, path_testset=path_testset, is_vulgen=is_vulgen)

    # create lora adaptors
    model = Create_lora.create_lora(model, rank=32, dropout=0.05)

    # config evaluation metrics
    metric = evaluate.load("sacrebleu")
    google_bleu = evaluate.load("google_bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        #ScareBleu
        results = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"sacreBleu": results["score"]}
        #GoogleBlue
        results = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["googleBleu"] = results["google_bleu"]
        #Accuracy
        count = 0
        for p, l in zip(decoded_preds, decoded_labels):
            if p == l[0]:
                count += 1
        total_tokens = len(decoded_labels)
        accuracy = count / total_tokens
        result['accuracy'] = accuracy
        #Genaration length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    # config env varibles
    # NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
    # NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
    # os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
    # os.environ["NEPTUNE_PROJECT"] = NEPTUNE_PROJECT
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # model configurations
    model.config.max_length=420
    model.config.use_cache=True
    model.config.decoder_start_token_id = tokenizer.bos_token_id  # Replace tokenizer.cls_token_id with the appropriate token ID
    model.config.pad_token_id = 50256
    model.config.decoder.pad_token_id = model.config.decoder.eos_token_id
    model.config.encoder.pad_token_id = model.config.encoder.eos_token_id
    model.config.encoder.max_length = 420
    model.config.decoder.max_length = 420

    # create trainer object
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.001,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        logging_dir="TensorBoard",
        do_train=True,
        do_eval=True,
        logging_strategy='epoch',
        generation_max_length=416,
        generation_num_beams=generation_num_beams,
        dataloader_num_workers=4,
        warmup_steps=22000,
        report_to="none",
        lr_scheduler_type="linear",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    trainer = Custom_trainer.CodeT5pTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    accelerator.wait_for_everyone()
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
    main(args.path_trainset, args.path_testset, args.is_vulgen, args.output_dir, args.learning_rate, args.batch_size_per_device, args.epochs, args.generation_num_beams)

    # command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml localization_model/Fine_tuning_accelerator.py --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1"