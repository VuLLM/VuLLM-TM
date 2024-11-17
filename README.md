# VuLLM - TM

## Introduction

VuLLM is tool for injecting vulnerabilities to C-code functions. VuLLM is utilizing Code LLM (CodeQwen1.5-7B) twice: once to learn to locations where vulnerabilities should be injected; and the second to learn the specific text modification instructions for generating the vulnerable C-Code function.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#Contact)

## Project Structure
```bash
VuLLM/
├── README.md                                               # Project overview and setup instructions
├── .gitignore                                              # Specifies intentionally untracked files to ignore
├── LICENSE                                                 # License information for the project
├── requirements.txt                                        # License information for the project
├── localiztion_model/                                      # Fine-tuning localiztion model on CodeT5+ as base model   
|   ├── run_deepspeed_train.py                              # Run Fine_tuninig_accelerator.py file           
│   └── CodeT5p/
│       ├── Fine_tuninig_accelerator.py                     # Fine-tuning the model on multuply GPUs
│       ├── Fine_tuninig_one_GPU.py                         # Fine-tuning the model on one GPU
|       └── Prepare_data_original.py                        # Preprocess VulGen Dataset for fine-tuning localiztion model
├── injection_model/                                        # Fine-tuning injection model on CodeT5+ as base model            
│   ├── Fine_tuninig_accelerator.py                         # Fine-tuning the model on multuply GPUs
│   ├── Fine_tuninig_one_GPU.py                             # Fine-tuning the model on one GPU
|   ├── run_deepspeed_train.py                              # Run Fine_tuninig_accelerator.py file           
|   └── Prepare_dataset_with_unique_lines.py                # Preprocess VulGen Dataset with unique duplicated lines for fine-tuning injection model
├── connected_models/                                       # Infer
nce both models
|   ├── pipeline_localization_vulgen.py                     # Inference localization model on VulGen test set and save the results to connected_models/localization_results/res_loc.csv
|   ├── pipeline_injection_vulgen.py                        # Inference Injection model on VulGen test set and save the results to connected_models/injection_results/res_inj.csv   
|   ├── replace_function_with_line_spaces_vulgen .py        # Operate the replacment component, get non-vulnerable function and the output of localization\injection model and return the vulnerable new function     
|   ├── pipeline_localization_custom.py                     # Inference localization model on custom test set and save the results to connected_models/localization_results/res_loc.csv
|   ├── pipeline_injection_custom.py                        # Inference Injection model on custom test set and save the results to connected_models/injection_results/res_inj.csv   
|   ├── replace_function_with_line_spaces_custom_dataset.py # Operate the replacment component, get non-vulnerable function and the output of localization\injection model and return the vulnerable new function     
│   ├── Dataset/                                            # Empty folder that can contain dataset to infernce the models on it
|   ├── generated_vul/                                      # Folder that contain the generate vulnerable functions
│   |   └── vulgen_res.csv                                  # CSV with new vulnerable function from VulGen test set
|   ├── localization_results/                               # Folder that contain the output of localization model
|   |   └── vulgen_res_loc.csv                              # CSV with the outpus of the localization model
|   ├── injection_results/                                  # Folder that contain the output of localization model
|       └── vulgen_inject_res.csv                           # CSV with the outpus of the injection model
├── pickle_files/                                           # Contain pickle files that are lists with the indexes to drop in the dataframe of VulGen dataset, becuse this sample has more than 2,048 tokens
|   ├── location_with_spaces_to_delete_train.pkl            # Indexes to drop in location train set
|   ├── location_with_spaces_to_delete_test.pkl             # Indexes to drop in location test set
|   ├── inject_with_spaces_to_delete_train.pkl              # Indexes to drop in injection train set
|   └──  inject_with_spaces_to_delete_test.pkl              # Indexes to drop in injection test set
├── utils/                                                  # Contain py files that useful functions
|   ├── CodeT5p_6B.py                                       # Contain help functions for fine-tuning and testing the models
|   ├── Create_lora.py                                      # Rapper for create LORA when fine-tuning models
|   ├── Custom_trainer.py                                   # Overwrite the Seq2SeqTrainer Huggingface models, to help fine-tuning
├── saved_models/                                           # Folder that contain models while fine-tuning, and in the end of fine-tuning
├── Exploratory data analysis/                              
|   └── add_diff_lines.py                                   # Contain functions that get dataset of non-vulnerable functions and add spaces to the ends of the duplicated lines of code in each function to make them unique
├── accelerate_config_files/                                # Contain confiuration files for fine-tuning with DeepSpeed and accelerate package from Huggingface to fine-tuning on multi-GPUs
|   ├── deepspeed_stage2.yaml                               # Configuration for DeepSpeed
|   └── deepspeed.json                                      # Configuration for DeepSpeed
├── detector_models/                                        # Contain all files of Devign and LineVul model for the effictevness test that show in Table 2
|    ├── devign/                                            # read README.md for this folder in this path detector_models/devign/VuLLM_README.md
|    └── LineVul/                                           # read README.md for this folder in this path detector_models/LineVul/VuLLM_README.md
├── Dataset_VulGen/                                         # Contain CSVs file with samples from VulGen dataset
|   ├── README.md                                           # There is Link to Google Drive to download this CSVs files
└── requirements.txt                                        # Project dependencies
```

## Installation
- python 3.9+<br>
To install the necessary dependencies, run the following command:
```sh
pip install -r requirements.txt
```


## Usage

### Download VulGen Dataset from Google Drive

Download files from here: https://drive.google.com/file/d/1hzq_i01IqKSIaGcKEpkw3OHUlD7neScT/view?usp=drive_link

### Fine-tuning the Localization Model

#### Running the Scripts
To fine-tune the CodeT5+ 6B model, use one of the following scripts depending on your setup:
- For a single GPU setup, run `localization_model/CodeT5p/Fine_tuning_one_GPU.py`.
- For multi-GPU setups using accelerators, run `localization_model/CodeT5p/Fine_tuning_accelerator.py` with this command in terminal: `NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml localization_model/CodeT5p/Fine_tuning_accelerator.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`, you can change the arguments.

#### Running with Neptune
Create `.env` file for this 2 lines if you want to use neptune:
- NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
- NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")

If you do not want running with Neptune delete this 2 lines, and change `report_to=None` argument in Seq2SeqTrainingArguments.

#### Fine-tuning on VulGen Dataset

- If you want to use VulGen Dataset `is_vulgen=True`
- If not 
    - 1.create CSV file with 2 column: `vul` and `nonvul` are pairs of the same function one is the vulnerable form `vul` and it corresponded fix version is `nonvul`
    - 2.Use `Exploratory data analysis/add_diff_lines.py` python file to create csv file with this columns: `vul`, `nonvul`, `lines_after_fix`

#### Arguments
When running the fine-tuning scripts, the following arguments can be specified:
- `path_trainset` (str): Path to train set csv file.
- `path_testset` (str): Path to test set csv file.
- `is_vulgen` (bool): Is the training set and the test set are from the vulgen dataset.
- `output_dir` (str): The directory where the fine-tuned model will be saved, both during and after the fine-tuning process.
- `learning_rate` (float): The learning rate for the fine-tuning model.
- `batch_size_per_device` (int): The batch size per GPU device.
- `epochs` (int): The number of epochs to run for the fine-tuning process.
- `generation_num_beams` (int): The number of beams to use during the generation phase in evaluation.

#### Example for running
`python localization_model/CodeT5p/Fine_tuning_one_GPU.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`

### Fine-tuning the Injection Model

#### Running the Scripts
To fine-tune the CodeT5+ 6B model, use one of the following scripts depending on your setup:
- For a single GPU setup, run `injection_model/CodeT5p/Fine_tuning_one_GPU.py`.
- For multi-GPU setups using accelerators, run `injection_model/CodeT5p/Fine_tuning_accelerator.py` with this command in terminal: `command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml injection_model/Fine_tuning_accelerator.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`, you can change the arguments.

#### Running with Neptune
Create `.env` file for this 2 lines if you want to use neptune:
- NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
- NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")

If you do not want running with Neptune delete this 2 lines, and change `report_to` argument in Seq2SeqTrainingArguments.

#### Fine-tuning on VulGen Dataset or Custom Dataset

- If you want to use VulGen Dataset argument `is_vulgen=True`
- If not 
    - 1.create CSV file with 2 column: `vul` and `nonvul` are pairs of the same function one is the vulnerable form `vul` and it corresponded fix version is `nonvul`
    - 2.Use `Exploratory data analysis/add_diff_lines.py` python file to create csv file with this columns: `vul`, `nonvul`, `lines_after_fix`

#### Arguments
When running the fine-tuning scripts, the following arguments can be specified:
- `path_trainset` (str): Path to train set csv file.
- `path_testset` (str): Path to test set csv file.
- `is_vulgen` (bool): Is the training set and the test set are from the vulgen dataset.
- `output_dir` (str): The directory where the fine-tuned model will be saved, both during and after the fine-tuning process.
- `learning_rate` (float): The learning rate for the fine-tuning model.
- `batch_size_per_device` (int): The batch size per GPU device.
- `epochs` (int): The number of epochs to run for the fine-tuning process.
- `generation_num_beams` (int): The number of beams to use during the generation phase in evaluation.

#### Example for running
`python Injection_model/Fine_tuning_one_GPU.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`


### Get Results of Acuurecy in Table 1 - Infernce all process (localization, injection models and replacment component).

#### Infernce Localization model on 775 samples from VulGen test set

Run `connected_models/pipeline_localization_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `output_dir` (srt): Path to where to save the loclization csv file.

To run on 775 samples from VulGen, path_testset: `Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv`.

#### Example for running

`python connected_models/pipeline_localization_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-inject-acc0.6793-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen False --output_dir connected_models/localization_results/vulgen_775_res_loc.csv`

#### Infernce Injection model on 775 samples from VulGen test set

Run `connected_models/pipeline_injection_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `path_res_local` (str): Path to localization results of the testset.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`python connected_models/pipline_injection_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen True --path_res_local connected_models/localization_results/vulgen_res_loc.csv --output_dir connected_models/injection_results/vulgen_res_inj.csv`

#### Operate Replacment component

Run `connected_models/replace_function_with_line_spaces_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `path_res_local` (str): Path to localization results of the testset.
- `path_res_inj` (str): Path to injection results of the testset.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `output_dir` (srt): Path to where to save the new vulnerable functions csv file.

#### Example for running

`python connected_models/replace_function_with_line_spaces_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --path_res_local connected_models/localization_results/vulgen_res_loc.csv  --path_res_inj connected_models/injection_results/vulgen_res_inj.csv --all_vulgen True --output_dir connected_models/genrated_vul/vulgen_res.csv`

### Infernce all process (localization, injection models and replacment component) for custom dataset.

#### Infernce Localization model on custom dataset

Run `connected_models/pipeline_localization_custom.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`python connected_models/pipeline_localization_custom.py --path_testset Dataset_VulGen/full_testset_changes_smaller_than_20_words.csv --model_huggingface_path saved_models/localization_newData/checkpoint-567275 --output_dir connected_models/localization_results/fullData_res_loc.csv`

#### Infernce Injection model on custom dataset

Run `connected_models/pipeline_injection_custom.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `path_res_local` (str): Path to localization results of the testset.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`connected_models/pipline_injection_custom.py --path_testset Dataset_VulGen/full_testset_changes_smaller_than_20_words.csv --model_huggingface_path saved_models/inject_newData/checkpoint-505103 --path_res_local connected_models/localization_results/fullData_res_loc.csv --output_dir connected_models/injection_results/fullData_res_inj.csv`

#### Operate Replacment component

Run `connected_models/replace_function_with_line_spaces_custom_dataset.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `path_res_local` (str): Path to localization results of the testset.
- `path_res_inj` (str): Path to injection results of the testset.
- `output_dir` (srt): Path to where to save the new vulnerable functions csv file.

#### Example for running

`python connected_models/replace_function_with_line_spaces_custom_dataset.py --path_testset Dataset_VulGen/full_testset_changes_smaller_than_20_words.csv --path_res_local connected_models/localization_results/fullData_res_loc.csv  --path_res_inj connected_models/injection_results/fullData_res_inj.csv  --output_dir connected_models/generated_vul/custom_res.csv`


### Get Results of Effectiveness in Table 2.

#### LineVul

Read README.md for this folder in this path `detector_models/LineVul/VuLLM_README.md`

#### Devign

Read README.md for this folder in this path `detector_models/Devign/VuLLM_README.md`


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact


