# Deivgn - Effectiveness

## Introduction

LineVul is a SOTA vulnerability detection, aimed at evaluating the effectiveness of VuLLM generated samples. This project facilitates running tests across different datasets and training configurations to benchmark performance accurately.

## Table of Contents

1. [General Information](#general-information)
3. [How to Run Tests](#how-to-run-tests)
4. [Training and Testing on New Data](#training-and-testing-on-new-data)

## General Information

- This README provides all the necessary information to run and test the LineVul project effectively.
- Explore the project on GitHub: [LineVul GitHub Repository](https://github.com/awsm-research/LineVul)

## How to Run Tests

The project includes a comprehensive set of scripts to run tests as described in Table 2 of the related article. In the `run_trails` folder, there are two subfolders: `balanced` and `imbalanced`, each containing scripts for different test configurations:

- `Baseline.sh`: Runs the test in Table 2, column "baseline."
- `Ground_truth.sh`: Runs the test in Table 2, column "ground_truth."
- `VuLLM.sh`: Runs the test in Table 2, column "VuLLM."
- `Syn.sh`: Runs the test in Table 2, column "syn."
- `VGX.sh`: Runs the test in Table 2, column "VGX."
- `VulGen.sh`: Runs the test in Table 2, column "VulGen."
- `Wild.sh`: Runs the test in Table 2, column "Wild."

## Training and Testing on New Data

To train and test the model on new data, follow these steps:
   - Create new sh file like 'detector_models/LineVul/run_trails/balanced/Baseline.sh'
   - Change this 3 fileds with path to new data set.
      -   --train_data_file
      -   --eval_data_file
      -   --test_data_file


