import subprocess

if __name__ == "__main__":
    # Command to run
    # command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml localization_model/Fine_tuning_accelerator.py"
    # Run the command
    command = "python localization_model/Fine_tuning_one_GPU.py"
    process = subprocess.Popen(command, shell=True)