import subprocess

# Command to run
# command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml --num_processes=5 --use_deepspeed --deepspeed_config_file accelerate_config_files/deepspeed.json one_model/Fine_tuning_accelerator.py"
command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml Injection_model/Fine_tuning_accelerator.py"

# Run the command
process = subprocess.Popen(command, shell=True)
