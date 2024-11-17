import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import PeftModel, PeftConfig



def create_model_and_tokenizer(checkpoint):
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token    
    
    # tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                            torch_dtype=torch.bfloat16,
                                            device_map=device_map)
    return model, tokenizer

def create_model_and_tokenizer_one_GPU(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.padding_side = 'right'
    # tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             device_map="auto", torch_dtype=torch.bfloat16)
    model.generation_config.do_sample = False
    model.generation_config.top_p = 0
    return model, tokenizer


def load_model_and_tokenizer(peft_model_id, from_hub=False, eval=False):
    device = "cuda"
    if from_hub:
        # device_index = Accelerator().process_index
        # device_map = {"": device_index}
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True,
                                                        torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model_id)

    else:
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
        # device_index = Accelerator().process_index
        # device_map = {"": device_index}
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path="saved_models/local_model/6B/second_round/checkpoint-7770",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).to(device)    

    
    if not eval:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for param in model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
            # model.gradient_checkpointing_enable()  # reduce number of stored activations
            model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        model.decoder.lm_head = CastOutputToFloat(model.decoder.lm_head)
    return model, tokenizer