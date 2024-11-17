
from typing import Optional, List, Tuple, Dict, Union, Any
import torch.nn as nn
import torch
from trl import SFTTrainer
from transformers.integrations.deepspeed import deepspeed_init
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize
from transformers.utils import logging
from torch.cuda.amp import autocast
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import PreTrainedTokenizer
# from torch import xla
# if is_torch_xla_available():
#     import xla.core.xla_model as xm
#     import xla.debug.metrics as met

class Custom_SFTTrainer(SFTTrainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        logger = logging.get_logger(__name__)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize accumulators for metrics
        total_loss = 0.0
        total_accuracy = 0.0
        total_google_bleu = 0.0
        total_sacre_bleu = 0.0
        total_gen_len = 0
        total_batches = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Prediction step
            loss, generated_tokens, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update accumulators
            if loss is not None:
                total_loss += loss.item()

            if generated_tokens is not None and labels is not None:
                # Compute batch-level metrics
                generated_tokens[0].cpu().numpy()
                labels.cpu().numpy()
                batch_metrics = self.compute_metrics((generated_tokens, labels))
                total_accuracy += batch_metrics.get('eval_accuracy', 0)
                total_google_bleu += batch_metrics.get('googleBleu', 0)
                total_sacre_bleu += batch_metrics.get('sacreBleu', 0)
                total_gen_len += batch_metrics.get('gen_len', 0)

            total_batches += 1

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # Finalize metrics
        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        average_google_bleu = total_google_bleu / total_batches
        average_sacre_bleu = total_sacre_bleu / total_batches
        average_gen_len = total_gen_len / total_batches

        metrics = {
            f"{metric_key_prefix}_loss": average_loss,
            f"{metric_key_prefix}_accuracy": average_accuracy,
            f"{metric_key_prefix}_googleBleu": average_google_bleu,
            f"{metric_key_prefix}_sacreBleu": average_sacre_bleu,
            f"{metric_key_prefix}_gen_len": average_gen_len,
        }
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=self.num_examples(dataloader) if has_length(dataloader) else None)
    
    


    def modify_attention(self, data, tokenizer: PreTrainedTokenizer, target_string: str):
        # Tokenize the target string to get the token ids
        target_tokens = tokenizer.encode(target_string, add_special_tokens=False)
        
        # Extract tensors from the BatchEncoding object
        input_ids = data['input_ids']
        masking_attention = data['attention_mask']
        
        # Find positions of the target string's tokens in the input_ids
        # Create a mask for where all the target tokens appear consecutively
        for start_index in range(input_ids.size(1) - len(target_tokens) + 1):
            if torch.all(input_ids[:, start_index:start_index + len(target_tokens)] == torch.tensor(target_tokens).to(input_ids.device)):
                # Found the target string, modify the masking_attention from the end of this string
                end_index = start_index + len(target_tokens)
                if end_index < masking_attention.size(1):
                    masking_attention[:, end_index:] = 0
                break  # Assuming only one occurrence or modifying after the first occurrence only
        original_data = data.copy()
        data['input_ids'] = input_ids[:, :end_index]
        data['attention_mask'] = masking_attention[:, :end_index]
        return data, original_data



    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        self.args.predict_with_generate = True
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        generate_inputs, inputs = self.modify_attention(inputs, self.tokenizer, "Instruction:\n")
        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )
        
        generation_inputs = generate_inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        # Arguments for generation
        gen_kwargs['max_new_tokens'] = 512 
        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # gen_kwargs['do_sample'] = True
        # gen_kwargs['top_p'] = 0.95
         
        with autocast(dtype=torch.bfloat16):
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
            
        generated_tokens = generated_tokens[0][len(generation_inputs['input_ids'][0]):]
        # print("len generated_tokens: ", len(generated_tokens))
        
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, [generated_tokens], labels

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
    #     loss, logits, labels = self.evaluate_completion_only(model, inputs['input_ids'], inputs['labels'], inputs['attention_mask'],"Instruction:\n")
    #     return loss, logits, labels
    
    
    
    # def evaluate_completion_only(self, model, input_ids, labels, attention_mask, instruction_text="Instruction:\n"):
    #     # Encode the instruction tokens to find their positions
    #     instruction_token_ids = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        
    #     # Find the index of the last "Instruction:" token in the input_ids
    #     # This assumes that all tokens are found in sequence and uses the index of the last one
    #     instruction_index = max((input_ids == token_id).nonzero(as_tuple=True)[1].max() for token_id in instruction_token_ids)
        
    #     # Prepare the input for generation, starting from the token after the last "Instruction:" token

    #     input_to_generate = input_ids[:, :instruction_index + 1]  # This should include up to and including the last token of "Instruction:"
    #     attention_mask = attention_mask[:, :instruction_index + 1]
    #     # If the tokenizer does not set a pad token, you might need to define it
    #     if self.tokenizer.pad_token is None:
    #         self.tokenizer.pad_token = self.tokenizer.eos_token  # Commonly used in models like GPT-2/3

    #     # Make sure the model knows the pad token id
    #     model.config.pad_token_id = self.tokenizer.pad_token_id
    #     with autocast(dtype=torch.bfloat16):
    #         # generated_tokens = model.generate(input_ids=input_to_generate, 
    #         #                                   attention_mask=attention_mask, 
    #         #                                   pad_token_id=self.tokenizer.pad_token_id, 
    #         #                                   max_new_tokens=input_to_generate.shape[-1])  # adjust generation length as needed
    #         # generated_tokens = generated_tokens[:, input_ids.shape[1]:][0]
    #         # input_text = "#write a quick sort algorithm"
    #         # inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
    #         # outputs = model.generate(**inputs, max_length=128, pad_token_id=self.tokenizer.pad_token_id )
    #         outputs = model(input_ids=input_to_generate, attention_mask=attention_mask)
    #     # Check if 'labels' is present and adjust accordingly
    #     if labels is not None:
    #         labels_to_use = labels[:, instruction_index + 1:]
    #     else:
    #         labels_to_use = None

    #     # Generating logits for the actual completion
    #     if labels_to_use is not None:
    #         outputs = model(input_ids=input_to_generate, labels=labels_to_use)
    #     else:
    #         outputs = model(input_ids=input_to_generate)

    #     logits = outputs.logits
    #     loss = outputs.loss if 'loss' in outputs else None

    #     return logits, loss

# Use this custom prediction step in your evaluation loop