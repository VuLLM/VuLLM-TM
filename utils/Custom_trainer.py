
from typing import Dict, Optional, List, Union, Any
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import Seq2SeqTrainer
import transformers
import torch
from torch import nn


class CodeT5pTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ):
        with torch.autocast("cuda"):
            if not self.args.predict_with_generate or prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
                )

            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

            # XXX: adapt synced_gpus for fairscale as well
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

            generation_inputs = inputs.copy()
            # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
            # (otherwise, it would continue generating from the padded `decoder_input_ids`)
            if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
            ):
                generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
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
        return loss, generated_tokens, labels
    

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                                            optimizer=optimizer,
                                            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                        num_training_steps=num_training_steps,
                                            num_cycles=2,
                                            last_epoch=-1)
            self._created_lr_scheduler = True
        return self.lr_scheduler