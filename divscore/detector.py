from typing import Union, Optional
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import entropy
from peft import PeftModel

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}




class DivScore(object):
    def __init__(self,
                 generalLM_name_or_path: str = "",
                 enhancedLM_name_or_path: str = "",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 2048 * 5,
                 device: str = "cuda:7",
                 ) -> None:

        self.DEVICE_1 = self.DEVICE_2 = device
        self.observer_model = AutoModelForCausalLM.from_pretrained(generalLM_name_or_path,
                                                                   device_map={"": self.DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(enhancedLM_name_or_path,
                                                                    device_map={"": self.DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(generalLM_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _load_model_with_lora(
            self,
            base_path: str,
            lora_path: Optional[str],
            device: str,
            use_bfloat16: bool
    ) -> AutoModelForCausalLM:
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        if lora_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                device_map={"": device},
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=huggingface_config["TOKEN"]
            )
            merged_model = PeftModel.from_pretrained(
                base_model,
                lora_path
            ).merge_and_unload()
            return merged_model.to(device)  # 确保模型在目标设备上
        else:
            return AutoModelForCausalLM.from_pretrained(
                base_path,
                device_map={"": device},
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=huggingface_config["TOKEN"]
            )


    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(self.DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.DEVICE_2)).logits

        if self.DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: str):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        entropy_score = entropy(performer_logits.to(self.DEVICE_1), performer_logits.to(self.DEVICE_1),
                        encodings.to(self.DEVICE_1), self.tokenizer.pad_token_id)
        ce_score = entropy(observer_logits.to(self.DEVICE_1), performer_logits.to(self.DEVICE_1),
                        encodings.to(self.DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = entropy_score / ce_score
        binoculars_scores = binoculars_scores.tolist()

        return binoculars_scores[0], entropy_score.item(), ce_score.item()

