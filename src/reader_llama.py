from typing import List, Optional


class DummyReader:
    def generate(self, prompt: str) -> str:
        _ = prompt
        return ""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]


class EchoReader:
    def generate(self, prompt: str) -> str:
        return prompt

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return prompts


class HFReader:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chat_template: str = "auto",
        attn_implementation: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "transformers/torch not available. Install: pip install transformers torch"
            ) from e

        torch_dtype = None
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
            raise ValueError("float16/bfloat16 not supported on cpu; use float32 or auto")

        if attn_implementation not in ("auto", "flash_attention_2", "sdpa", "eager"):
            raise ValueError("attn_implementation must be auto|flash_attention_2|sdpa|eager")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            if device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto", **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                self.model.to(device)
        except Exception as e:
            if attn_implementation == "flash_attention_2":
                raise RuntimeError(
                    "Failed to enable flash_attention_2. "
                    "Make sure flash-attn is installed and transformers is recent."
                ) from e
            raise
        self.model.eval()

        if not getattr(self.model.config, "is_encoder_decoder", False):
            self.tokenizer.padding_side = "left"

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if chat_template not in ("auto", "on", "off"):
            raise ValueError("chat_template must be auto|on|off")
        self.use_chat_template = False
        if chat_template == "on":
            self.use_chat_template = True
        elif chat_template == "auto":
            self.use_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def _build_input(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str) -> str:
        import torch

        input_texts = [self._build_input(prompt)]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = int(inputs["attention_mask"][0].sum().item())
        gen_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        import torch

        input_texts = [self._build_input(p) for p in prompts]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results: List[str] = []
        attention = inputs["attention_mask"]
        for i in range(output_ids.shape[0]):
            input_len = int(attention[i].sum().item())
            gen_ids = output_ids[i][input_len:]
            results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return results


def build_reader(
    reader_type: str,
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    chat_template: str = "auto",
    attn_implementation: str = "auto",
):
    if reader_type == "dummy":
        return DummyReader()
    if reader_type == "echo":
        return EchoReader()
    if reader_type == "hf":
        if not model_path:
            raise ValueError("model_path is required for reader_type=hf")
        return HFReader(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            chat_template=chat_template,
            attn_implementation=attn_implementation,
        )
    raise ValueError(f"Unsupported reader type: {reader_type}")
