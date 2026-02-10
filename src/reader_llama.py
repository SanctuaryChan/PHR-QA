from typing import Optional


class DummyReader:
    def generate(self, prompt: str) -> str:
        _ = prompt
        return ""


class EchoReader:
    def generate(self, prompt: str) -> str:
        return prompt


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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype
            )
            self.model.to(device)
        self.model.eval()

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

        input_text = self._build_input(prompt)
        inputs = self.tokenizer(input_text, return_tensors="pt")
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
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def build_reader(
    reader_type: str,
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    chat_template: str = "auto",
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
        )
    raise ValueError(f"Unsupported reader type: {reader_type}")
