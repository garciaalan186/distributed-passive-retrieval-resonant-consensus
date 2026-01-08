"""
Infrastructure: Transformers Backend

Concrete implementation of IModelBackend using HuggingFace transformers.
"""

import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersBackend:
    """
    Model backend using HuggingFace transformers.

    Wraps the model and tokenizer for inference.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
    ):
        """
        Initialize backend.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            torch_dtype: Optional dtype for model
            device_map: Optional device map for multi-GPU (e.g., "auto")
        """
        self.model_id = model_id
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with optional device_map for multi-GPU
        load_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
        if device_map:
            load_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)

        self.model.eval()  # Set to evaluation mode

    def generate(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract generated portion (remove prompt)
        generated_text = full_text[len(prompt):].strip()

        return generated_text

    def get_model_id(self) -> str:
        """Get model identifier."""
        return self.model_id
