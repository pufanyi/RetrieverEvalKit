from typing import Literal, override

import torch
from PIL import Image
from transformers.image_utils import load_image
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt

from img_search.utils.logging import ensure_loguru_bridge

from .encoder import Encoder


class SiglipVLLMEncoder(Encoder):
    def __init__(
        self,
        model_name="google/siglip-so400m-patch14-384",
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        data_parallel: bool = False,
    ):
        super().__init__("SiglipVLLM")
        self.model_name = model_name
        self.dtype = dtype
        self._model = None

    def build(self):
        self._model = LLM(
            model=self.model_name,
            task="embed",
            override_pooler_config=PoolerConfig(pooling_type="MEAN", normalize=True),
            dtype=self.dtype,
            trust_remote_code=True,
        )
        ensure_loguru_bridge("vllm", default_level="WARNING")

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    def get_text_prompt(self, text: str) -> TextPrompt:
        return TextPrompt(prompt=text)

    def get_image_prompt(self, image: Image.Image | str) -> TextPrompt:
        pil_image = load_image(image)
        return TextPrompt(
            prompt="",  # SigLIP doesn't require text prompt for image encoding
            multi_modal_data={"image": pil_image},
        )

    @override
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        prompt_name: Literal["query", "passage"] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")

        inputs = []
        if texts:
            inputs = [self.get_text_prompt(text) for text in texts]
        elif images:
            inputs = [self.get_image_prompt(image) for image in images]
        else:
            raise ValueError("Either texts or images must be provided")

        outputs = self.model.encode(inputs)
        embeddings = []

        for output in outputs:
            # Get the embeddings directly from output
            embeddings_tensor = output.outputs.data.detach().clone()

            # Pool embeddings (mean pooling)
            pooled_output = embeddings_tensor.mean(dim=0, dtype=torch.float32)

            # Normalize embeddings
            normalized_embedding = torch.nn.functional.normalize(pooled_output, dim=-1)
            embeddings.append(normalized_embedding)

        return torch.stack(embeddings)
