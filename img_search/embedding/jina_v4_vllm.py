from typing import Literal, override

import torch
from PIL import Image
from transformers.image_utils import load_image
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt

from img_search.utils.logging import ensure_loguru_bridge

from .encoder import Encoder


class JinaV4VLLMEncoder(Encoder):
    def __init__(
        self,
        model_name="jinaai/jina-embeddings-v4-vllm-retrieval",
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        data_parallel: bool = False,
    ):
        super().__init__("JinaV4VLLM")
        self.model_name = model_name
        self.dtype = dtype
        self.VISION_START_TOKEN_ID, self.VISION_END_TOKEN_ID = 151652, 151653

    def build(self):
        self._model = LLM(
            model=self.model_name,
            task="auto",
            override_pooler_config=PoolerConfig(pooling_type="LAST", normalize=False),
            dtype=self.dtype,
            runner="pooling",
        )
        ensure_loguru_bridge("vllm", default_level="WARNING")

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    def get_text_prompt(
        self, text: str, prompt_name: Literal["query", "passage", "code"]
    ) -> TextPrompt:
        if prompt_name == "query":
            return TextPrompt(
                prompt=f"Query: {text}",
            )
        elif prompt_name == "passage":
            return TextPrompt(
                prompt=f"Passage: {text}",
            )
        else:
            raise ValueError(
                f"Prompt name {prompt_name} not found, available prompt names: {['query', 'passage', 'code']}"
            )

    def get_image_prompt(self, image: Image.Image | str) -> TextPrompt:
        pil_image = load_image(image)
        return TextPrompt(
            prompt=(
                "<|im_start|>user\n"
                + "<|vision_start|><|image_pad|><|vision_end|>"
                + "Describe the image."
                + "<|im_end|>\n"
            ),
            multi_modal_data={"image": pil_image},
        )

    @override
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        prompt_name: Literal["query", "passage"] | None = "query",
        **kwargs,
    ) -> torch.Tensor:
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")
        inputs = []
        if texts:
            inputs = [self.get_text_prompt(text, prompt_name) for text in texts]
        elif images:
            inputs = [self.get_image_prompt(image) for image in images]
        outputs = self.model.encode(inputs)
        embeddings = []
        for output in outputs:
            if self.VISION_START_TOKEN_ID in output.prompt_token_ids:
                # Gather only vision tokens
                img_start_pos = torch.where(
                    torch.tensor(output.prompt_token_ids) == self.VISION_START_TOKEN_ID
                )[0][0]
                img_end_pos = torch.where(
                    torch.tensor(output.prompt_token_ids) == self.VISION_END_TOKEN_ID
                )[0][0]
                embeddings_tensor = output.outputs.data.detach().clone()[
                    img_start_pos : img_end_pos + 1
                ]
            else:
                # Use all tokens for text-only prompts
                embeddings_tensor = output.outputs.data.detach().clone()

            # Pool and normalize embeddings
            pooled_output = (
                embeddings_tensor.sum(dim=0, dtype=torch.float32)
                / embeddings_tensor.shape[0]
            )
            embeddings.append(torch.nn.functional.normalize(pooled_output, dim=-1))
        return embeddings
