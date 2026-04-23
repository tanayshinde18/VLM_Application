import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
PROMPT = (
    "Describe what is happening in this image in one neat sentence. "
    "Mention the main subject, action, and visible setting."
)
IMAGE_SIZE = (224, 224)


class SmolVLM500MCaptioner:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def _prepare_image(self, image_path: str | Path) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image.resize(IMAGE_SIZE)

    def _build_inputs(self, image: Image.Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prepared = {}
        for key, value in inputs.items():
            tensor = value.to(self.device)
            if torch.is_floating_point(tensor):
                tensor = tensor.to(self.dtype)
            prepared[key] = tensor
        return prepared

    def caption_image(self, image_path: str | Path, max_new_tokens: int = 48) -> str:
        image = self._prepare_image(image_path)
        inputs = self._build_inputs(image)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[1]
        caption_ids = generated_ids[:, prompt_length:]
        caption = self.processor.batch_decode(caption_ids, skip_special_tokens=True)[0].strip()
        return caption


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption an image with SmolVLM-500M-Instruct.")
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum tokens to generate.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    captioner = SmolVLM500MCaptioner()
    caption = captioner.caption_image(image_path, max_new_tokens=args.max_new_tokens)
    print(caption)


if __name__ == "__main__":
    main()
