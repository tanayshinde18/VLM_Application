import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

class SMOLCaptionGenerator:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-256M-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            _attn_implementation="eager",
        ).to(self.device)
        self.default_prompt = (
            "Describe only what is visible in this CCTV frame <image>. "
            "Write one short sentence about the scene and any clearly visible danger."
        )

    def _build_prompt(self, context=None):
        if not context or context.strip() == "":
            return self.default_prompt
        if "<image>" not in context:
            return context.strip() + " <image>"
        return context.strip()

    def generate_caption(self, image_path, context=None, max_length=30):
        image = load_image(image_path)
        prompt = self._build_prompt(context)

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length
            )

        prompt_length = inputs["input_ids"].shape[1]
        caption_ids = generated_ids[:, prompt_length:]
        caption = self.processor.batch_decode(
            caption_ids,
            skip_special_tokens=True
        )[0]

        return {
            "prompt": prompt,
            "caption": caption.strip(),
        }
