from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIPBaseCaptionGenerator:
    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def generate_caption(self, image_path, context=None, max_length=30):
        image = Image.open(image_path).convert("RGB")

        if context and context.strip():
            inputs = self.processor(image, context.strip(), return_tensors="pt").to(self.device)
            prompt = context.strip()
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            prompt = ""

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
            )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return {
            "prompt": prompt,
            "caption": caption,
        }
