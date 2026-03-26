from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class CaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base",context=None,device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _prepare_image(image, max_side=640):
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= max_side:
            return image

        scale = max_side / float(longest_side)
        resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(resized_size)
       
    def generate_caption(self, image_path,context, max_length=30 ):
        """
        Generate a caption for a given image.
        Args:
            image_path (str): Path to the image file.
            context (str): Optional context to guide caption generation.
            max_length (int): Maximum length of the generated caption.
        Returns:
            str: The generated caption.
        """
        image = Image.open(image_path).convert("RGB")
        image = self._prepare_image(image)
            ##niche wla if else bhi un comment kr dena aur upr wala image rehne dena same hai wo dono me 
        if context:
            inputs = self.processor(image, context, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        ## this above one input is created for context awared one 
        ## this niche wala is also commented and the added above is the GPT context aware wala
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
            )
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption
