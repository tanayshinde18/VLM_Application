import torch
from transformers import AutoProcessor, AutoModelForVision2Seq ,AutoModelForImageTextToText
from transformers.image_utils import load_image

class SMOLCaptionGenerator:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-256M-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        ).to(self.device)
        self.default_prompt=(
            "Analyze the following CCTV frame: <image>. "
            "NO PREAMBLE"
            "Describe the scene in detail and mention if there is any unusual activity in about 100 words, "
            "accident, robbery, theft, fight, fire, or other mishappening."

        )
      

    def generate_caption(self, image_path, context=None, max_length=150):
        image = load_image(image_path)

        #context="Describe this CCTV frame in detail and mention if there is any unusual activity,accident, robbery, theft, fight, fire, or other mishappening"
        # Direct text+image input without chat template

        if not context or context.strip()=="":
            prompt=self.default_prompt
        else:
            if "<image>" not in context:
                prompt=context.strip()+" <image>"
            else:
                prompt=context



        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            generated_ids=self.model.generate(
                **inputs,
                max_new_tokens=max_length
            )
        
        generated_texts=self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
            
        return generated_texts.strip()
