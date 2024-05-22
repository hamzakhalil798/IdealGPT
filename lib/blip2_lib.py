# from lavis.models import load_model_and_preprocess
# from PIL import Image

# class Blip2Lavis():
#     def __init__(self, name="blip2_opt", model_type="pretrain_opt6.7b", device="cuda"):
#         self.model_type = model_type
#         self.blip2, self.blip2_vis_processors, _ = load_model_and_preprocess(
#             name=name, model_type=model_type, is_eval=True, device=device)
#         # if 't5xl' in self.model_type:
#         #     self.blip2 = self.blip2.float()
#         self.device = device

#     def ask(self, img_path, question, length_penalty=1.0, max_length=30):
#         raw_image = Image.open(img_path).convert('RGB')
#         image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
#         if 't5' in self.model_type:
#             answer = self.blip2.predict_answers({"image": image, "text_input": question}, length_penalty=length_penalty, max_length=max_length)
#         else:
#             answer = self.blip2.generate({"image": image, "prompt": question}, length_penalty=length_penalty, max_length=max_length)
#         answer = answer[0]
#         return answer

#     def caption(self, img_path, prompt='a photo of'):
#         # TODO: Multiple captions
#         raw_image = Image.open(img_path).convert('RGB')
#         image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
#         # caption = self.blip2.generate({"image": image})
#         caption = self.blip2.generate({"image": image, "prompt": prompt})
#         caption = caption[0].replace('\n', ' ').strip()  # trim caption
#         return caption


import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

class Blip2HuggingFace:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", model_type='Blip2HuggingFace',device="cuda" if torch.cuda.is_available() else "cpu"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = device
        self.model.to(device)

    def caption(self, img_path):
        image = self._load_image(img_path)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def ask(self, img_path, question_prompt,max_length=10):
        image = self._load_image(img_path)
        prompt = f"Question: {question_prompt} Answer:"
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def _load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image
