import hashlib
import math
import numpy as np
import os,io
import requests
import time
import torch
from PIL import Image
import base64
from io import BytesIO
import json

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from tqdm import tqdm
from typing import List, Optional
from djl_python import Input, Output
from safetensors.numpy import load_file, save_file

CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
}


@dataclass 
class Config:
    # models can optionally be passed in directly
    caption_model = None
    caption_processor = None

    # blip settings
    caption_max_length: int = 200
    caption_model_name: Optional[str] = 'blip2-2.7b' # use a key from CAPTION_MODELS or None
    caption_offload: bool = False
    
    # interrogator settings
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

   
class Blip2():
    def __init__(self, config: Config, properties):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.load_caption_model(properties)
        self.caption_offloaded = True

    def load_caption_model(self, properties):
        if self.config.caption_model is None and self.config.caption_model_name:
            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if "model_id" in properties and any(os.listdir(properties["model_id"])):
                model_path = properties["model_id"]
                files_in_folder = os.listdir(model_path)
                print('model path files:')
                for file in files_in_folder:
                    print(file)

            print(f'model path: {model_path}')
            if self.config.caption_model_name.startswith('blip2-'):
                caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype, device_map="auto", cache_dir="/tmp", )
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
            self.caption_processor = AutoProcessor.from_pretrained(model_path)

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    def generate_caption(self, pil_image: Image, prompt: Optional[str]=None) -> str:
        assert self.caption_model is not None, "No caption model loaded."
        self._prepare_caption()
        inputs = self.caption_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        if not self.config.caption_model_name.startswith('git-'):
            inputs = inputs.to(self.dtype)

        with torch.no_grad():
            tokens = self.caption_model.generate(**inputs, max_new_tokens=self.config.caption_max_length)
        
        return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def _prepare_caption(self):
        if self.caption_offloaded:
            self.caption_model = self.caption_model.to(self.device)
            self.caption_offloaded = False


with open('./model_name.json', 'rb') as openfile:
    json_object = json.load(openfile)

# config = Config(caption_model_name=json_object.pop('caption_model_name'))
# _service = Blip2(config)
model_name = json_object.pop('caption_model_name')
config = None
_service = None

def handle(inputs: Input) -> Optional[Output]:
    global config, _service
    if not _service:
        config = Config()
        config.caption_model_name=model_name
        _service = Blip2(config, inputs.get_properties())
    
    if inputs.is_empty():
        return None
    data = inputs.get_as_json()

    base64_image_string = data.pop("image")
    
    f = BytesIO(base64.b64decode(base64_image_string))
    input_image = Image.open(f).convert("RGB")
    
    if 'prompt' in data:
        prompt = data.pop("prompt")
    else:
        prompt = None
        
    if 'parameters' in data:
        params = data["parameters"]
        if "max_length" in params.keys():
            config.caption_max_length = params.pop("max_length")
            
    generated_text = _service.generate_caption(input_image, prompt)

    return Output().add(generated_text)
