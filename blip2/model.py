from djl_python import Input, Output
import os
import deepspeed
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline

from PIL import Image
import base64
from io import BytesIO

model = None
processor = None

dtype = torch.float16

def get_model(properties):
    tensor_parallel_degree = properties["tensor_parallel_degree"] if "tensor_parallel_degree" in properties else 1
    local_rank = os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0
    processor = Blip2Processor.from_pretrained(properties["model_id"], cache_dir="/tmp")
    # logging.info(f"Loading model in {properties['model_id']}")
    model = Blip2ForConditionalGeneration.from_pretrained(properties["model_id"], device_map="auto", cache_dir="/tmp", torch_dtype=dtype)
    #embedding = model.language_model.get_input_embeddings()
    """
    model.language_model = deepspeed.init_inference(model.language_model,
                                           tensor_parallel={"tp_size": tensor_parallel_degree},
                                           dtype=model.dtype,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)
    """
    device = f"cuda:{local_rank}"
    model.to(device)
    #embedding.to(device)
    #def inject_function():
    #    return embedding
    #model.language_model.get_input_embeddings=inject_function
    return processor, model


def handle(inputs: Input) -> None:
    global model, processor
    if not model:
        processor, model = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    #data = inputs.get_as_string()
    data = inputs.get_as_json()
    
    local_rank = os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0
    device = f"cuda:{local_rank}"
    
    base64_image_string = data.pop("image")
    
    f = BytesIO(base64.b64decode(base64_image_string))
    input_image = Image.open(f).convert("RGB")
    
    if 'prompt' in data:
        prompt = data.pop("prompt")
        inputs = processor(images=input_image, text=prompt, return_tensors="pt").to(device, dtype)
    else:
        inputs = processor(images=input_image, return_tensors="pt").to(device, dtype)
        
    out = model.generate(**inputs, max_new_tokens=200)
    generated_text = processor.decode(out[0], skip_special_tokens=True).strip()
    
    return Output().add(generated_text)
