from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os, sys

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "/workspace/training/model_out/"

unet = UNet2DConditionModel.from_pretrained("/workspace/training/model_out/checkpoint-400/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/workspace/training/model_out/checkpoint-400/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

# Perform inference, or save, or push to the hub
prompt = "A photo of sks dog in formal business suit, humandoid"
for x in range(20):
  print(x)
  image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
  image.save("/workspace/images/dog-bucket.png")
  os.rename('/workspace/images/dog-bucket.png', os.path.join('/workspace/images/',str(x)+'.png'))
