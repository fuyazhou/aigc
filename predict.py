

# if Checkpoints only save the unet
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "checkpoint-path"

unet = UNet2DConditionModel.from_pretrained(model_path + "/unet")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet)

pipe.to("cuda")
image = pipe(prompt="").images[0]
image.save("image.png")



# if last Checkpoints  saved
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe.to("cuda")

image = pipe(prompt="").images[0]
image.save("image.png")

