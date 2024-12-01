"""from diffusers import StableDiffusionPipeline
import torch
# Define the local directory for the model
local_model_path = "/projectnb/ece601/24FallA2Group16/stable-diffusion-3.5-large-turbo"

# Download the model to the specified path
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
    use_auth_token="<your hf token>",
    cache_dir="./stable-diffusion-3.5-large-turbo",
    torch_dtype=torch.float16,
)
print(f"Model successfully downloaded to {local_model_path}")
"""



import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", 
    torch_dtype=torch.float32,
    cache_dir="./stable-diffusion-3.5-medium")
pipe = pipe.to("cuda")

image = pipe(
    "A modern minimalist bedroom with clean white walls, a low-profile platform bed with gray bedding, and a large floor-to-ceiling window letting in natural light. Features include sleek wooden furniture, warm ambient lighting, and a soft beige area rug. The decor is simple, with potted plants and abstract wall art. The color scheme is white, gray, and beige, creating a calm and relaxing atmosphere.",
    num_inference_steps=40,
    guidance_scale=5,
).images[0]
image.save("test.png")
