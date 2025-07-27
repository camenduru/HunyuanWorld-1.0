# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].
import os
import torch
import numpy as np 

import cv2
from PIL import Image

import argparse

# huanyuan3d text to panorama
from hy3dworld import Text2PanoramaPipelines

# huanyuan3d image to panorama
from hy3dworld import Image2PanoramaPipelines
from hy3dworld import Perspective

from diffusers.quantizers import PipelineQuantizationConfig


class Text2PanoramaDemo:
    def __init__(self):
        # set default parameters
        self.height = 960
        self.width = 1920

        # panorama parameters
        # these parameters are used to control the panorama generation
        # you can adjust them according to your needs
        self.guidance_scale = 30
        self.shifting_extend = 0
        self.num_inference_steps = 50
        self.true_cfg_scale = 0.0
        self.blend_extend = 6

        # model paths
        self.lora_path = "tencent/HunyuanWorld-1"
        self.model_path = "black-forest-labs/FLUX.1-dev"
        # load the pipeline
        # use bfloat16 to save some VRAM

        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
            components_to_quantize=["transformer"]
        )

        self.pipe = Text2PanoramaPipelines.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        # and enable lora weights
        self.pipe.load_lora_weights(
            self.lora_path,
            subfolder="HunyuanWorld-PanoDiT-Text",
            weight_name="lora.safetensors",
            torch_dtype=torch.bfloat16
        )
        # save some VRAM by offloading the model to CPU
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()  # and enable vae tiling to save some VRAM

    def run(self, prompt, negative_prompt=None, seed=42, output_path='output_panorama'):
        # get panorama
        image = self.pipe(
            prompt,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            blend_extend=self.blend_extend,
            true_cfg_scale=self.true_cfg_scale,
        ).images[0]

        # create output directory if it does not exist
        os.makedirs(output_path, exist_ok=True)
        # save the panorama image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        # save the image to the output path
        image.save(os.path.join(output_path, 'panorama.png'))

        return image


class Image2PanoramaDemo:
    def __init__(self):
        # set default parameters
        self.height, self.width = 960, 1920  # 768, 1536 #

        # panorama parameters
        # these parameters are used to control the panorama generation
        # you can adjust them according to your needs
        self.THETA = 0
        self.PHI = 0
        self.FOV = 80
        self.guidance_scale = 30
        self.num_inference_steps = 50
        self.true_cfg_scale = 2.0
        self.shifting_extend = 0
        self.blend_extend = 6

        # model paths
        self.lora_path = "tencent/HunyuanWorld-1"
        self.model_path = "black-forest-labs/FLUX.1-Fill-dev"
        # load the pipeline
        # use bfloat16 to save some VRAM

        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
            components_to_quantize=["transformer"]
        )

        self.pipe = Image2PanoramaPipelines.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        # and enable lora weights
        self.pipe.load_lora_weights(
            self.lora_path,
            subfolder="HunyuanWorld-PanoDiT-Image",
            weight_name="lora.safetensors",
            torch_dtype=torch.bfloat16
        )
        # save some VRAM by offloading the model to CPU
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()  # and enable vae tiling to save some VRAM

        # set general prompts
        self.general_negative_prompt = (
            "human, person, people, messy,"
            "low-quality, blur, noise, low-resolution"
        )
        self.general_positive_prompt = "high-quality,  high-resolution, sharp, clear, 8k"

    def run(self, prompt, negative_prompt, image_path, seed=42, output_path='output_panorama'):
        # preprocess prompt
        prompt = prompt + ", " + self.general_positive_prompt
        negative_prompt = self.general_negative_prompt + ", " + negative_prompt

        # read image
        perspective_img = cv2.imread(image_path)
        height_fov, width_fov = perspective_img.shape[:2]
        if width_fov > height_fov:
            ratio = width_fov / height_fov
            w = int((self.FOV / 360) * self.width)
            h = int(w / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ratio = height_fov / width_fov
            h = int((self.FOV / 180) * self.height)
            w = int(h / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)

        
        equ = Perspective(perspective_img, self.FOV,
                          self.THETA, self.PHI, crop_bound=False)
        img, mask = equ.GetEquirec(self.height, self.width)
        # erode mask
        mask = cv2.erode(mask.astype(np.uint8), np.ones(
            (3, 3), np.uint8), iterations=5)

        img = img * mask

        mask = mask.astype(np.uint8) * 255
        mask = 255 - mask

        mask = Image.fromarray(mask[:, :, 0])
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        image = self.pipe(
            prompt=prompt,
            image=img,
            mask_image=mask,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            blend_extend=self.blend_extend,
            shifting_extend=self.shifting_extend,
            true_cfg_scale=self.true_cfg_scale,
        ).images[0]

        image.save(os.path.join(output_path, 'panorama.png'))

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text/Image to Panorama Demo")
    parser.add_argument("--prompt", type=str,
                        default="", help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str,
                        default="", help="Negative prompt for image generation")
    parser.add_argument("--image_path", type=str,
                        default=None, help="Path to the input image")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_path", type=str, default="results",
                        help="Path to save the output results")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}")

    if args.image_path is None:
        print("No image path provided, using text-to-panorama generation.")
        demo_T2P = Text2PanoramaDemo()
        panorama_image = demo_T2P.run(
            args.prompt, args.negative_prompt, args.seed, args.output_path)
    else:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(
                f"Image path {args.image_path} does not exist.")
        print(f"Using image at {args.image_path} for panorama generation.")
        demo_I2P = Image2PanoramaDemo()
        panorama_image = demo_I2P.run(
            args.prompt, args.negative_prompt, args.image_path, args.seed, args.output_path)
