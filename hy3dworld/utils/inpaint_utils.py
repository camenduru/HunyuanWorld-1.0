import torch
import numpy as np
import cv2
import math
from ..models import FluxFillPipeline

from diffusers.quantizers import PipelineQuantizationConfig


def get_smooth_mask(general_mask, ksize=(120, 120)):
    r"""Generate a smooth mask from the general mask using morphological dilation.
    Args:
        general_mask (np.ndarray): The input mask to be smoothed, expected to be a binary mask
            with shape [H, W] and dtype uint8 (0 or 1).
        ksize (tuple): The size of the structuring element used for dilation, specified as
            (height, width). Default is (120, 120).
    Returns:
        np.ndarray: The smoothed mask, with the same shape as the input mask, where
            the values are either 0 or 1 (uint8).
    """
    # Ensure kernel size is a tuple of integers
    ksize = (int(ksize[0]), int(ksize[1]))  
    
    # Create rectangular structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    
    # Apply dilation to expand mask regions
    mask_array = cv2.dilate(general_mask.astype(
        np.uint8), kernel)  # [1024, 2048] uint8 1
    
    # Convert back to binary mask
    mask_array = (mask_array > 0).astype(np.uint8)
    
    return mask_array


def build_inpaint_model(model_path, lora_path, subfolder, device=0):
    r"""Build the inpainting model pipeline.
    Args:
        model_path (str): The path to the pre-trained model.
        lora_path (str): The path to the LoRA weights.
        device (int): The device ID to load the model onto (default: 0).
    Returns:
        pipe: The inpainting pipeline object.
    """
    # Initialize pipeline with bfloat16 precision for memory efficiency

    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
        components_to_quantize=["transformer"]
    )

    pipe = FluxFillPipeline.from_pretrained(
        model_path, quantization_config=quant_config, torch_dtype=torch.bfloat16).to(f"cuda:{device}")
    pipe.load_lora_weights(
        lora_path,
        subfolder=subfolder,
        weight_name="lora.safetensors",  # default weight name
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
    pipe.device_id = device
    return pipe


def get_adaptive_smooth_mask_ksize_ctrl(general_masks, mask_infos, basek=100, threshold=10000, r=1):
    r"""Generate a smooth mask with adaptive kernel size control based on mask area.
    Args:
        general_masks (np.ndarray): The input mask array, expected to be a 2D array of shape [H, W]
            where each pixel value corresponds to a mask ID.
        mask_infos (list): A list of dictionaries containing information about each mask, including
            the area and label of the mask.
        basek (int): The base kernel size for smoothing, default is 100.
        threshold (int): The area threshold to determine the scaling factor for the kernel size,
            default is 10000.
        r (int): A scaling factor for the kernel size, default is 1.
    Returns:
        np.ndarray: The smoothed mask array, with the same shape as the input mask,
            where the values are either 0 or 1 (uint8).
    """
    # Initialize output mask
    mask_array = np.zeros_like(general_masks).astype(np.bool_)

    # Process each mask region individually
    for i in range(len(mask_infos)):
        mask_info = mask_infos[i]
        area = mask_info["area"]

        # Calculate size ratio with threshold clamping
        ratio = area / threshold
        ratio = math.sqrt(min(ratio, 1.0))

        # Extract current object mask
        mask = (general_masks == i + 1).astype(np.uint8)
        
        # Default kernel for other objects
        mask = get_smooth_mask(mask, ksize=(
            int(basek*ratio)*r, int((basek+10)*ratio)*r)).astype(np.bool_)
        
        # Combine with existing masks
        mask_array = np.logical_or(mask_array, mask)
    
    return mask_array.astype(np.uint8)
