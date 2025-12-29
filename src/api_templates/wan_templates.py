"""
WAN (Wan2.1) Video Generation Templates

Provides ComfyUI workflow templates for Wan2.1 video generation:
- Image-to-Video (I2V)
- Text-to-Video (T2V)

With support for different resolutions, durations, and quality settings.
"""

from typing import Dict, Any, Optional, List
import random
import logging

from .templates import BaseTemplate, TemplateRegistry
from .schemas import (
    VideoGenerateRequest,
    ModelConfig,
    SamplerConfig,
)

logger = logging.getLogger(__name__)


# Resolution mappings for WAN models
WAN_RESOLUTIONS = {
    "480p": {"width": 848, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
}

# Frame count based on duration (at 24fps)
WAN_FRAME_COUNTS = {
    3: 81,   # ~3 seconds at 24fps
    5: 121,  # ~5 seconds at 24fps
    10: 241, # ~10 seconds at 24fps
}

# Default WAN model configurations
WAN_MODELS = {
    "wan2.1-i2v-480p": {
        "unet": "wan2.1_i2v_480p_bf16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "clip_vision": "clip-vit-large-patch14.safetensors",
        "hf_repos": {
            "unet": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip_vision": "openai/clip-vit-large-patch14",
        }
    },
    "wan2.1-i2v-720p": {
        "unet": "wan2.1_i2v_720p_bf16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "clip_vision": "clip-vit-large-patch14.safetensors",
        "hf_repos": {
            "unet": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip_vision": "openai/clip-vit-large-patch14",
        }
    },
    "wan2.1-t2v-480p": {
        "unet": "wan2.1_t2v_480p_bf16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "hf_repos": {
            "unet": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        }
    },
    "wan2.1-t2v-720p": {
        "unet": "wan2.1_t2v_720p_bf16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "hf_repos": {
            "unet": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        }
    },
    "wan2.1-t2v-1.3b": {
        "unet": "wan2.1_t2v_1.3B_bf16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "hf_repos": {
            "unet": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "clip": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "vae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        }
    },
}


@TemplateRegistry.register
class WanImageToVideoTemplate(BaseTemplate):
    """
    WAN 2.1 Image-to-Video Template

    Generates video from a source image using Wan2.1 I2V models.
    Supports 480p and 720p resolutions with various durations.
    """

    name = "wan_i2v"
    description = "WAN 2.1 Image-to-Video generation"
    version = "1.0.0"

    default_models = ModelConfig(
        unet="wan2.1_i2v_720p_bf16.safetensors",
        clip="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        clip_vision="clip-vit-large-patch14.safetensors",
        hf_cache_enabled=True,
    )

    default_sampler = SamplerConfig(
        sampler_name="euler",
        scheduler="normal",
        steps=30,
        cfg_scale=6.0,
        denoise=1.0,
    )

    # Map logical names to node IDs
    node_map = {
        "unet_loader": "1",
        "clip_loader": "2",
        "vae_loader": "3",
        "clip_vision_loader": "4",
        "positive_prompt": "5",
        "negative_prompt": "6",
        "image_loader": "7",
        "clip_vision_encode": "8",
        "empty_latent": "9",
        "sampler": "10",
        "scheduler": "11",
        "guider": "12",
        "noise": "13",
        "sampler_custom": "14",
        "vae_decode": "15",
        "video_save": "16",
    }

    def get_base_workflow(self) -> Dict[str, Any]:
        """Return the base WAN I2V workflow"""
        return {
            # UNet Loader (WAN diffusion model)
            "1": {
                "inputs": {
                    "unet_name": self.default_models.unet,
                    "weight_dtype": "bf16"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load WAN UNet"}
            },
            # CLIP Text Encoder (UMT5)
            "2": {
                "inputs": {
                    "clip_name": self.default_models.clip,
                    "type": "wan"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load CLIP (UMT5)"}
            },
            # VAE Loader
            "3": {
                "inputs": {
                    "vae_name": self.default_models.vae
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Load VAE"}
            },
            # CLIP Vision Loader (for image conditioning)
            "4": {
                "inputs": {
                    "clip_name": self.default_models.clip_vision
                },
                "class_type": "CLIPVisionLoader",
                "_meta": {"title": "Load CLIP Vision"}
            },
            # Positive Prompt Encoding
            "5": {
                "inputs": {
                    "text": "",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            # Negative Prompt Encoding
            "6": {
                "inputs": {
                    "text": "",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            # Load Input Image
            "7": {
                "inputs": {
                    "image": "input_image.png",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Source Image"}
            },
            # CLIP Vision Encode
            "8": {
                "inputs": {
                    "clip_vision": ["4", 0],
                    "image": ["7", 0]
                },
                "class_type": "CLIPVisionEncode",
                "_meta": {"title": "CLIP Vision Encode"}
            },
            # Empty Latent Video (creates the video latent space)
            "9": {
                "inputs": {
                    "width": 1280,
                    "height": 720,
                    "length": 121,
                    "batch_size": 1
                },
                "class_type": "EmptyWanLatentVideo",
                "_meta": {"title": "Empty Latent Video"}
            },
            # Sampler Settings
            "10": {
                "inputs": {
                    "sampler_name": self.default_sampler.sampler_name
                },
                "class_type": "KSamplerSelect",
                "_meta": {"title": "Sampler"}
            },
            # Scheduler
            "11": {
                "inputs": {
                    "scheduler": self.default_sampler.scheduler,
                    "steps": self.default_sampler.steps,
                    "denoise": self.default_sampler.denoise,
                    "model": ["1", 0]
                },
                "class_type": "BasicScheduler",
                "_meta": {"title": "Scheduler"}
            },
            # Guider with CFG
            "12": {
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "cfg": self.default_sampler.cfg_scale
                },
                "class_type": "CFGGuider",
                "_meta": {"title": "CFG Guider"}
            },
            # Random Noise
            "13": {
                "inputs": {
                    "noise_seed": 0
                },
                "class_type": "RandomNoise",
                "_meta": {"title": "Random Noise"}
            },
            # WAN I2V Conditioning (combines image and text)
            "17": {
                "inputs": {
                    "clip_vision_output": ["8", 0],
                    "conditioning": ["5", 0],
                    "vae": ["3", 0],
                    "image": ["7", 0],
                    "width": 1280,
                    "height": 720,
                    "length": 121
                },
                "class_type": "WanImageToVideoConditioning",
                "_meta": {"title": "WAN I2V Conditioning"}
            },
            # Updated Guider to use I2V conditioning
            "18": {
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["17", 0],
                    "negative": ["6", 0],
                    "cfg": self.default_sampler.cfg_scale
                },
                "class_type": "CFGGuider",
                "_meta": {"title": "CFG Guider (I2V)"}
            },
            # Custom Sampler
            "14": {
                "inputs": {
                    "noise": ["13", 0],
                    "guider": ["18", 0],
                    "sampler": ["10", 0],
                    "sigmas": ["11", 0],
                    "latent_image": ["9", 0]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "Sample"}
            },
            # VAE Decode to Video
            "15": {
                "inputs": {
                    "samples": ["14", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            # Save Video
            "16": {
                "inputs": {
                    "filename_prefix": "wan_i2v",
                    "fps": 24,
                    "images": ["15", 0]
                },
                "class_type": "SaveAnimatedWEBP",
                "_meta": {"title": "Save Video"}
            }
        }

    def _apply_request_params(
        self, workflow: Dict[str, Any], request: VideoGenerateRequest
    ) -> Dict[str, Any]:
        """Apply video generation request parameters"""
        # Set prompts
        workflow["5"]["inputs"]["text"] = request.prompt
        workflow["6"]["inputs"]["text"] = request.negative_prompt or ""

        # Set resolution
        res = WAN_RESOLUTIONS.get(request.resolution, WAN_RESOLUTIONS["720p"])
        width, height = res["width"], res["height"]

        workflow["9"]["inputs"]["width"] = width
        workflow["9"]["inputs"]["height"] = height
        workflow["17"]["inputs"]["width"] = width
        workflow["17"]["inputs"]["height"] = height

        # Set frame count based on duration
        frame_count = WAN_FRAME_COUNTS.get(request.duration, WAN_FRAME_COUNTS[5])
        workflow["9"]["inputs"]["length"] = frame_count
        workflow["17"]["inputs"]["length"] = frame_count

        # Set FPS
        workflow["16"]["inputs"]["fps"] = request.fps

        # Set seed
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow["13"]["inputs"]["noise_seed"] = seed

        # Select appropriate model for resolution
        model_key = f"wan2.1-i2v-{request.resolution}"
        if model_key in WAN_MODELS:
            model_config = WAN_MODELS[model_key]
            workflow["1"]["inputs"]["unet_name"] = model_config["unet"]

        # Handle source image
        if request.source_image_base64:
            # Image will be uploaded separately, just set the reference name
            workflow["7"]["inputs"]["image"] = "input_image.png"

        return workflow

    def get_required_models(self) -> Dict[str, str]:
        """Return required models with HuggingFace repos"""
        return {
            "unet": "wan2.1_i2v_720p_bf16.safetensors",
            "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae": "wan_2.1_vae.safetensors",
            "clip_vision": "clip-vit-large-patch14.safetensors",
        }

    def get_required_nodes(self) -> List[str]:
        """Return required custom nodes"""
        return [
            "ComfyUI-WAN",  # For WAN-specific nodes
        ]


@TemplateRegistry.register
class WanTextToVideoTemplate(BaseTemplate):
    """
    WAN 2.1 Text-to-Video Template

    Generates video from text prompt using Wan2.1 T2V models.
    Supports 480p, 720p resolutions and 1.3B model variant.
    """

    name = "wan_t2v"
    description = "WAN 2.1 Text-to-Video generation"
    version = "1.0.0"

    default_models = ModelConfig(
        unet="wan2.1_t2v_720p_bf16.safetensors",
        clip="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        hf_cache_enabled=True,
    )

    default_sampler = SamplerConfig(
        sampler_name="euler",
        scheduler="normal",
        steps=30,
        cfg_scale=6.0,
        denoise=1.0,
    )

    node_map = {
        "unet_loader": "1",
        "clip_loader": "2",
        "vae_loader": "3",
        "positive_prompt": "4",
        "negative_prompt": "5",
        "empty_latent": "6",
        "sampler": "7",
        "scheduler": "8",
        "guider": "9",
        "noise": "10",
        "sampler_custom": "11",
        "vae_decode": "12",
        "video_save": "13",
    }

    def get_base_workflow(self) -> Dict[str, Any]:
        """Return the base WAN T2V workflow"""
        return {
            # UNet Loader (WAN diffusion model)
            "1": {
                "inputs": {
                    "unet_name": self.default_models.unet,
                    "weight_dtype": "bf16"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load WAN UNet"}
            },
            # CLIP Text Encoder (UMT5)
            "2": {
                "inputs": {
                    "clip_name": self.default_models.clip,
                    "type": "wan"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load CLIP (UMT5)"}
            },
            # VAE Loader
            "3": {
                "inputs": {
                    "vae_name": self.default_models.vae
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Load VAE"}
            },
            # Positive Prompt Encoding
            "4": {
                "inputs": {
                    "text": "",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            # Negative Prompt Encoding
            "5": {
                "inputs": {
                    "text": "blurry, distorted, low quality, watermark",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            # Empty Latent Video
            "6": {
                "inputs": {
                    "width": 1280,
                    "height": 720,
                    "length": 121,
                    "batch_size": 1
                },
                "class_type": "EmptyWanLatentVideo",
                "_meta": {"title": "Empty Latent Video"}
            },
            # Sampler Settings
            "7": {
                "inputs": {
                    "sampler_name": self.default_sampler.sampler_name
                },
                "class_type": "KSamplerSelect",
                "_meta": {"title": "Sampler"}
            },
            # Scheduler
            "8": {
                "inputs": {
                    "scheduler": self.default_sampler.scheduler,
                    "steps": self.default_sampler.steps,
                    "denoise": self.default_sampler.denoise,
                    "model": ["1", 0]
                },
                "class_type": "BasicScheduler",
                "_meta": {"title": "Scheduler"}
            },
            # CFG Guider
            "9": {
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "cfg": self.default_sampler.cfg_scale
                },
                "class_type": "CFGGuider",
                "_meta": {"title": "CFG Guider"}
            },
            # Random Noise
            "10": {
                "inputs": {
                    "noise_seed": 0
                },
                "class_type": "RandomNoise",
                "_meta": {"title": "Random Noise"}
            },
            # Custom Sampler
            "11": {
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["9", 0],
                    "sampler": ["7", 0],
                    "sigmas": ["8", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "Sample"}
            },
            # VAE Decode to Video
            "12": {
                "inputs": {
                    "samples": ["11", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            # Save Video
            "13": {
                "inputs": {
                    "filename_prefix": "wan_t2v",
                    "fps": 24,
                    "images": ["12", 0]
                },
                "class_type": "SaveAnimatedWEBP",
                "_meta": {"title": "Save Video"}
            }
        }

    def _apply_request_params(
        self, workflow: Dict[str, Any], request: VideoGenerateRequest
    ) -> Dict[str, Any]:
        """Apply video generation request parameters"""
        # Set prompts
        workflow["4"]["inputs"]["text"] = request.prompt
        if request.negative_prompt:
            workflow["5"]["inputs"]["text"] = request.negative_prompt

        # Set resolution
        res = WAN_RESOLUTIONS.get(request.resolution, WAN_RESOLUTIONS["720p"])
        width, height = res["width"], res["height"]

        workflow["6"]["inputs"]["width"] = width
        workflow["6"]["inputs"]["height"] = height

        # Set frame count based on duration
        frame_count = WAN_FRAME_COUNTS.get(request.duration, WAN_FRAME_COUNTS[5])
        workflow["6"]["inputs"]["length"] = frame_count

        # Set FPS
        workflow["13"]["inputs"]["fps"] = request.fps

        # Set seed
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow["10"]["inputs"]["noise_seed"] = seed

        # Select appropriate model for resolution
        model_key = f"wan2.1-t2v-{request.resolution}"
        if model_key in WAN_MODELS:
            model_config = WAN_MODELS[model_key]
            workflow["1"]["inputs"]["unet_name"] = model_config["unet"]

        return workflow

    def get_required_models(self) -> Dict[str, str]:
        """Return required models"""
        return {
            "unet": "wan2.1_t2v_720p_bf16.safetensors",
            "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae": "wan_2.1_vae.safetensors",
        }

    def get_required_nodes(self) -> List[str]:
        """Return required custom nodes"""
        return [
            "ComfyUI-WAN",
        ]


@TemplateRegistry.register
class WanFastVideoTemplate(BaseTemplate):
    """
    WAN 2.1 Fast Video Template

    Optimized for faster generation with reduced steps.
    Uses the 1.3B T2V model for better speed/quality tradeoff.
    """

    name = "wan_fast"
    description = "WAN 2.1 Fast Video (1.3B model, fewer steps)"
    version = "1.0.0"

    default_models = ModelConfig(
        unet="wan2.1_t2v_1.3B_bf16.safetensors",
        clip="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        hf_cache_enabled=True,
    )

    default_sampler = SamplerConfig(
        sampler_name="euler",
        scheduler="normal",
        steps=20,  # Fewer steps for speed
        cfg_scale=5.0,
        denoise=1.0,
    )

    node_map = {
        "unet_loader": "1",
        "clip_loader": "2",
        "vae_loader": "3",
        "positive_prompt": "4",
        "negative_prompt": "5",
        "empty_latent": "6",
        "sampler": "7",
        "scheduler": "8",
        "guider": "9",
        "noise": "10",
        "sampler_custom": "11",
        "vae_decode": "12",
        "video_save": "13",
    }

    def get_base_workflow(self) -> Dict[str, Any]:
        """Return the base fast video workflow"""
        # Similar to T2V but with optimized settings
        return {
            "1": {
                "inputs": {
                    "unet_name": self.default_models.unet,
                    "weight_dtype": "bf16"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load WAN UNet (1.3B)"}
            },
            "2": {
                "inputs": {
                    "clip_name": self.default_models.clip,
                    "type": "wan"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load CLIP (UMT5)"}
            },
            "3": {
                "inputs": {
                    "vae_name": self.default_models.vae
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Load VAE"}
            },
            "4": {
                "inputs": {
                    "text": "",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "5": {
                "inputs": {
                    "text": "blurry, distorted, low quality",
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "6": {
                "inputs": {
                    "width": 848,
                    "height": 480,
                    "length": 81,  # Shorter for speed
                    "batch_size": 1
                },
                "class_type": "EmptyWanLatentVideo",
                "_meta": {"title": "Empty Latent Video"}
            },
            "7": {
                "inputs": {
                    "sampler_name": self.default_sampler.sampler_name
                },
                "class_type": "KSamplerSelect",
                "_meta": {"title": "Sampler"}
            },
            "8": {
                "inputs": {
                    "scheduler": self.default_sampler.scheduler,
                    "steps": self.default_sampler.steps,
                    "denoise": self.default_sampler.denoise,
                    "model": ["1", 0]
                },
                "class_type": "BasicScheduler",
                "_meta": {"title": "Scheduler"}
            },
            "9": {
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "cfg": self.default_sampler.cfg_scale
                },
                "class_type": "CFGGuider",
                "_meta": {"title": "CFG Guider"}
            },
            "10": {
                "inputs": {
                    "noise_seed": 0
                },
                "class_type": "RandomNoise",
                "_meta": {"title": "Random Noise"}
            },
            "11": {
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["9", 0],
                    "sampler": ["7", 0],
                    "sigmas": ["8", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "Sample"}
            },
            "12": {
                "inputs": {
                    "samples": ["11", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "13": {
                "inputs": {
                    "filename_prefix": "wan_fast",
                    "fps": 24,
                    "images": ["12", 0]
                },
                "class_type": "SaveAnimatedWEBP",
                "_meta": {"title": "Save Video"}
            }
        }

    def _apply_request_params(
        self, workflow: Dict[str, Any], request: VideoGenerateRequest
    ) -> Dict[str, Any]:
        """Apply video generation request parameters"""
        workflow["4"]["inputs"]["text"] = request.prompt
        if request.negative_prompt:
            workflow["5"]["inputs"]["text"] = request.negative_prompt

        # Fast mode uses fixed 480p for speed
        res = WAN_RESOLUTIONS["480p"]
        workflow["6"]["inputs"]["width"] = res["width"]
        workflow["6"]["inputs"]["height"] = res["height"]

        # Shorter duration for speed
        frame_count = min(WAN_FRAME_COUNTS.get(request.duration, 81), 121)
        workflow["6"]["inputs"]["length"] = frame_count

        workflow["13"]["inputs"]["fps"] = request.fps

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow["10"]["inputs"]["noise_seed"] = seed

        return workflow

    def get_required_models(self) -> Dict[str, str]:
        return {
            "unet": "wan2.1_t2v_1.3B_bf16.safetensors",
            "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae": "wan_2.1_vae.safetensors",
        }

    def get_required_nodes(self) -> List[str]:
        return ["ComfyUI-WAN"]
