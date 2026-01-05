"""
Pydantic schemas for OpenAI-style API input

These schemas translate OpenAI-compatible API requests into
structured data that can be used to generate ComfyUI workflows.
"""

from typing import Optional, List, Literal, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class VideoResolution(str, Enum):
    """Supported video resolutions"""
    RES_480P = "480p"
    RES_720P = "720p"
    RES_1080P = "1080p"


class VideoDuration(int, Enum):
    """Supported video durations in seconds"""
    SHORT = 3
    MEDIUM = 5
    LONG = 10


class ImageSize(str, Enum):
    """Supported image sizes"""
    SIZE_256 = "256x256"
    SIZE_512 = "512x512"
    SIZE_1024 = "1024x1024"
    SIZE_1792_1024 = "1792x1024"
    SIZE_1024_1792 = "1024x1792"


class ImageQuality(str, Enum):
    """Image quality levels"""
    STANDARD = "standard"
    HD = "hd"


class ImageStyle(str, Enum):
    """Image style options"""
    VIVID = "vivid"
    NATURAL = "natural"


class ResponseFormat(str, Enum):
    """Response format options"""
    URL = "url"
    B64_JSON = "b64_json"


@dataclass
class LoRAConfig:
    """Configuration for a LoRA model"""
    name: str
    weight: float = 1.0
    clip_weight: Optional[float] = None  # If None, uses weight
    # HuggingFace source for auto-download
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "clip_weight": self.clip_weight if self.clip_weight is not None else self.weight,
            "hf_repo": self.hf_repo,
            "hf_filename": self.hf_filename,
        }


@dataclass
class LyCORISConfig:
    """Configuration for a LyCORIS model (LoHa, LoKr, etc.)"""
    name: str
    weight: float = 1.0
    # LyCORIS-specific parameters
    mode: Literal["full", "merge", "auto"] = "auto"
    # HuggingFace source
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "mode": self.mode,
            "hf_repo": self.hf_repo,
            "hf_filename": self.hf_filename,
        }


@dataclass
class ModelConfig:
    """Configuration for model selection"""
    # Main model checkpoint
    checkpoint: Optional[str] = None
    # UNet model (for architectures like FLUX, WAN)
    unet: Optional[str] = None
    # VAE model
    vae: Optional[str] = None
    # CLIP models
    clip: Optional[str] = None
    clip_vision: Optional[str] = None
    # Text encoders
    t5: Optional[str] = None
    # HuggingFace caching config
    hf_cache_enabled: bool = True
    hf_repos: Optional[Dict[str, str]] = None  # model_type -> hf_repo mapping

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "unet": self.unet,
            "vae": self.vae,
            "clip": self.clip,
            "clip_vision": self.clip_vision,
            "t5": self.t5,
            "hf_cache_enabled": self.hf_cache_enabled,
            "hf_repos": self.hf_repos,
        }


@dataclass
class SamplerConfig:
    """Configuration for sampler settings"""
    sampler_name: str = "euler"
    scheduler: str = "normal"
    steps: int = 20
    cfg_scale: float = 7.0
    denoise: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoise": self.denoise,
        }


@dataclass
class TemplateConfig:
    """Configuration for template selection and overrides"""
    template_name: str
    # Override default model settings
    model: Optional[ModelConfig] = None
    # Override sampler settings
    sampler: Optional[SamplerConfig] = None
    # LoRA configurations
    loras: Optional[List[LoRAConfig]] = None
    # LyCORIS configurations
    lycoris: Optional[List[LyCORISConfig]] = None
    # Custom node overrides (node_id -> {param: value})
    node_overrides: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "template_name": self.template_name,
            "node_overrides": self.node_overrides,
        }
        if self.model:
            result["model"] = self.model.to_dict()
        if self.sampler:
            result["sampler"] = self.sampler.to_dict()
        if self.loras:
            result["loras"] = [l.to_dict() for l in self.loras]
        if self.lycoris:
            result["lycoris"] = [l.to_dict() for l in self.lycoris]
        return result


@dataclass
class VideoGenerateRequest:
    """
    OpenAI-style video generation request

    Compatible with /v1/videos/generations endpoint
    """
    prompt: str
    negative_prompt: Optional[str] = None
    # Source image for image-to-video
    source_image_url: Optional[str] = None
    source_image_base64: Optional[str] = None
    # Video settings
    duration: int = 5  # seconds: 3, 5, or 10
    resolution: str = "720p"  # 480p, 720p, 1080p
    fps: int = 24
    # Generation settings
    seed: Optional[int] = None
    # Model/template selection
    model: Optional[str] = None  # Model name or template ID
    template: Optional[TemplateConfig] = None
    # Optional LoRA/LyCORIS
    loras: Optional[List[LoRAConfig]] = None
    lycoris: Optional[List[LyCORISConfig]] = None
    # Webhook for async processing
    webhook_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoGenerateRequest":
        """Create from dictionary (API input)"""
        # Parse nested configs
        template = None
        if "template" in data and data["template"]:
            template_data = data["template"]
            model_config = None
            if "model" in template_data and template_data["model"]:
                model_config = ModelConfig(**template_data["model"])
            sampler_config = None
            if "sampler" in template_data and template_data["sampler"]:
                sampler_config = SamplerConfig(**template_data["sampler"])
            loras = None
            if "loras" in template_data and template_data["loras"]:
                loras = [LoRAConfig(**l) for l in template_data["loras"]]
            lycoris_list = None
            if "lycoris" in template_data and template_data["lycoris"]:
                lycoris_list = [LyCORISConfig(**l) for l in template_data["lycoris"]]
            template = TemplateConfig(
                template_name=template_data.get("template_name", "wan_i2v"),
                model=model_config,
                sampler=sampler_config,
                loras=loras,
                lycoris=lycoris_list,
                node_overrides=template_data.get("node_overrides"),
            )

        # Parse top-level LoRAs
        loras = None
        if "loras" in data and data["loras"]:
            loras = [LoRAConfig(**l) if isinstance(l, dict) else l for l in data["loras"]]

        lycoris_list = None
        if "lycoris" in data and data["lycoris"]:
            lycoris_list = [LyCORISConfig(**l) if isinstance(l, dict) else l for l in data["lycoris"]]

        return cls(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt"),
            source_image_url=data.get("source_image_url"),
            source_image_base64=data.get("source_image_base64"),
            duration=data.get("duration", 5),
            resolution=data.get("resolution", "720p"),
            fps=data.get("fps", 24),
            seed=data.get("seed"),
            model=data.get("model"),
            template=template,
            loras=loras,
            lycoris=lycoris_list,
            webhook_url=data.get("webhook_url"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "source_image_url": self.source_image_url,
            "source_image_base64": self.source_image_base64,
            "duration": self.duration,
            "resolution": self.resolution,
            "fps": self.fps,
            "seed": self.seed,
            "model": self.model,
            "webhook_url": self.webhook_url,
        }
        if self.template:
            result["template"] = self.template.to_dict()
        if self.loras:
            result["loras"] = [l.to_dict() for l in self.loras]
        if self.lycoris:
            result["lycoris"] = [l.to_dict() for l in self.lycoris]
        return result


@dataclass
class ImageGenerateRequest:
    """
    OpenAI-style image generation request

    Compatible with /v1/images/generations endpoint (DALL-E style)
    """
    prompt: str
    model: str = "wan"  # Model name
    n: int = 1  # Number of images (1-10)
    quality: str = "standard"  # standard or hd
    response_format: str = "url"  # url or b64_json
    size: str = "1024x1024"
    style: str = "vivid"  # vivid or natural
    # Extended parameters
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # Template configuration
    template: Optional[TemplateConfig] = None
    # LoRA/LyCORIS
    loras: Optional[List[LoRAConfig]] = None
    lycoris: Optional[List[LyCORISConfig]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageGenerateRequest":
        """Create from dictionary (API input)"""
        # Parse template config
        template = None
        if "template" in data and data["template"]:
            template_data = data["template"]
            model_config = None
            if "model" in template_data and template_data["model"]:
                model_config = ModelConfig(**template_data["model"])
            sampler_config = None
            if "sampler" in template_data and template_data["sampler"]:
                sampler_config = SamplerConfig(**template_data["sampler"])
            template = TemplateConfig(
                template_name=template_data.get("template_name", "default"),
                model=model_config,
                sampler=sampler_config,
                node_overrides=template_data.get("node_overrides"),
            )

        # Parse LoRAs
        loras = None
        if "loras" in data and data["loras"]:
            loras = [LoRAConfig(**l) if isinstance(l, dict) else l for l in data["loras"]]

        lycoris_list = None
        if "lycoris" in data and data["lycoris"]:
            lycoris_list = [LyCORISConfig(**l) if isinstance(l, dict) else l for l in data["lycoris"]]

        return cls(
            prompt=data["prompt"],
            model=data.get("model", "wan"),
            n=data.get("n", 1),
            quality=data.get("quality", "standard"),
            response_format=data.get("response_format", "url"),
            size=data.get("size", "1024x1024"),
            style=data.get("style", "vivid"),
            negative_prompt=data.get("negative_prompt"),
            seed=data.get("seed"),
            template=template,
            loras=loras,
            lycoris=lycoris_list,
        )

    def get_dimensions(self) -> tuple:
        """Parse size string to width, height tuple"""
        try:
            parts = self.size.lower().split("x")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 1024, 1024

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "prompt": self.prompt,
            "model": self.model,
            "n": self.n,
            "quality": self.quality,
            "response_format": self.response_format,
            "size": self.size,
            "style": self.style,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }
        if self.template:
            result["template"] = self.template.to_dict()
        if self.loras:
            result["loras"] = [l.to_dict() for l in self.loras]
        if self.lycoris:
            result["lycoris"] = [l.to_dict() for l in self.lycoris]
        return result
