"""
API Templates for ComfyUI Worker

This module provides OpenAI-style API input translation to ComfyUI workflows
with support for:
- WAN (Wan2.1) video generation templates
- HuggingFace model caching
- LoRA and LyCORIS support
- Model/template selection

Usage:
    # Simple API request
    from api_templates import APIToComfyTranslator

    translator = APIToComfyTranslator()
    result = translator.translate({
        "prompt": "A beautiful sunset over the ocean",
        "duration": 5,
        "resolution": "720p",
    })
    workflow = result.workflow

    # Or use the convenience function
    from api_templates.translator import create_video_workflow

    result = create_video_workflow(
        prompt="A majestic eagle soaring",
        duration=5,
        resolution="720p",
        loras=[{"name": "cinematic", "weight": 0.7}],
    )
"""

from .schemas import (
    VideoGenerateRequest,
    ImageGenerateRequest,
    LoRAConfig,
    LyCORISConfig,
    ModelConfig,
    TemplateConfig,
    SamplerConfig,
)
from .translator import APIToComfyTranslator, translate_api_request, create_video_workflow
from .templates import TemplateRegistry, BaseTemplate
from .wan_templates import (
    WanImageToVideoTemplate,
    WanTextToVideoTemplate,
    WanFastVideoTemplate,
)
from .hf_cache import HuggingFaceModelCache, get_hf_cache, ensure_wan_models
from .lora import LoRALoader, LyCORISLoader, apply_lora_and_lycoris
from .utils import (
    list_templates,
    get_template_defaults,
    validate_video_request,
    debug_workflow,
    estimate_generation_time,
    create_simple_request,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Schemas
    "VideoGenerateRequest",
    "ImageGenerateRequest",
    "LoRAConfig",
    "LyCORISConfig",
    "ModelConfig",
    "TemplateConfig",
    "SamplerConfig",
    # Core
    "APIToComfyTranslator",
    "translate_api_request",
    "create_video_workflow",
    "TemplateRegistry",
    "BaseTemplate",
    # Templates
    "WanImageToVideoTemplate",
    "WanTextToVideoTemplate",
    "WanFastVideoTemplate",
    # Caching
    "HuggingFaceModelCache",
    "get_hf_cache",
    "ensure_wan_models",
    # LoRA
    "LoRALoader",
    "LyCORISLoader",
    "apply_lora_and_lycoris",
    # Utils
    "list_templates",
    "get_template_defaults",
    "validate_video_request",
    "debug_workflow",
    "estimate_generation_time",
    "create_simple_request",
]
