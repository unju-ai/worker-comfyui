"""
API to ComfyUI Translator

Translates OpenAI-style API requests into ComfyUI workflows
using registered templates.
"""

import logging
import base64
import urllib.request
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

from .schemas import (
    VideoGenerateRequest,
    ImageGenerateRequest,
    TemplateConfig,
    LoRAConfig,
    LyCORISConfig,
)
from .templates import TemplateRegistry, BaseTemplate
from .hf_cache import get_hf_cache, ensure_wan_models
from .lora import apply_lora_and_lycoris

logger = logging.getLogger(__name__)


# Model name to template mapping
MODEL_TEMPLATE_MAP = {
    # Video models
    "wan": "wan_t2v",
    "wan-t2v": "wan_t2v",
    "wan-i2v": "wan_i2v",
    "wan-fast": "wan_fast",
    "wan2.1": "wan_t2v",
    "wan2.1-t2v": "wan_t2v",
    "wan2.1-i2v": "wan_i2v",
    # Legacy/aliases
    "video": "wan_t2v",
    "video-fast": "wan_fast",
}


@dataclass
class TranslationResult:
    """Result of translating an API request to a workflow"""
    workflow: Dict[str, Any]
    images: Optional[List[Dict[str, str]]] = None  # Images to upload
    template_name: str = ""
    model_paths: Optional[Dict[str, str]] = None  # Resolved model paths
    warnings: Optional[List[str]] = None


class APIToComfyTranslator:
    """
    Translates OpenAI-style API requests to ComfyUI workflows

    Supports:
    - Video generation (VideoGenerateRequest)
    - Image generation (ImageGenerateRequest)
    - Template selection and customization
    - Model overrides
    - LoRA/LyCORIS injection
    """

    def __init__(self, auto_download_models: bool = True):
        """
        Initialize the translator

        Args:
            auto_download_models: Whether to auto-download missing models
        """
        self.auto_download = auto_download_models
        self._hf_cache = get_hf_cache()

    def translate(
        self,
        request: Union[VideoGenerateRequest, ImageGenerateRequest, Dict[str, Any]],
    ) -> TranslationResult:
        """
        Translate an API request to a ComfyUI workflow

        Args:
            request: API request (VideoGenerateRequest, ImageGenerateRequest, or dict)

        Returns:
            TranslationResult with workflow and metadata
        """
        # Convert dict to proper request type
        if isinstance(request, dict):
            request = self._parse_request(request)

        # Handle by request type
        if isinstance(request, VideoGenerateRequest):
            return self._translate_video_request(request)
        elif isinstance(request, ImageGenerateRequest):
            return self._translate_image_request(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

    def _parse_request(self, data: Dict[str, Any]) -> Union[VideoGenerateRequest, ImageGenerateRequest]:
        """Parse a dictionary into the appropriate request type"""
        # Detect request type based on fields
        if "duration" in data or "fps" in data or "source_image_url" in data:
            return VideoGenerateRequest.from_dict(data)
        elif "size" in data or "n" in data or data.get("response_format"):
            return ImageGenerateRequest.from_dict(data)
        else:
            # Default to video for prompts without clear type
            return VideoGenerateRequest.from_dict(data)

    def _translate_video_request(self, request: VideoGenerateRequest) -> TranslationResult:
        """Translate a video generation request"""
        warnings = []

        # Determine template to use
        template_name = self._get_template_name(request)
        template = TemplateRegistry.get(template_name)

        if not template:
            # Fall back to default
            logger.warning(f"Template {template_name} not found, using wan_t2v")
            template = TemplateRegistry.get("wan_t2v")
            template_name = "wan_t2v"
            warnings.append(f"Template {request.model or 'unknown'} not found, using wan_t2v")

        if not template:
            raise ValueError("No video templates available")

        # Ensure required models are available
        model_paths = {}
        if self.auto_download:
            mode = "i2v" if request.source_image_base64 or request.source_image_url else "t2v"
            model_paths = ensure_wan_models(request.resolution, mode)

        # Build workflow from template
        workflow = template.build_workflow(request, request.template)

        # Apply LoRA/LyCORIS if specified
        loras = request.loras or (request.template.loras if request.template else None)
        lycoris = request.lycoris or (request.template.lycoris if request.template else None)

        if loras or lycoris:
            # Get the model source node from template
            model_node = template.node_map.get("unet_loader", "1")
            workflow = apply_lora_and_lycoris(
                workflow,
                loras=loras,
                lycoris=lycoris,
                model_source_node=model_node,
            )

        # Handle input images
        images = None
        if request.source_image_base64:
            images = [{"name": "input_image.png", "image": request.source_image_base64}]
        elif request.source_image_url:
            # Download and encode image
            image_data = self._download_image(request.source_image_url)
            if image_data:
                images = [{"name": "input_image.png", "image": image_data}]
            else:
                warnings.append(f"Could not download image from {request.source_image_url}")

        return TranslationResult(
            workflow=workflow,
            images=images,
            template_name=template_name,
            model_paths=model_paths if model_paths else None,
            warnings=warnings if warnings else None,
        )

    def _translate_image_request(self, request: ImageGenerateRequest) -> TranslationResult:
        """Translate an image generation request"""
        warnings = []

        # For now, use video template for images (single frame)
        # This can be extended with dedicated image templates
        template_name = MODEL_TEMPLATE_MAP.get(request.model, "wan_t2v")
        template = TemplateRegistry.get(template_name)

        if not template:
            logger.warning(f"No template for model {request.model}")
            warnings.append(f"Model {request.model} not supported, using default")
            template = TemplateRegistry.get("wan_t2v")
            template_name = "wan_t2v"

        if not template:
            raise ValueError("No templates available")

        # Create a video request with single frame
        video_request = VideoGenerateRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=3,  # Minimum duration
            resolution="720p",
            seed=request.seed,
            model=request.model,
            template=request.template,
            loras=request.loras,
            lycoris=request.lycoris,
        )

        # Override to single frame for image output
        workflow = template.build_workflow(video_request, request.template)

        # Set to single frame
        if "6" in workflow and "length" in workflow["6"].get("inputs", {}):
            workflow["6"]["inputs"]["length"] = 1
        if "9" in workflow and "length" in workflow["9"].get("inputs", {}):
            workflow["9"]["inputs"]["length"] = 1

        # Apply dimensions from size
        width, height = request.get_dimensions()
        self._apply_dimensions(workflow, width, height)

        return TranslationResult(
            workflow=workflow,
            template_name=template_name,
            warnings=warnings if warnings else None,
        )

    def _get_template_name(self, request: VideoGenerateRequest) -> str:
        """Determine the template name for a request"""
        # Check for explicit template
        if request.template and request.template.template_name:
            return request.template.template_name

        # Check model name mapping
        if request.model:
            model_lower = request.model.lower()
            if model_lower in MODEL_TEMPLATE_MAP:
                return MODEL_TEMPLATE_MAP[model_lower]

        # Auto-detect based on request
        if request.source_image_base64 or request.source_image_url:
            return "wan_i2v"

        return "wan_t2v"

    def _apply_dimensions(self, workflow: Dict[str, Any], width: int, height: int):
        """Apply dimensions to workflow nodes"""
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            if "width" in inputs:
                inputs["width"] = width
            if "height" in inputs:
                inputs["height"] = height

    def _download_image(self, url: str) -> Optional[str]:
        """Download image from URL and return base64 encoded data"""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                image_data = response.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    @staticmethod
    def list_available_templates() -> List[Dict[str, str]]:
        """List all available templates"""
        return TemplateRegistry.list_templates()

    @staticmethod
    def get_template_info(name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a template"""
        return TemplateRegistry.get_template_info(name)

    @staticmethod
    def list_supported_models() -> List[str]:
        """List supported model names"""
        return list(MODEL_TEMPLATE_MAP.keys())


def translate_api_request(
    job_input: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, str]]], Optional[List[str]]]:
    """
    Convenience function to translate API input to ComfyUI workflow

    Args:
        job_input: Raw job input dictionary

    Returns:
        Tuple of (workflow, images_to_upload, warnings)
    """
    # Check if this is already a workflow (legacy format)
    if "workflow" in job_input:
        return job_input["workflow"], job_input.get("images"), None

    # Check if this is an API request format
    if "prompt" in job_input:
        translator = APIToComfyTranslator()
        result = translator.translate(job_input)
        return result.workflow, result.images, result.warnings

    # Unknown format
    raise ValueError("Input must contain either 'workflow' or 'prompt'")


def create_video_workflow(
    prompt: str,
    negative_prompt: Optional[str] = None,
    duration: int = 5,
    resolution: str = "720p",
    fps: int = 24,
    seed: Optional[int] = None,
    source_image: Optional[str] = None,
    model: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    lycoris: Optional[List[Dict[str, Any]]] = None,
) -> TranslationResult:
    """
    Convenience function to create a video generation workflow

    Args:
        prompt: Text prompt for video generation
        negative_prompt: Negative prompt
        duration: Video duration in seconds (3, 5, or 10)
        resolution: Resolution (480p, 720p, 1080p)
        fps: Frames per second
        seed: Random seed
        source_image: Base64 image for I2V, or None for T2V
        model: Model/template name
        loras: List of LoRA configs
        lycoris: List of LyCORIS configs

    Returns:
        TranslationResult with workflow
    """
    # Parse LoRA/LyCORIS configs
    lora_configs = None
    if loras:
        lora_configs = [
            LoRAConfig(**l) if isinstance(l, dict) else l
            for l in loras
        ]

    lycoris_configs = None
    if lycoris:
        lycoris_configs = [
            LyCORISConfig(**l) if isinstance(l, dict) else l
            for l in lycoris
        ]

    request = VideoGenerateRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_image_base64=source_image,
        duration=duration,
        resolution=resolution,
        fps=fps,
        seed=seed,
        model=model,
        loras=lora_configs,
        lycoris=lycoris_configs,
    )

    translator = APIToComfyTranslator()
    return translator.translate(request)
