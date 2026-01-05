"""
Utility functions for API Templates

Provides helper functions for common operations like
listing templates, validating requests, and debugging workflows.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .templates import TemplateRegistry
from .schemas import VideoGenerateRequest, ImageGenerateRequest

logger = logging.getLogger(__name__)


def list_templates() -> List[Dict[str, str]]:
    """
    List all available workflow templates

    Returns:
        List of template info dicts with name, description, version
    """
    return TemplateRegistry.list_templates()


def get_template_defaults(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Get default configuration for a template

    Args:
        template_name: Name of the template

    Returns:
        Dict with default models, sampler settings, and node map
    """
    return TemplateRegistry.get_template_info(template_name)


def validate_video_request(request_data: Dict[str, Any]) -> tuple:
    """
    Validate a video generation request

    Args:
        request_data: Request dictionary

    Returns:
        Tuple of (is_valid, errors_list)
    """
    errors = []

    # Check required fields
    if "prompt" not in request_data:
        errors.append("Missing required field: prompt")

    # Validate prompt length
    prompt = request_data.get("prompt", "")
    if len(prompt) < 1:
        errors.append("Prompt cannot be empty")
    if len(prompt) > 1000:
        errors.append("Prompt exceeds maximum length of 1000 characters")

    # Validate negative prompt if present
    neg_prompt = request_data.get("negative_prompt", "")
    if len(neg_prompt) > 500:
        errors.append("Negative prompt exceeds maximum length of 500 characters")

    # Validate duration
    duration = request_data.get("duration", 5)
    if duration not in [3, 5, 10]:
        errors.append(f"Invalid duration: {duration}. Must be 3, 5, or 10 seconds")

    # Validate resolution
    resolution = request_data.get("resolution", "720p")
    if resolution not in ["480p", "720p", "1080p"]:
        errors.append(f"Invalid resolution: {resolution}. Must be 480p, 720p, or 1080p")

    # Validate FPS
    fps = request_data.get("fps", 24)
    if fps < 1 or fps > 60:
        errors.append(f"Invalid FPS: {fps}. Must be between 1 and 60")

    # Validate seed if present
    seed = request_data.get("seed")
    if seed is not None and (seed < 0 or seed > 2**32 - 1):
        errors.append(f"Invalid seed: {seed}. Must be between 0 and 2^32-1")

    # Validate LoRA configs
    loras = request_data.get("loras", [])
    for i, lora in enumerate(loras):
        if "name" not in lora:
            errors.append(f"LoRA at index {i} missing required field: name")
        weight = lora.get("weight", 1.0)
        if weight < -2.0 or weight > 2.0:
            errors.append(f"LoRA '{lora.get('name', i)}' weight {weight} out of range [-2.0, 2.0]")

    return len(errors) == 0, errors


def debug_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a workflow and return debug information

    Args:
        workflow: ComfyUI workflow dictionary

    Returns:
        Debug info including node count, connections, required models
    """
    nodes = []
    connections = []
    required_models = set()
    node_types = {}

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "Unknown")
        title = node.get("_meta", {}).get("title", class_type)

        nodes.append({
            "id": node_id,
            "type": class_type,
            "title": title,
        })

        # Count node types
        node_types[class_type] = node_types.get(class_type, 0) + 1

        # Find connections
        inputs = node.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                connections.append({
                    "from_node": input_value[0],
                    "from_output": input_value[1],
                    "to_node": node_id,
                    "to_input": input_name,
                })

            # Check for model references
            if input_name in ["ckpt_name", "unet_name", "vae_name", "lora_name", "clip_name"]:
                if isinstance(input_value, str):
                    required_models.add(input_value)

    return {
        "node_count": len(nodes),
        "connection_count": len(connections),
        "nodes": nodes,
        "connections": connections,
        "node_types": node_types,
        "required_models": list(required_models),
    }


def workflow_to_mermaid(workflow: Dict[str, Any]) -> str:
    """
    Convert a workflow to Mermaid diagram format

    Args:
        workflow: ComfyUI workflow dictionary

    Returns:
        Mermaid diagram string
    """
    lines = ["graph TD"]

    # Add nodes
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "Unknown")
        title = node.get("_meta", {}).get("title", class_type)
        # Escape special characters
        safe_title = title.replace('"', "'")
        lines.append(f'    {node_id}["{safe_title}"]')

    # Add connections
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                from_node = input_value[0]
                lines.append(f"    {from_node} --> {node_id}")

    return "\n".join(lines)


def estimate_generation_time(
    duration: int = 5,
    resolution: str = "720p",
    steps: int = 30,
) -> Dict[str, Any]:
    """
    Estimate video generation time based on parameters

    Args:
        duration: Video duration in seconds
        resolution: Video resolution
        steps: Sampling steps

    Returns:
        Estimation dict with min/max times and factors
    """
    # Base times in seconds (rough estimates)
    base_times = {
        "480p": {"base": 30, "per_frame": 0.5},
        "720p": {"base": 60, "per_frame": 1.0},
        "1080p": {"base": 120, "per_frame": 2.0},
    }

    # Frame counts
    frame_counts = {3: 81, 5: 121, 10: 241}

    res_config = base_times.get(resolution, base_times["720p"])
    frames = frame_counts.get(duration, 121)

    # Calculate time
    base_time = res_config["base"]
    frame_time = res_config["per_frame"] * frames
    step_factor = steps / 30.0  # Normalize to 30 steps

    min_time = (base_time + frame_time) * step_factor * 0.8
    max_time = (base_time + frame_time) * step_factor * 1.5

    return {
        "min_seconds": round(min_time),
        "max_seconds": round(max_time),
        "frames": frames,
        "resolution": resolution,
        "steps": steps,
        "note": "Estimates vary based on GPU and model loading time",
    }


def create_simple_request(
    prompt: str,
    duration: int = 5,
    resolution: str = "720p",
) -> Dict[str, Any]:
    """
    Create a simple video generation request

    Args:
        prompt: Text prompt
        duration: Duration in seconds
        resolution: Resolution string

    Returns:
        Request dictionary ready for API
    """
    return {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "fps": 24,
    }


def merge_template_config(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge template configuration with overrides

    Args:
        base_config: Base template configuration
        overrides: Override values

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge dicts
            result[key] = merge_template_config(result[key], value)
        else:
            result[key] = value

    return result
