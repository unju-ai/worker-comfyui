"""
LoRA and LyCORIS Support Module

Provides utilities for loading and managing LoRA and LyCORIS models
in ComfyUI workflows. Supports:
- Standard LoRA models
- LyCORIS variants (LoHa, LoKr, etc.)
- Automatic HuggingFace downloading
- Weight stacking and blending
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from .schemas import LoRAConfig, LyCORISConfig
from .hf_cache import get_hf_cache, MODEL_TYPE_DIRS

logger = logging.getLogger(__name__)


# Popular LoRA repositories on HuggingFace
POPULAR_LORA_REPOS = {
    # Add commonly used LoRAs here for easy reference
    "lcm-lora": {
        "repo": "latent-consistency/lcm-lora-sdxl",
        "filename": "lcm-lora-sdxl.safetensors",
    },
}


@dataclass
class LoRAInfo:
    """Information about a loaded LoRA"""
    name: str
    path: str
    weight: float
    clip_weight: float
    type: str = "lora"  # lora, loha, lokr, etc.
    is_loaded: bool = False
    metadata: Optional[Dict[str, Any]] = None


class LoRALoader:
    """
    Manages LoRA loading and application to workflows

    Handles:
    - Local LoRA discovery
    - HuggingFace LoRA downloading
    - Weight configuration
    - Workflow node injection
    """

    def __init__(self, lora_dir: Optional[str] = None):
        # Search directories for LoRAs
        self.search_dirs = [
            Path(lora_dir) if lora_dir else Path("/runpod-volume/models/loras"),
            Path("/comfyui/models/loras"),
            Path.home() / ".cache" / "comfyui" / "loras",
        ]
        self._available_loras: Dict[str, str] = {}
        self._scan_loras()

    def _scan_loras(self):
        """Scan directories for available LoRAs"""
        for search_dir in self.search_dirs:
            if not search_dir.exists():
                continue

            for file in search_dir.rglob("*.safetensors"):
                name = file.stem
                if name not in self._available_loras:
                    self._available_loras[name] = str(file)
                    logger.debug(f"Found LoRA: {name} at {file}")

            for file in search_dir.rglob("*.ckpt"):
                name = file.stem
                if name not in self._available_loras:
                    self._available_loras[name] = str(file)

        logger.info(f"Found {len(self._available_loras)} LoRA models")

    def get_available_loras(self) -> List[str]:
        """Get list of available LoRA names"""
        return list(self._available_loras.keys())

    def find_lora(self, name: str) -> Optional[str]:
        """
        Find a LoRA by name, downloading if necessary

        Args:
            name: LoRA name (with or without extension)

        Returns:
            Path to LoRA file or None
        """
        # Clean up name
        clean_name = name
        for ext in [".safetensors", ".ckpt", ".pt"]:
            if clean_name.endswith(ext):
                clean_name = clean_name[:-len(ext)]
                break

        # Check local availability
        if clean_name in self._available_loras:
            return self._available_loras[clean_name]

        # Try with original name (might have extension)
        if name in self._available_loras:
            return self._available_loras[name]

        # Search in directories
        for search_dir in self.search_dirs:
            if not search_dir.exists():
                continue

            # Try exact match
            for ext in [".safetensors", ".ckpt", ".pt"]:
                path = search_dir / f"{clean_name}{ext}"
                if path.exists():
                    self._available_loras[clean_name] = str(path)
                    return str(path)

        # Check if it's a known HuggingFace LoRA
        if name in POPULAR_LORA_REPOS:
            repo_info = POPULAR_LORA_REPOS[name]
            cache = get_hf_cache()
            path = cache.get_model_path(
                filename=repo_info["filename"],
                model_type="lora",
                repo=repo_info["repo"],
            )
            if path:
                self._available_loras[name] = path
                return path

        logger.warning(f"LoRA not found: {name}")
        return None

    def ensure_lora(self, config: LoRAConfig) -> Optional[LoRAInfo]:
        """
        Ensure a LoRA is available and return its info

        Args:
            config: LoRA configuration

        Returns:
            LoRAInfo if available, None otherwise
        """
        # Try to find locally or download
        path = None

        # If HuggingFace source specified, download
        if config.hf_repo and config.hf_filename:
            cache = get_hf_cache()
            path = cache.get_model_path(
                filename=config.hf_filename,
                model_type="lora",
                repo=config.hf_repo,
            )

        # Otherwise search locally
        if not path:
            path = self.find_lora(config.name)

        if not path:
            return None

        return LoRAInfo(
            name=config.name,
            path=path,
            weight=config.weight,
            clip_weight=config.clip_weight if config.clip_weight is not None else config.weight,
            type="lora",
            is_loaded=True,
        )

    def inject_loras_to_workflow(
        self,
        workflow: Dict[str, Any],
        loras: List[LoRAConfig],
        model_source_node: str,
        model_output_index: int = 0,
        clip_output_index: int = 1,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Inject LoRA loader nodes into a workflow

        Args:
            workflow: ComfyUI workflow dict
            loras: List of LoRA configurations
            model_source_node: Node ID that outputs the model
            model_output_index: Output index for model
            clip_output_index: Output index for CLIP

        Returns:
            Tuple of (modified workflow, last LoRA node ID)
        """
        if not loras:
            return workflow, model_source_node

        # Find highest node ID
        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())

        prev_node_id = model_source_node
        prev_model_out = model_output_index
        prev_clip_out = clip_output_index

        for i, lora_config in enumerate(loras):
            # Ensure LoRA exists
            lora_info = self.ensure_lora(lora_config)
            if not lora_info:
                logger.warning(f"Skipping unavailable LoRA: {lora_config.name}")
                continue

            new_node_id = str(max_node_id + 1 + i)

            # Get the filename for ComfyUI
            lora_filename = os.path.basename(lora_info.path)

            # Create LoRA loader node
            lora_node = {
                "inputs": {
                    "lora_name": lora_filename,
                    "strength_model": lora_info.weight,
                    "strength_clip": lora_info.clip_weight,
                    "model": [prev_node_id, prev_model_out],
                    "clip": [prev_node_id, prev_clip_out],
                },
                "class_type": "LoraLoader",
                "_meta": {"title": f"LoRA: {lora_config.name}"},
            }

            workflow[new_node_id] = lora_node
            prev_node_id = new_node_id
            prev_model_out = 0  # LoraLoader outputs model at 0
            prev_clip_out = 1  # LoraLoader outputs clip at 1

            logger.info(f"Injected LoRA: {lora_config.name} (weight: {lora_info.weight})")

        return workflow, prev_node_id


class LyCORISLoader:
    """
    Manages LyCORIS model loading and application

    LyCORIS (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion)
    includes variants like:
    - LoHa (Low-Rank Hadamard Product)
    - LoKr (Low-Rank Kronecker Product)
    - Full/Partial fine-tuning
    """

    def __init__(self, lycoris_dir: Optional[str] = None):
        self.search_dirs = [
            Path(lycoris_dir) if lycoris_dir else Path("/runpod-volume/models/loras"),
            Path("/comfyui/models/loras"),
            # Some setups have a separate lycoris directory
            Path("/runpod-volume/models/lycoris"),
            Path("/comfyui/models/lycoris"),
        ]
        self._available_lycoris: Dict[str, str] = {}
        self._scan_lycoris()

    def _scan_lycoris(self):
        """Scan directories for available LyCORIS models"""
        for search_dir in self.search_dirs:
            if not search_dir.exists():
                continue

            for file in search_dir.rglob("*.safetensors"):
                name = file.stem
                if name not in self._available_lycoris:
                    self._available_lycoris[name] = str(file)

        logger.info(f"Found {len(self._available_lycoris)} LyCORIS models")

    def get_available_lycoris(self) -> List[str]:
        """Get list of available LyCORIS names"""
        return list(self._available_lycoris.keys())

    def find_lycoris(self, name: str) -> Optional[str]:
        """Find a LyCORIS model by name"""
        clean_name = name
        for ext in [".safetensors", ".ckpt", ".pt"]:
            if clean_name.endswith(ext):
                clean_name = clean_name[:-len(ext)]
                break

        if clean_name in self._available_lycoris:
            return self._available_lycoris[clean_name]

        if name in self._available_lycoris:
            return self._available_lycoris[name]

        # Search in directories
        for search_dir in self.search_dirs:
            if not search_dir.exists():
                continue

            for ext in [".safetensors", ".ckpt", ".pt"]:
                path = search_dir / f"{clean_name}{ext}"
                if path.exists():
                    self._available_lycoris[clean_name] = str(path)
                    return str(path)

        return None

    def ensure_lycoris(self, config: LyCORISConfig) -> Optional[LoRAInfo]:
        """
        Ensure a LyCORIS model is available

        Args:
            config: LyCORIS configuration

        Returns:
            LoRAInfo if available, None otherwise
        """
        path = None

        # If HuggingFace source specified, download
        if config.hf_repo and config.hf_filename:
            cache = get_hf_cache()
            path = cache.get_model_path(
                filename=config.hf_filename,
                model_type="lycoris",
                repo=config.hf_repo,
            )

        if not path:
            path = self.find_lycoris(config.name)

        if not path:
            return None

        return LoRAInfo(
            name=config.name,
            path=path,
            weight=config.weight,
            clip_weight=config.weight,  # LyCORIS typically uses same weight
            type="lycoris",
            is_loaded=True,
        )

    def inject_lycoris_to_workflow(
        self,
        workflow: Dict[str, Any],
        lycoris_list: List[LyCORISConfig],
        model_source_node: str,
        model_output_index: int = 0,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Inject LyCORIS loader nodes into a workflow

        Note: Uses LoraLoaderModelOnly for better compatibility,
        but can also use the full LoraLoader if CLIP modification is needed.

        Args:
            workflow: ComfyUI workflow dict
            lycoris_list: List of LyCORIS configurations
            model_source_node: Node ID that outputs the model
            model_output_index: Output index for model

        Returns:
            Tuple of (modified workflow, last LyCORIS node ID)
        """
        if not lycoris_list:
            return workflow, model_source_node

        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())

        prev_node_id = model_source_node
        prev_model_out = model_output_index

        for i, lycoris_config in enumerate(lycoris_list):
            lycoris_info = self.ensure_lycoris(lycoris_config)
            if not lycoris_info:
                logger.warning(f"Skipping unavailable LyCORIS: {lycoris_config.name}")
                continue

            new_node_id = str(max_node_id + 1 + i)

            lycoris_filename = os.path.basename(lycoris_info.path)

            # LyCORIS models work with standard LoRA loader
            # Using LoraLoaderModelOnly for model-only modifications
            lycoris_node = {
                "inputs": {
                    "lora_name": lycoris_filename,
                    "strength_model": lycoris_info.weight,
                    "model": [prev_node_id, prev_model_out],
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": {"title": f"LyCORIS: {lycoris_config.name}"},
            }

            workflow[new_node_id] = lycoris_node
            prev_node_id = new_node_id
            prev_model_out = 0

            logger.info(f"Injected LyCORIS: {lycoris_config.name} (weight: {lycoris_info.weight})")

        return workflow, prev_node_id


def apply_lora_and_lycoris(
    workflow: Dict[str, Any],
    loras: Optional[List[LoRAConfig]] = None,
    lycoris: Optional[List[LyCORISConfig]] = None,
    model_source_node: str = "1",
    model_output_index: int = 0,
    clip_output_index: int = 1,
) -> Dict[str, Any]:
    """
    Convenience function to apply both LoRA and LyCORIS to a workflow

    Args:
        workflow: ComfyUI workflow dict
        loras: List of LoRA configurations
        lycoris: List of LyCORIS configurations
        model_source_node: Node ID that outputs the base model
        model_output_index: Output index for model
        clip_output_index: Output index for CLIP

    Returns:
        Modified workflow with LoRA/LyCORIS injected
    """
    current_node = model_source_node

    # Apply LoRAs first
    if loras:
        lora_loader = LoRALoader()
        workflow, current_node = lora_loader.inject_loras_to_workflow(
            workflow,
            loras,
            current_node,
            model_output_index,
            clip_output_index,
        )
        # After LoRA, model is at 0, clip at 1
        model_output_index = 0
        clip_output_index = 1

    # Then apply LyCORIS
    if lycoris:
        lycoris_loader = LyCORISLoader()
        workflow, current_node = lycoris_loader.inject_lycoris_to_workflow(
            workflow,
            lycoris,
            current_node,
            model_output_index,
        )

    # Update nodes that reference the original model source
    _update_model_references(workflow, model_source_node, current_node)

    return workflow


def _update_model_references(
    workflow: Dict[str, Any],
    old_source: str,
    new_source: str,
):
    """Update workflow nodes to reference the new model source"""
    if old_source == new_source:
        return

    # Find nodes that reference the old model source
    for node_id, node in workflow.items():
        if node_id == new_source:
            continue

        inputs = node.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                if input_value[0] == old_source:
                    # Check if this is a model/clip input based on name
                    if input_name in ["model", "clip", "unet"]:
                        # Update to use the new source
                        inputs[input_name] = [new_source, input_value[1]]
