"""
Base template system for ComfyUI workflow generation

Provides a registry and base class for creating workflow templates
that can be customized via API parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import copy
import logging

from .schemas import (
    VideoGenerateRequest,
    ImageGenerateRequest,
    TemplateConfig,
    ModelConfig,
    SamplerConfig,
    LoRAConfig,
    LyCORISConfig,
)

logger = logging.getLogger(__name__)


class BaseTemplate(ABC):
    """
    Abstract base class for workflow templates

    Templates define the structure of a ComfyUI workflow and provide
    methods to customize it based on API parameters.
    """

    # Template metadata
    name: str = "base"
    description: str = "Base template"
    version: str = "1.0.0"

    # Default model configuration
    default_models: ModelConfig = ModelConfig()

    # Default sampler configuration
    default_sampler: SamplerConfig = SamplerConfig()

    # Node ID mappings for easy customization
    # Maps logical names to node IDs in the workflow
    node_map: Dict[str, str] = {}

    def __init__(self):
        self._workflow: Dict[str, Any] = {}

    @abstractmethod
    def get_base_workflow(self) -> Dict[str, Any]:
        """Return the base workflow structure"""
        pass

    def build_workflow(
        self,
        request: Any,
        config: Optional[TemplateConfig] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete workflow from a request

        Args:
            request: VideoGenerateRequest or ImageGenerateRequest
            config: Optional template configuration overrides

        Returns:
            Complete ComfyUI workflow dictionary
        """
        # Start with base workflow
        workflow = copy.deepcopy(self.get_base_workflow())

        # Apply request parameters
        workflow = self._apply_request_params(workflow, request)

        # Apply config overrides
        if config:
            workflow = self._apply_config(workflow, config)

        # Apply LoRA/LyCORIS if present
        if hasattr(request, "loras") and request.loras:
            workflow = self._inject_loras(workflow, request.loras)
        if hasattr(request, "lycoris") and request.lycoris:
            workflow = self._inject_lycoris(workflow, request.lycoris)

        return workflow

    @abstractmethod
    def _apply_request_params(
        self, workflow: Dict[str, Any], request: Any
    ) -> Dict[str, Any]:
        """Apply request-specific parameters to the workflow"""
        pass

    def _apply_config(
        self, workflow: Dict[str, Any], config: TemplateConfig
    ) -> Dict[str, Any]:
        """Apply template configuration overrides"""
        # Apply model overrides
        if config.model:
            workflow = self._apply_model_config(workflow, config.model)

        # Apply sampler overrides
        if config.sampler:
            workflow = self._apply_sampler_config(workflow, config.sampler)

        # Apply node-specific overrides
        if config.node_overrides:
            for node_id, params in config.node_overrides.items():
                # Handle both logical names and direct node IDs
                actual_node_id = self.node_map.get(node_id, node_id)
                if actual_node_id in workflow:
                    workflow[actual_node_id]["inputs"].update(params)
                else:
                    logger.warning(f"Node {node_id} not found in workflow")

        # Apply LoRA/LyCORIS from config
        if config.loras:
            workflow = self._inject_loras(workflow, config.loras)
        if config.lycoris:
            workflow = self._inject_lycoris(workflow, config.lycoris)

        return workflow

    def _apply_model_config(
        self, workflow: Dict[str, Any], model: ModelConfig
    ) -> Dict[str, Any]:
        """Apply model configuration to relevant nodes"""
        # Override checkpoint loader
        if model.checkpoint and "checkpoint_loader" in self.node_map:
            node_id = self.node_map["checkpoint_loader"]
            if node_id in workflow:
                workflow[node_id]["inputs"]["ckpt_name"] = model.checkpoint

        # Override UNet loader
        if model.unet and "unet_loader" in self.node_map:
            node_id = self.node_map["unet_loader"]
            if node_id in workflow:
                workflow[node_id]["inputs"]["unet_name"] = model.unet

        # Override VAE
        if model.vae and "vae_loader" in self.node_map:
            node_id = self.node_map["vae_loader"]
            if node_id in workflow:
                workflow[node_id]["inputs"]["vae_name"] = model.vae

        # Override CLIP
        if model.clip and "clip_loader" in self.node_map:
            node_id = self.node_map["clip_loader"]
            if node_id in workflow:
                if "clip_name" in workflow[node_id]["inputs"]:
                    workflow[node_id]["inputs"]["clip_name"] = model.clip

        return workflow

    def _apply_sampler_config(
        self, workflow: Dict[str, Any], sampler: SamplerConfig
    ) -> Dict[str, Any]:
        """Apply sampler configuration to relevant nodes"""
        # Find and update sampler node
        if "sampler" in self.node_map:
            node_id = self.node_map["sampler"]
            if node_id in workflow:
                inputs = workflow[node_id]["inputs"]
                if "sampler_name" in inputs:
                    inputs["sampler_name"] = sampler.sampler_name
                if "scheduler" in inputs:
                    inputs["scheduler"] = sampler.scheduler
                if "steps" in inputs:
                    inputs["steps"] = sampler.steps
                if "cfg" in inputs:
                    inputs["cfg"] = sampler.cfg_scale
                if "denoise" in inputs:
                    inputs["denoise"] = sampler.denoise

        # Also check for scheduler node (some workflows separate these)
        if "scheduler" in self.node_map:
            node_id = self.node_map["scheduler"]
            if node_id in workflow:
                inputs = workflow[node_id]["inputs"]
                if "scheduler" in inputs:
                    inputs["scheduler"] = sampler.scheduler
                if "steps" in inputs:
                    inputs["steps"] = sampler.steps
                if "denoise" in inputs:
                    inputs["denoise"] = sampler.denoise

        return workflow

    def _inject_loras(
        self, workflow: Dict[str, Any], loras: List[LoRAConfig]
    ) -> Dict[str, Any]:
        """
        Inject LoRA loader nodes into the workflow

        This creates a chain of LoRA loaders that modify the model/clip
        """
        if not loras:
            return workflow

        # Find the model source node
        model_source_node = self.node_map.get(
            "unet_loader", self.node_map.get("checkpoint_loader")
        )
        if not model_source_node:
            logger.warning("No model source node found for LoRA injection")
            return workflow

        # Find nodes that consume the model output
        model_output_index = 0  # Usually model is output 0
        model_consumers = self._find_consumers(workflow, model_source_node, model_output_index)

        # Create LoRA chain
        prev_node_id = model_source_node
        prev_model_output = model_output_index

        # Track the highest node ID for new nodes
        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())

        for i, lora in enumerate(loras):
            new_node_id = str(max_node_id + 1 + i)

            # Create LoRA loader node
            lora_node = {
                "inputs": {
                    "lora_name": lora.name,
                    "strength_model": lora.weight,
                    "strength_clip": lora.clip_weight if lora.clip_weight is not None else lora.weight,
                    "model": [prev_node_id, prev_model_output],
                    "clip": [prev_node_id, 1],  # CLIP is usually output 1
                },
                "class_type": "LoraLoader",
                "_meta": {"title": f"LoRA: {lora.name}"},
            }

            workflow[new_node_id] = lora_node
            prev_node_id = new_node_id
            prev_model_output = 0  # LoRA output is always at index 0

        # Update consumers to use the last LoRA in chain
        for consumer_id, input_name in model_consumers:
            workflow[consumer_id]["inputs"][input_name] = [prev_node_id, 0]

        return workflow

    def _inject_lycoris(
        self, workflow: Dict[str, Any], lycoris_list: List[LyCORISConfig]
    ) -> Dict[str, Any]:
        """
        Inject LyCORIS loader nodes into the workflow

        LyCORIS uses a different loader class but similar pattern to LoRA
        """
        if not lycoris_list:
            return workflow

        # Find the model source - check after LoRA chain if present
        model_source_node = self.node_map.get(
            "unet_loader", self.node_map.get("checkpoint_loader")
        )
        if not model_source_node:
            logger.warning("No model source node found for LyCORIS injection")
            return workflow

        # Find the last node in the model chain (could be LoRA)
        last_model_node = model_source_node
        for node_id, node in workflow.items():
            if node.get("class_type") == "LoraLoader":
                last_model_node = node_id

        model_consumers = self._find_consumers(workflow, last_model_node, 0)

        # Track the highest node ID
        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())

        prev_node_id = last_model_node
        for i, lycoris in enumerate(lycoris_list):
            new_node_id = str(max_node_id + 1 + i)

            # LyCORIS uses LoraLoaderModelOnly or a custom LyCORIS node
            # Using LoraLoaderModelOnly for compatibility
            lycoris_node = {
                "inputs": {
                    "lora_name": lycoris.name,
                    "strength_model": lycoris.weight,
                    "model": [prev_node_id, 0],
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": {"title": f"LyCORIS: {lycoris.name}"},
            }

            workflow[new_node_id] = lycoris_node
            prev_node_id = new_node_id

        # Update consumers
        for consumer_id, input_name in model_consumers:
            workflow[consumer_id]["inputs"][input_name] = [prev_node_id, 0]

        return workflow

    def _find_consumers(
        self, workflow: Dict[str, Any], source_node: str, output_index: int
    ) -> List[tuple]:
        """Find all nodes that consume a specific output"""
        consumers = []
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            for input_name, input_value in inputs.items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    if input_value[0] == source_node and input_value[1] == output_index:
                        consumers.append((node_id, input_name))
        return consumers

    def get_required_models(self) -> Dict[str, str]:
        """Return dict of required models (type -> filename)"""
        return {}

    def get_required_nodes(self) -> List[str]:
        """Return list of required ComfyUI custom nodes"""
        return []


class TemplateRegistry:
    """
    Registry for workflow templates

    Provides template lookup and management.
    """

    _templates: Dict[str, Type[BaseTemplate]] = {}
    _instances: Dict[str, BaseTemplate] = {}

    @classmethod
    def register(cls, template_class: Type[BaseTemplate]) -> Type[BaseTemplate]:
        """Register a template class (can be used as decorator)"""
        name = template_class.name
        cls._templates[name] = template_class
        logger.info(f"Registered template: {name}")
        return template_class

    @classmethod
    def get(cls, name: str) -> Optional[BaseTemplate]:
        """Get a template instance by name"""
        if name not in cls._instances:
            if name not in cls._templates:
                return None
            cls._instances[name] = cls._templates[name]()
        return cls._instances[name]

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """List all registered templates with metadata"""
        templates = []
        for name, template_class in cls._templates.items():
            templates.append({
                "name": name,
                "description": template_class.description,
                "version": template_class.version,
            })
        return templates

    @classmethod
    def get_template_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a template"""
        template = cls.get(name)
        if not template:
            return None

        return {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "default_models": template.default_models.to_dict() if template.default_models else {},
            "default_sampler": template.default_sampler.to_dict() if template.default_sampler else {},
            "node_map": template.node_map,
            "required_models": template.get_required_models(),
            "required_nodes": template.get_required_nodes(),
        }
