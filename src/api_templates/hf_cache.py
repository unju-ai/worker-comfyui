"""
HuggingFace Model Caching System

Provides automatic model downloading and caching from HuggingFace Hub.
Supports:
- Automatic model download on first use
- Local caching to avoid re-downloads
- Token authentication for gated models
- Multiple model types (checkpoints, LoRAs, VAEs, etc.)
"""

import os
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Default HuggingFace repositories for WAN models
HF_MODEL_REPOS = {
    # WAN 2.1 Models (Comfy-Org repackaged)
    "wan2.1_i2v_480p_bf16.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/diffusion_models/wan2.1_i2v_480p_bf16.safetensors",
        "type": "unet",
    },
    "wan2.1_i2v_720p_bf16.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/diffusion_models/wan2.1_i2v_720p_bf16.safetensors",
        "type": "unet",
    },
    "wan2.1_t2v_480p_bf16.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/diffusion_models/wan2.1_t2v_480p_bf16.safetensors",
        "type": "unet",
    },
    "wan2.1_t2v_720p_bf16.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/diffusion_models/wan2.1_t2v_720p_bf16.safetensors",
        "type": "unet",
    },
    "wan2.1_t2v_1.3B_bf16.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
        "type": "unet",
    },
    # Text Encoders
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "type": "clip",
    },
    # VAE
    "wan_2.1_vae.safetensors": {
        "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "path": "split_files/vae/wan_2.1_vae.safetensors",
        "type": "vae",
    },
    # CLIP Vision
    "clip-vit-large-patch14.safetensors": {
        "repo": "openai/clip-vit-large-patch14",
        "path": "model.safetensors",
        "type": "clip_vision",
    },
}

# Model type to ComfyUI directory mapping
MODEL_TYPE_DIRS = {
    "checkpoint": "checkpoints",
    "unet": "unet",
    "vae": "vae",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "lora": "loras",
    "lycoris": "loras",  # LyCORIS uses same dir as LoRA
    "controlnet": "controlnet",
    "embeddings": "embeddings",
}


@dataclass
class CacheEntry:
    """Represents a cached model entry"""
    filename: str
    repo: str
    path: str
    model_type: str
    local_path: str
    size_bytes: int = 0
    downloaded_at: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class CacheConfig:
    """Configuration for the cache system"""
    # Base directory for cached models
    cache_dir: str = "/runpod-volume/models"
    # Alternative: use ComfyUI models directory
    comfy_models_dir: str = "/comfyui/models"
    # HuggingFace token for gated models
    hf_token: Optional[str] = None
    # Maximum cache size in GB (0 = unlimited)
    max_cache_size_gb: float = 0
    # Enable automatic downloads
    auto_download: bool = True
    # Prefer network volume over local
    prefer_network_volume: bool = True


class HuggingFaceModelCache:
    """
    Manages HuggingFace model caching for ComfyUI

    Features:
    - Automatic download from HuggingFace Hub
    - Local file caching with metadata
    - Support for gated models via token auth
    - Thread-safe operations
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # Get HF token from config or environment
        self.hf_token = self.config.hf_token or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

        # Determine base cache directory
        if self.config.prefer_network_volume and os.path.exists("/runpod-volume"):
            self.base_dir = Path(self.config.cache_dir)
        else:
            self.base_dir = Path(self.config.comfy_models_dir)

        # Cache metadata file
        self.metadata_file = self.base_dir / ".hf_cache_metadata.json"
        self._metadata: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        # Load existing metadata
        self._load_metadata()

        logger.info(f"HuggingFace cache initialized at {self.base_dir}")

    def _load_metadata(self):
        """Load cache metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._metadata[key] = CacheEntry(**entry_data)
                logger.info(f"Loaded {len(self._metadata)} cached model entries")
        except Exception as e:
            logger.warning(f"Could not load cache metadata: {e}")
            self._metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, "w") as f:
                data = {k: v.__dict__ for k, v in self._metadata.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")

    def get_model_path(
        self,
        filename: str,
        model_type: Optional[str] = None,
        repo: Optional[str] = None,
        hf_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the local path for a model, downloading if necessary

        Args:
            filename: Model filename
            model_type: Type of model (unet, vae, lora, etc.)
            repo: HuggingFace repo (optional, uses defaults)
            hf_path: Path within the repo (optional)

        Returns:
            Local path to the model, or None if not available
        """
        with self._lock:
            # Check if already cached
            if filename in self._metadata:
                entry = self._metadata[filename]
                if os.path.exists(entry.local_path):
                    logger.debug(f"Cache hit: {filename}")
                    return entry.local_path

            # Look up in known repos
            model_info = HF_MODEL_REPOS.get(filename)
            if model_info:
                repo = repo or model_info["repo"]
                hf_path = hf_path or model_info["path"]
                model_type = model_type or model_info["type"]

            # Try to find locally without downloading
            local_path = self._find_local_model(filename, model_type)
            if local_path:
                self._add_to_cache(filename, repo or "", hf_path or "", model_type or "unknown", local_path)
                return local_path

            # Download if auto_download enabled and we have repo info
            if self.config.auto_download and repo and hf_path:
                return self._download_model(filename, repo, hf_path, model_type or "unknown")

            logger.warning(f"Model not found and auto-download disabled: {filename}")
            return None

    def _find_local_model(self, filename: str, model_type: Optional[str] = None) -> Optional[str]:
        """Search for model in local directories"""
        search_dirs = []

        # Add type-specific directory
        if model_type and model_type in MODEL_TYPE_DIRS:
            type_dir = MODEL_TYPE_DIRS[model_type]
            search_dirs.append(self.base_dir / type_dir)
            search_dirs.append(Path(self.config.comfy_models_dir) / type_dir)

        # Add general search paths
        search_dirs.extend([
            self.base_dir,
            Path(self.config.comfy_models_dir),
            Path("/runpod-volume/models"),
        ])

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Direct match
            direct_path = search_dir / filename
            if direct_path.exists():
                return str(direct_path)

            # Search subdirectories
            for path in search_dir.rglob(filename):
                return str(path)

        return None

    def _download_model(
        self,
        filename: str,
        repo: str,
        hf_path: str,
        model_type: str,
    ) -> Optional[str]:
        """Download model from HuggingFace Hub"""
        try:
            # Determine target directory
            type_dir = MODEL_TYPE_DIRS.get(model_type, model_type)
            target_dir = self.base_dir / type_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            logger.info(f"Downloading {filename} from {repo}/{hf_path}...")

            # Try using huggingface_hub if available
            try:
                from huggingface_hub import hf_hub_download

                downloaded_path = hf_hub_download(
                    repo_id=repo,
                    filename=hf_path,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=self.hf_token,
                )

                # If the downloaded file has a different name, rename it
                if os.path.basename(downloaded_path) != filename:
                    final_path = target_dir / filename
                    os.rename(downloaded_path, final_path)
                    downloaded_path = str(final_path)

                self._add_to_cache(filename, repo, hf_path, model_type, downloaded_path)
                logger.info(f"Successfully downloaded {filename}")
                return downloaded_path

            except ImportError:
                # Fallback to direct HTTP download
                return self._download_http(filename, repo, hf_path, model_type, target_path)

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None

    def _download_http(
        self,
        filename: str,
        repo: str,
        hf_path: str,
        model_type: str,
        target_path: Path,
    ) -> Optional[str]:
        """Fallback HTTP download without huggingface_hub"""
        import urllib.request

        url = f"https://huggingface.co/{repo}/resolve/main/{hf_path}"
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:
                with open(target_path, "wb") as f:
                    # Download in chunks to handle large files
                    chunk_size = 8192 * 1024  # 8MB chunks
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

            self._add_to_cache(filename, repo, hf_path, model_type, str(target_path))
            logger.info(f"Successfully downloaded {filename} via HTTP")
            return str(target_path)

        except Exception as e:
            logger.error(f"HTTP download failed for {filename}: {e}")
            if target_path.exists():
                target_path.unlink()
            return None

    def _add_to_cache(
        self,
        filename: str,
        repo: str,
        hf_path: str,
        model_type: str,
        local_path: str,
    ):
        """Add model to cache metadata"""
        size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

        entry = CacheEntry(
            filename=filename,
            repo=repo,
            path=hf_path,
            model_type=model_type,
            local_path=local_path,
            size_bytes=size,
            downloaded_at=datetime.now().isoformat(),
        )

        self._metadata[filename] = entry
        self._save_metadata()

    def is_cached(self, filename: str) -> bool:
        """Check if a model is cached and available"""
        if filename not in self._metadata:
            return False
        entry = self._metadata[filename]
        return os.path.exists(entry.local_path)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        total_size = sum(e.size_bytes for e in self._metadata.values() if os.path.exists(e.local_path))
        return {
            "base_dir": str(self.base_dir),
            "total_models": len(self._metadata),
            "total_size_gb": round(total_size / (1024**3), 2),
            "auto_download": self.config.auto_download,
            "has_hf_token": self.hf_token is not None,
        }

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached models"""
        models = []
        for filename, entry in self._metadata.items():
            exists = os.path.exists(entry.local_path)
            models.append({
                "filename": filename,
                "type": entry.model_type,
                "repo": entry.repo,
                "local_path": entry.local_path,
                "exists": exists,
                "size_gb": round(entry.size_bytes / (1024**3), 2) if exists else 0,
            })
        return models

    def ensure_models(self, model_list: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
        """
        Ensure multiple models are available

        Args:
            model_list: List of dicts with filename, type, repo, path

        Returns:
            Dict mapping filename to local path (or None if unavailable)
        """
        results = {}
        for model in model_list:
            path = self.get_model_path(
                filename=model["filename"],
                model_type=model.get("type"),
                repo=model.get("repo"),
                hf_path=model.get("path"),
            )
            results[model["filename"]] = path
        return results

    def clear_cache(self, keep_models: Optional[List[str]] = None):
        """Clear cached models (optionally keeping some)"""
        keep_set = set(keep_models or [])

        for filename, entry in list(self._metadata.items()):
            if filename in keep_set:
                continue

            if os.path.exists(entry.local_path):
                try:
                    os.remove(entry.local_path)
                    logger.info(f"Removed cached model: {filename}")
                except Exception as e:
                    logger.warning(f"Could not remove {filename}: {e}")

            del self._metadata[filename]

        self._save_metadata()


# Global cache instance
_global_cache: Optional[HuggingFaceModelCache] = None


def get_hf_cache() -> HuggingFaceModelCache:
    """Get or create the global HuggingFace cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = HuggingFaceModelCache()
    return _global_cache


def ensure_wan_models(resolution: str = "720p", mode: str = "t2v") -> Dict[str, Optional[str]]:
    """
    Convenience function to ensure WAN models are available

    Args:
        resolution: Video resolution (480p, 720p, 1080p)
        mode: Generation mode (t2v, i2v)

    Returns:
        Dict of model paths
    """
    cache = get_hf_cache()

    models = [
        {"filename": f"wan2.1_{mode}_{resolution}_bf16.safetensors", "type": "unet"},
        {"filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "clip"},
        {"filename": "wan_2.1_vae.safetensors", "type": "vae"},
    ]

    if mode == "i2v":
        models.append({"filename": "clip-vit-large-patch14.safetensors", "type": "clip_vision"})

    return cache.ensure_models(models)
