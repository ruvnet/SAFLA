"""
Model versioning and management for the SAFLA ML system.

This module provides version control and management capabilities for 
embedding models, including saving, loading, and comparing model versions.
"""

import pickle
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import EmbeddingConfig, ModelMetrics
from .models import TransformerEmbeddingModel

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manager for model versions and checkpoints."""
    
    def __init__(self, storage_path: str = "./model_versions"):
        """
        Initialize model version manager.
        
        Args:
            storage_path: Path to store model versions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.version_history: List[str] = []
        self.current_version: Optional[str] = None
        
        # Load existing version metadata
        self._load_version_metadata()
        
        logger.info(f"Initialized ModelVersionManager with storage: {storage_path}")
    
    def save_model_version(self, model: TransformerEmbeddingModel, 
                          version_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a model version with metadata."""
        try:
            # Generate version ID if not provided
            if version_id is None:
                version_id = self._generate_version_id(model, metadata)
            
            # Prepare version directory
            version_dir = self.storage_path / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model checkpoint
            model_path = version_dir / "model.pt"
            model.save_checkpoint(str(model_path), metadata)
            
            # Save additional metadata
            version_info = {
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "config": model.config,
                "model_size": model.get_model_size(),
                "metadata": metadata or {},
                "performance_metrics": metadata.get("performance_metrics", {})
            }
            
            # Save version metadata
            metadata_path = version_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(version_info, f)
            
            # Update internal tracking
            self.versions[version_id] = version_info
            if version_id not in self.version_history:
                self.version_history.append(version_id)
            
            # Update current version if this is better
            if self._is_better_version(version_info):
                self.current_version = version_id
            
            # Save version metadata to disk
            self._save_version_metadata()
            
            logger.info(f"Saved model version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model version {version_id}: {e}")
            return False
    
    def load_model_version(self, version_id: str) -> Optional[TransformerEmbeddingModel]:
        """Load a specific model version."""
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return None
        
        try:
            version_info = self.versions[version_id]
            model_path = version_info["model_path"]
            
            # Load model from checkpoint
            model, _ = TransformerEmbeddingModel.load_checkpoint(model_path)
            
            logger.info(f"Loaded model version: {version_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model version {version_id}: {e}")
            return None
    
    def get_best_version(self) -> Optional[str]:
        """Get the best performing model version."""
        if not self.versions:
            return None
        
        best_version = None
        best_score = -1.0
        
        for version_id, version_info in self.versions.items():
            metrics = version_info.get("performance_metrics", {})
            score = metrics.get("embedding_quality_score", 0.0)
            
            if score > best_score:
                best_score = score
                best_version = version_id
        
        return best_version
    
    def get_latest_version(self) -> Optional[str]:
        """Get the most recently saved version."""
        if not self.version_history:
            return None
        return self.version_history[-1]
    
    def compare_versions(self, version_1: str, version_2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        if version_1 not in self.versions or version_2 not in self.versions:
            raise ValueError("One or both versions not found")
        
        v1_info = self.versions[version_1]
        v2_info = self.versions[version_2]
        
        # Compare performance metrics
        v1_metrics = v1_info.get("performance_metrics", {})
        v2_metrics = v2_info.get("performance_metrics", {})
        
        # Compare model sizes
        v1_size = v1_info.get("model_size", {})
        v2_size = v2_info.get("model_size", {})
        
        # Compute differences
        metric_diffs = {}
        for metric in ["embedding_quality_score", "silhouette_score", "retrieval_accuracy"]:
            v1_val = v1_metrics.get(metric, 0.0)
            v2_val = v2_metrics.get(metric, 0.0)
            metric_diffs[metric] = {
                "version_1": v1_val,
                "version_2": v2_val,
                "difference": v2_val - v1_val,
                "improvement": v2_val > v1_val
            }
        
        size_diff = {
            "version_1_params": v1_size.get("total_parameters", 0),
            "version_2_params": v2_size.get("total_parameters", 0),
            "parameter_difference": v2_size.get("total_parameters", 0) - v1_size.get("total_parameters", 0)
        }
        
        return {
            "version_1": version_1,
            "version_2": version_2,
            "metric_comparison": metric_diffs,
            "size_comparison": size_diff,
            "timestamp_1": v1_info.get("timestamp"),
            "timestamp_2": v2_info.get("timestamp"),
            "recommendation": self._get_version_recommendation(v1_info, v2_info)
        }
    
    def delete_version(self, version_id: str) -> bool:
        """Delete a model version and its files."""
        if version_id not in self.versions:
            logger.warning(f"Version {version_id} not found")
            return False
        
        try:
            # Remove version directory
            version_dir = self.storage_path / version_id
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)
            
            # Update internal tracking
            del self.versions[version_id]
            if version_id in self.version_history:
                self.version_history.remove(version_id)
            
            # Update current version if this was current
            if self.current_version == version_id:
                self.current_version = self.get_best_version()
            
            # Save updated metadata
            self._save_version_metadata()
            
            logger.info(f"Deleted model version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions."""
        version_list = []
        
        for version_id in self.version_history:
            if version_id in self.versions:
                version_info = self.versions[version_id].copy()
                version_info["is_current"] = version_id == self.current_version
                version_info["is_best"] = version_id == self.get_best_version()
                version_list.append(version_info)
        
        # Sort by timestamp (newest first)
        version_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return version_list
    
    def cleanup_old_versions(self, keep_count: int = 10) -> int:
        """Remove old versions, keeping only the most recent ones."""
        if len(self.version_history) <= keep_count:
            return 0
        
        # Sort versions by timestamp
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True
        )
        
        # Keep best version, current version, and recent versions
        keep_versions = set()
        
        # Always keep best and current
        if self.current_version:
            keep_versions.add(self.current_version)
        best_version = self.get_best_version()
        if best_version:
            keep_versions.add(best_version)
        
        # Keep most recent versions
        for version_id, _ in sorted_versions[:keep_count]:
            keep_versions.add(version_id)
        
        # Delete versions not in keep set
        deleted_count = 0
        for version_id in list(self.versions.keys()):
            if version_id not in keep_versions:
                if self.delete_version(version_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old model versions")
        return deleted_count
    
    def _generate_version_id(self, model: TransformerEmbeddingModel, 
                            metadata: Optional[Dict[str, Any]]) -> str:
        """Generate a unique version ID."""
        # Create hash from model state and timestamp
        timestamp = datetime.now().isoformat()
        model_hash = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()[:8]

        # Include metadata in hash if available
        if metadata:
            metadata_str = str(sorted(metadata.items()))
            metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()[:4]
            version_id = f"v_{timestamp.replace(':', '-').replace('.', '-')}_{model_hash}_{metadata_hash}"
        else:
            version_id = f"v_{timestamp.replace(':', '-').replace('.', '-')}_{model_hash}"
        
        return version_id
    
    def _is_better_version(self, version_info: Dict[str, Any]) -> bool:
        """Check if this version is better than current."""
        if self.current_version is None:
            return True
        
        current_metrics = self.versions[self.current_version].get("performance_metrics", {})
        new_metrics = version_info.get("performance_metrics", {})
        
        current_score = current_metrics.get("embedding_quality_score", 0.0)
        new_score = new_metrics.get("embedding_quality_score", 0.0)
        
        return new_score > current_score
    
    def _get_version_recommendation(self, v1_info: Dict[str, Any], 
                                  v2_info: Dict[str, Any]) -> str:
        """Get recommendation on which version to use."""
        v1_metrics = v1_info.get("performance_metrics", {})
        v2_metrics = v2_info.get("performance_metrics", {})
        
        v1_score = v1_metrics.get("embedding_quality_score", 0.0)
        v2_score = v2_metrics.get("embedding_quality_score", 0.0)
        
        score_diff = abs(v2_score - v1_score)
        
        if score_diff < 0.01:
            return "Versions have similar performance - consider other factors"
        elif v2_score > v1_score:
            return f"Version 2 is better (+{score_diff:.3f} quality score)"
        else:
            return f"Version 1 is better (+{score_diff:.3f} quality score)"
    
    def _load_version_metadata(self):
        """Load version metadata from disk."""
        metadata_file = self.storage_path / "versions_metadata.pkl"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.versions = saved_data.get("versions", {})
                    self.version_history = saved_data.get("version_history", [])
                    self.current_version = saved_data.get("current_version")
                
                logger.info(f"Loaded {len(self.versions)} version records")
                
            except Exception as e:
                logger.error(f"Failed to load version metadata: {e}")
                self.versions = {}
                self.version_history = []
                self.current_version = None
    
    def _save_version_metadata(self):
        """Save version metadata to disk."""
        metadata_file = self.storage_path / "versions_metadata.pkl"
        
        try:
            save_data = {
                "versions": self.versions,
                "version_history": self.version_history,
                "current_version": self.current_version,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for versions."""
        total_size = 0
        version_sizes = {}
        
        for version_id in self.versions:
            version_dir = self.storage_path / version_id
            if version_dir.exists():
                size = sum(f.stat().st_size for f in version_dir.rglob('*') if f.is_file())
                version_sizes[version_id] = size
                total_size += size
        
        return {
            "total_versions": len(self.versions),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_path),
            "version_sizes": version_sizes,
            "current_version": self.current_version,
            "best_version": self.get_best_version()
        }