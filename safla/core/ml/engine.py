"""
Main neural embedding engine for the SAFLA ML system.

This module provides the core embedding generation and processing capabilities,
integrating all the ML components into a unified interface.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np

from .base import EmbeddingConfig, EmbeddingResult, EmbeddingType
from .models import TransformerEmbeddingModel

logger = logging.getLogger(__name__)


class NeuralEmbeddingEngine:
    """Main neural embedding engine with multi-modal support."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize neural embedding engine.
        
        Args:
            config: Embedding configuration, uses defaults if None
        """
        self.config = config or EmbeddingConfig()
        self.modality_processors = {}
        self.embedding_cache = {} if self.config.cache_embeddings else None
        self.processing_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "processing_time_total": 0.0,
            "last_activity": None
        }
        
        # Initialize processors for different modalities
        self._initialize_processors()
        
        logger.info(f"Initialized Neural Embedding Engine with config: {self.config.model_name}")
    
    def _initialize_processors(self):
        """Initialize modality-specific processors."""
        try:
            # Initialize text processor (primary)
            text_model = TransformerEmbeddingModel(self.config)
            self.modality_processors[EmbeddingType.TEXT] = text_model
            
            # Initialize other modality processors as needed
            if self.config.multi_modal_fusion != "none":
                # For now, reuse text model for multi-modal processing
                self.modality_processors[EmbeddingType.MULTIMODAL] = text_model
                
                # Placeholder for other modalities
                self.modality_processors[EmbeddingType.IMAGE] = self._create_image_processor()
                self.modality_processors[EmbeddingType.AUDIO] = self._create_audio_processor()
                self.modality_processors[EmbeddingType.CODE] = text_model  # Reuse text model
                
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def _create_image_processor(self):
        """Create image embedding processor."""
        # Placeholder for image processor
        return None
    
    def _create_audio_processor(self):
        """Create audio embedding processor."""
        # Placeholder for audio processor
        return None
    
    async def generate_embeddings(self, inputs: Union[str, List[str], Dict[str, Any]], 
                                modality: EmbeddingType = EmbeddingType.TEXT,
                                **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for given inputs.
        
        Args:
            inputs: Input data (text, list of texts, or multi-modal dict)
            modality: Type of embedding to generate
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult with generated embeddings
        """
        start_time = datetime.now()
        self.processing_stats["total_requests"] += 1
        self.processing_stats["last_activity"] = start_time
        
        try:
            # Check cache if enabled
            if self.embedding_cache is not None:
                cache_key = self._get_cache_key(inputs, modality)
                if cache_key in self.embedding_cache:
                    self.processing_stats["cache_hits"] += 1
                    cached_result = self.embedding_cache[cache_key]
                    cached_result.metadata["cache_hit"] = True
                    return cached_result
            
            # Process based on modality
            if modality == EmbeddingType.MULTIMODAL:
                result = await self._process_multimodal_async(inputs, **kwargs)
            else:
                result = await self._process_single_modality_async(inputs, modality, **kwargs)
            
            # Cache result if caching is enabled
            if self.embedding_cache is not None and cache_key:
                self.embedding_cache[cache_key] = result
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats["processing_time_total"] += processing_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def _process_single_modality_async(self, inputs: Union[str, List[str]], 
                                           modality: EmbeddingType,
                                           **kwargs) -> EmbeddingResult:
        """Process single modality inputs asynchronously."""
        processor = self.modality_processors.get(modality)
        if processor is None:
            raise ValueError(f"No processor available for modality: {modality}")
        
        # Convert inputs to batch format
        if isinstance(inputs, str):
            batch_inputs = [inputs]
            single_input = True
        else:
            batch_inputs = inputs
            single_input = False
        
        # Process in batches
        all_embeddings = []
        all_attention_weights = []
        
        for i in range(0, len(batch_inputs), self.config.batch_size):
            batch = batch_inputs[i:i + self.config.batch_size]
            
            # Prepare model inputs
            model_inputs = self._prepare_model_inputs(batch, modality)
            
            # Run inference
            with torch.no_grad():
                outputs = processor(model_inputs, modality)
                all_embeddings.append(outputs["embeddings"].cpu())
                
                if outputs.get("attention_weights") is not None:
                    all_attention_weights.append(outputs["attention_weights"].cpu())
        
        # Concatenate batch results
        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_attention = torch.cat(all_attention_weights, dim=0) if all_attention_weights else None
        
        # Return single embedding if single input
        if single_input:
            final_embeddings = final_embeddings[0:1]
            final_attention = final_attention[0:1] if final_attention is not None else None
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            metadata={
                "modality": modality.value,
                "batch_size": len(batch_inputs),
                "model_name": self.config.model_name,
                "cache_hit": False
            },
            processing_time=0.0,  # Will be set by caller
            model_version=self.config.model_name,
            attention_weights=final_attention
        )
    
    async def _process_multimodal_async(self, inputs: Dict[str, Any], **kwargs) -> EmbeddingResult:
        """Process multi-modal inputs asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_multimodal_sync, inputs, **kwargs
        )
    
    def _process_multimodal_sync(self, inputs: Dict[str, Any], **kwargs) -> EmbeddingResult:
        """Process multi-modal inputs synchronously."""
        start_time = datetime.now()
        
        processed_inputs = {}
        metadata = {"modalities": list(inputs.keys())}
        
        # Process each modality
        for modality_str, data in inputs.items():
            try:
                modality = EmbeddingType(modality_str)
                if modality in self.modality_processors and self.modality_processors[modality]:
                    processed_inputs[modality_str] = self._preprocess_modality_data(data, modality)
            except ValueError:
                logger.warning(f"Unsupported modality: {modality_str}")
                continue
        
        if not processed_inputs:
            raise ValueError("No valid modalities found in input")
        
        # Generate unified embedding
        model = self.modality_processors[EmbeddingType.TEXT]  # Use text model for fusion
        
        with torch.no_grad():
            result = model(processed_inputs, EmbeddingType.MULTIMODAL)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EmbeddingResult(
            embeddings=result["embeddings"].cpu(),
            metadata=metadata,
            processing_time=processing_time,
            model_version=self.config.model_name,
            attention_weights=result.get("attention_weights")
        )
    
    def _prepare_model_inputs(self, batch_inputs: List[str], modality: EmbeddingType) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model processing."""
        if modality == EmbeddingType.TEXT or modality == EmbeddingType.CODE:
            processor = self.modality_processors[EmbeddingType.TEXT]
            tokenizer = processor.tokenizer
            
            return tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            ).to(processor.device)
        else:
            raise NotImplementedError(f"Input preparation for {modality} not implemented")
    
    def _preprocess_modality_data(self, data: Any, modality: EmbeddingType) -> torch.Tensor:
        """Preprocess data for specific modality."""
        if modality == EmbeddingType.TEXT:
            # Tokenize text
            tokenizer = self.modality_processors[EmbeddingType.TEXT].tokenizer
            return tokenizer(
                data,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
        elif modality == EmbeddingType.IMAGE:
            # Convert image to tensor (placeholder)
            return torch.randn(1, 2048)  # Mock image features
        elif modality == EmbeddingType.AUDIO:
            # Convert audio to tensor (placeholder)
            return torch.randn(1, 1024)  # Mock audio features
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _get_cache_key(self, inputs: Union[str, List[str], Dict[str, Any]], 
                      modality: EmbeddingType) -> Optional[str]:
        """Generate cache key for inputs."""
        if not self.config.cache_embeddings:
            return None
        
        try:
            import hashlib
            
            # Create a string representation of inputs
            if isinstance(inputs, str):
                key_data = f"{modality.value}:{inputs}"
            elif isinstance(inputs, list):
                key_data = f"{modality.value}:{':'.join(inputs)}"
            elif isinstance(inputs, dict):
                key_data = f"{modality.value}:{str(sorted(inputs.items()))}"
            else:
                return None
            
            # Add config parameters that affect output
            key_data += f":{self.config.model_name}:{self.config.embedding_dim}"
            key_data += f":{self.config.normalize_embeddings}:{self.config.max_sequence_length}"
            
            return hashlib.sha256(key_data.encode()).hexdigest()
            
        except Exception:
            return None
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.embedding_cache is None:
            return {"caching_enabled": False}
        
        cache_size = len(self.embedding_cache)
        hit_rate = (self.processing_stats["cache_hits"] / 
                   max(1, self.processing_stats["total_requests"]))
        
        return {
            "caching_enabled": True,
            "cache_size": cache_size,
            "hit_rate": hit_rate,
            "total_requests": self.processing_stats["total_requests"],
            "cache_hits": self.processing_stats["cache_hits"]
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_requests = self.processing_stats["total_requests"]
        avg_processing_time = (
            self.processing_stats["processing_time_total"] / max(1, total_requests)
        )
        
        return {
            "total_requests": total_requests,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.processing_stats["processing_time_total"],
            "last_activity": self.processing_stats["last_activity"],
            "cache_stats": self.get_cache_stats()
        }
    
    def update_config(self, new_config: EmbeddingConfig):
        """Update engine configuration."""
        old_model_name = self.config.model_name
        self.config = new_config
        
        # Reinitialize if model changed
        if new_config.model_name != old_model_name:
            self._initialize_processors()
            self.clear_cache()  # Clear cache since model changed
            
        logger.info(f"Updated configuration: {new_config.model_name}")
    
    def get_supported_modalities(self) -> List[str]:
        """Get list of supported modalities."""
        return [modality.value for modality in self.modality_processors.keys() 
                if self.modality_processors[modality] is not None]
    
    async def batch_process(self, input_list: List[Dict[str, Any]], 
                          max_concurrent: int = 5) -> List[EmbeddingResult]:
        """Process multiple inputs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(input_item: Dict[str, Any]) -> EmbeddingResult:
            async with semaphore:
                return await self.generate_embeddings(
                    input_item.get("inputs"),
                    modality=EmbeddingType(input_item.get("modality", "text")),
                    **input_item.get("kwargs", {})
                )
        
        tasks = [process_single(item) for item in input_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process item {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        pass