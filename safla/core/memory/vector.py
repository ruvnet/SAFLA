"""
Vector memory implementation for the SAFLA memory system.

This module provides high-dimensional vector storage and similarity search
capabilities using FAISS for efficient nearest neighbor search.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import threading
from datetime import datetime
import uuid

from .base import MemoryItem, SimilarityMetric, MemoryInterface

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using fallback vector search")


@dataclass
class VectorMemory(MemoryItem):
    """
    Vector memory item with embedding and similarity search capabilities.
    """
    vector: Optional[np.ndarray] = None
    dimension: int = 768
    encoding_method: str = "sentence_transformer"
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """Initialize vector if not provided."""
        if self.vector is None and self.content is not None:
            # Generate vector from content (simplified)
            self.vector = self._generate_embedding()
    
    def _generate_embedding(self) -> np.ndarray:
        """Generate embedding from content."""
        # Simplified embedding generation - in practice would use actual models
        if isinstance(self.content, str):
            # Simple hash-based embedding for demonstration
            import hashlib
            hash_val = hashlib.sha256(self.content.encode()).hexdigest()
            # Convert to numeric representation
            numeric_hash = [ord(c) for c in hash_val[:self.dimension//16]]
            # Pad or truncate to desired dimension
            while len(numeric_hash) < self.dimension:
                numeric_hash.extend(numeric_hash[:min(16, self.dimension - len(numeric_hash))])
            return np.array(numeric_hash[:self.dimension], dtype=np.float32)
        return np.random.random(self.dimension).astype(np.float32)
    
    def calculate_similarity(self, other: 'VectorMemory', 
                           metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """Calculate similarity with another vector memory."""
        if self.vector is None or other.vector is None:
            return 0.0
        
        if metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(self.vector, other.vector)
        elif metric == SimilarityMetric.EUCLIDEAN:
            return 1.0 / (1.0 + np.linalg.norm(self.vector - other.vector))
        elif metric == SimilarityMetric.MANHATTAN:
            return 1.0 / (1.0 + np.sum(np.abs(self.vector - other.vector)))
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(self.vector, other.vector)
        
        return 0.0
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


class VectorMemoryManager(MemoryInterface):
    """
    Manager for vector memory operations with FAISS indexing.
    """
    
    def __init__(self, dimension: int = 768, index_type: str = "IVF"):
        """
        Initialize vector memory manager.
        
        Args:
            dimension: Vector dimension
            index_type: FAISS index type ("IVF", "HNSW", "Flat")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.memories: Dict[str, VectorMemory] = {}
        self._lock = threading.RLock()
        
        # Initialize FAISS index
        self.index = None
        self.index_to_memory_id: List[str] = []
        self.index_metadata = {
            "total_vectors": 0,
            "last_rebuild": None,
            "index_type": index_type
        }
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on configuration."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using linear search fallback")
            return
        
        try:
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                logger.warning(f"Unknown index type {self.index_type}, using Flat")
                self.index = faiss.IndexFlatIP(self.dimension)
                
            logger.info(f"Initialized FAISS {self.index_type} index with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None, 
              vector: Optional[np.ndarray] = None) -> str:
        """Store vector memory item."""
        with self._lock:
            # Create vector memory
            memory = VectorMemory(
                content=content,
                metadata=metadata or {},
                vector=vector,
                dimension=self.dimension
            )
            
            # Store in memory dict
            self.memories[memory.memory_id] = memory
            
            # Add to FAISS index
            if self.index is not None and memory.vector is not None:
                try:
                    # FAISS expects float32 vectors
                    vector_array = memory.vector.astype(np.float32).reshape(1, -1)
                    
                    # Train index if needed (for IVF)
                    if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                        if len(self.memories) >= 100:  # Need enough vectors to train
                            all_vectors = np.array([
                                m.vector for m in self.memories.values() 
                                if m.vector is not None
                            ]).astype(np.float32)
                            self.index.train(all_vectors)
                    
                    if hasattr(self.index, 'is_trained') and self.index.is_trained:
                        self.index.add(vector_array)
                        self.index_to_memory_id.append(memory.memory_id)
                    elif not hasattr(self.index, 'is_trained'):
                        self.index.add(vector_array)
                        self.index_to_memory_id.append(memory.memory_id)
                    
                    self.index_metadata["total_vectors"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to add vector to FAISS index: {e}")
            
            logger.debug(f"Stored vector memory: {memory.memory_id}")
            return memory.memory_id
    
    def retrieve(self, query: Union[str, np.ndarray], k: int = 5,
                threshold: float = 0.5) -> List[VectorMemory]:
        """Retrieve similar vectors."""
        with self._lock:
            if isinstance(query, str):
                # Generate query vector from string
                query_memory = VectorMemory(content=query, dimension=self.dimension)
                query_vector = query_memory.vector
            else:
                query_vector = query
            
            if query_vector is None:
                return []
            
            # Use FAISS index if available
            if self.index is not None and len(self.index_to_memory_id) > 0:
                try:
                    return self._faiss_search(query_vector, k, threshold)
                except Exception as e:
                    logger.error(f"FAISS search failed: {e}")
                    # Fall back to linear search
            
            # Linear search fallback
            return self._linear_search(query_vector, k, threshold)
    
    def _faiss_search(self, query_vector: np.ndarray, k: int, 
                     threshold: float) -> List[VectorMemory]:
        """Search using FAISS index."""
        query_array = query_vector.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_array, min(k, len(self.index_to_memory_id)))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.index_to_memory_id):
                memory_id = self.index_to_memory_id[idx]
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    # Convert distance to similarity score
                    similarity = float(distance)  # FAISS returns similarity for IP
                    if similarity >= threshold:
                        memory.update_access()
                        results.append(memory)
        
        return results
    
    def _linear_search(self, query_vector: np.ndarray, k: int, 
                      threshold: float) -> List[VectorMemory]:
        """Linear search through all vectors."""
        similarities = []
        
        for memory in self.memories.values():
            if memory.vector is not None:
                similarity = VectorMemory._cosine_similarity(query_vector, memory.vector)
                if similarity >= threshold:
                    similarities.append((similarity, memory))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for similarity, memory in similarities[:k]:
            memory.update_access()
            results.append(memory)
        
        return results
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update vector memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            
            # If content changed, regenerate vector and update index
            if 'content' in updates:
                memory.vector = memory._generate_embedding()
                self._rebuild_index()  # Rebuild index with new vector
            
            logger.debug(f"Updated vector memory: {memory_id}")
            return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete vector memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            del self.memories[memory_id]
            
            # Remove from index (requires rebuild for FAISS)
            if memory_id in self.index_to_memory_id:
                self._rebuild_index()
            
            logger.debug(f"Deleted vector memory: {memory_id}")
            return True
    
    def clear(self) -> None:
        """Clear all vector memories."""
        with self._lock:
            self.memories.clear()
            self.index_to_memory_id.clear()
            self._initialize_index()
            
            self.index_metadata = {
                "total_vectors": 0,
                "last_rebuild": datetime.now().isoformat(),
                "index_type": self.index_type
            }
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current memories."""
        if self.index is None:
            return
        
        try:
            # Reinitialize index
            self._initialize_index()
            self.index_to_memory_id.clear()
            
            # Add all vectors back
            vectors = []
            memory_ids = []
            
            for memory in self.memories.values():
                if memory.vector is not None:
                    vectors.append(memory.vector)
                    memory_ids.append(memory.memory_id)
            
            if vectors:
                vector_array = np.array(vectors).astype(np.float32)
                
                # Train if needed
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(vectors) >= 100:
                        self.index.train(vector_array)
                
                if hasattr(self.index, 'is_trained') and self.index.is_trained:
                    self.index.add(vector_array)
                    self.index_to_memory_id.extend(memory_ids)
                elif not hasattr(self.index, 'is_trained'):
                    self.index.add(vector_array)
                    self.index_to_memory_id.extend(memory_ids)
            
            self.index_metadata["total_vectors"] = len(memory_ids)
            self.index_metadata["last_rebuild"] = datetime.now().isoformat()
            
            logger.info(f"Rebuilt FAISS index with {len(memory_ids)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        with self._lock:
            total_memories = len(self.memories)
            
            # Calculate dimension distribution
            dimensions = [m.dimension for m in self.memories.values()]
            avg_dimension = np.mean(dimensions) if dimensions else 0
            
            # Calculate access statistics
            access_counts = [m.access_count for m in self.memories.values()]
            avg_access = np.mean(access_counts) if access_counts else 0
            
            return {
                "total_memories": total_memories,
                "indexed_vectors": len(self.index_to_memory_id),
                "average_dimension": float(avg_dimension),
                "average_access_count": float(avg_access),
                "index_metadata": self.index_metadata.copy(),
                "faiss_available": FAISS_AVAILABLE,
                "index_type": self.index_type
            }