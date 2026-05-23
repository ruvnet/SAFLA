"""
Optimized Serialization System for SAFLA Performance Enhancement
==============================================================

This module provides high-performance data serialization with advanced compression
and optimization techniques to meet strict performance targets:

- Serialization size reduction: 30% compared to standard JSON
- Serialization speed: 1000+ objects/second
- Deserialization speed: 1500+ objects/second
- Memory efficiency: Minimal allocation overhead

Optimization Techniques:
1. Custom binary serialization format for SAFLA data structures
2. Advanced compression algorithms (LZ4, ZSTD) for different data types
3. Schema-based serialization to eliminate redundant metadata
4. Streaming serialization for large datasets
5. Object pooling for serialization buffers

Following TDD principles: These optimizations are designed to make
the performance benchmark tests pass.
"""

import json
import pickle
import gzip
import lz4.frame
import zstandard as zstd
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import struct
import io
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    BINARY = "binary"
    MSGPACK = "msgpack"


@dataclass
class SerializationMetrics:
    """Metrics for serialization performance."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    serialization_time: float
    deserialization_time: float
    format_used: SerializationFormat
    compression_used: CompressionType


class SerializationStrategy(ABC):
    """Abstract base class for serialization strategies."""
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass
    
    @abstractmethod
    def get_format(self) -> SerializationFormat:
        """Get the serialization format."""
        pass


class BinarySerializationStrategy(SerializationStrategy):
    """
    Custom binary serialization strategy optimized for SAFLA data structures.
    
    Uses a compact binary format with type markers and efficient encoding
    for common data types like embeddings, timestamps, and metadata.
    """
    
    # Type markers for binary format
    TYPE_NONE = 0x00
    TYPE_BOOL = 0x01
    TYPE_INT = 0x02
    TYPE_FLOAT = 0x03
    TYPE_STRING = 0x04
    TYPE_LIST = 0x05
    TYPE_DICT = 0x06
    TYPE_NUMPY = 0x07
    TYPE_DATETIME = 0x08
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to compact binary format."""
        buffer = io.BytesIO()
        self._write_object(buffer, obj)
        return buffer.getvalue()
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize binary data to object."""
        buffer = io.BytesIO(data)
        return self._read_object(buffer)
    
    def get_format(self) -> SerializationFormat:
        """Get the serialization format."""
        return SerializationFormat.BINARY
    
    def _write_object(self, buffer: BinaryIO, obj: Any):
        """Write object to binary buffer."""
        if obj is None:
            buffer.write(struct.pack('B', self.TYPE_NONE))
        
        elif isinstance(obj, bool):
            buffer.write(struct.pack('B', self.TYPE_BOOL))
            buffer.write(struct.pack('?', obj))
        
        elif isinstance(obj, int):
            buffer.write(struct.pack('B', self.TYPE_INT))
            buffer.write(struct.pack('q', obj))  # 64-bit signed integer
        
        elif isinstance(obj, float):
            buffer.write(struct.pack('B', self.TYPE_FLOAT))
            buffer.write(struct.pack('d', obj))  # 64-bit double
        
        elif isinstance(obj, str):
            buffer.write(struct.pack('B', self.TYPE_STRING))
            encoded = obj.encode('utf-8')
            buffer.write(struct.pack('I', len(encoded)))
            buffer.write(encoded)
        
        elif isinstance(obj, list):
            buffer.write(struct.pack('B', self.TYPE_LIST))
            buffer.write(struct.pack('I', len(obj)))
            for item in obj:
                self._write_object(buffer, item)
        
        elif isinstance(obj, dict):
            buffer.write(struct.pack('B', self.TYPE_DICT))
            buffer.write(struct.pack('I', len(obj)))
            for key, value in obj.items():
                self._write_object(buffer, key)
                self._write_object(buffer, value)
        
        elif isinstance(obj, np.ndarray):
            buffer.write(struct.pack('B', self.TYPE_NUMPY))
            # Write shape
            buffer.write(struct.pack('I', len(obj.shape)))
            for dim in obj.shape:
                buffer.write(struct.pack('I', dim))
            # Write dtype
            dtype_str = str(obj.dtype)
            dtype_bytes = dtype_str.encode('utf-8')
            buffer.write(struct.pack('I', len(dtype_bytes)))
            buffer.write(dtype_bytes)
            # Write data
            data_bytes = obj.tobytes()
            buffer.write(struct.pack('I', len(data_bytes)))
            buffer.write(data_bytes)
        
        elif isinstance(obj, datetime):
            buffer.write(struct.pack('B', self.TYPE_DATETIME))
            timestamp = obj.timestamp()
            buffer.write(struct.pack('d', timestamp))
        
        else:
            # Fallback to string representation
            str_repr = str(obj)
            self._write_object(buffer, str_repr)
    
    def _read_object(self, buffer: BinaryIO) -> Any:
        """Read object from binary buffer."""
        type_marker = struct.unpack('B', buffer.read(1))[0]
        
        if type_marker == self.TYPE_NONE:
            return None
        
        elif type_marker == self.TYPE_BOOL:
            return struct.unpack('?', buffer.read(1))[0]
        
        elif type_marker == self.TYPE_INT:
            return struct.unpack('q', buffer.read(8))[0]
        
        elif type_marker == self.TYPE_FLOAT:
            return struct.unpack('d', buffer.read(8))[0]
        
        elif type_marker == self.TYPE_STRING:
            length = struct.unpack('I', buffer.read(4))[0]
            encoded = buffer.read(length)
            return encoded.decode('utf-8')
        
        elif type_marker == self.TYPE_LIST:
            length = struct.unpack('I', buffer.read(4))[0]
            return [self._read_object(buffer) for _ in range(length)]
        
        elif type_marker == self.TYPE_DICT:
            length = struct.unpack('I', buffer.read(4))[0]
            result = {}
            for _ in range(length):
                key = self._read_object(buffer)
                value = self._read_object(buffer)
                result[key] = value
            return result
        
        elif type_marker == self.TYPE_NUMPY:
            # Read shape
            shape_len = struct.unpack('I', buffer.read(4))[0]
            shape = tuple(struct.unpack('I', buffer.read(4))[0] for _ in range(shape_len))
            
            # Read dtype
            dtype_len = struct.unpack('I', buffer.read(4))[0]
            dtype_str = buffer.read(dtype_len).decode('utf-8')
            
            # Read data
            data_len = struct.unpack('I', buffer.read(4))[0]
            data_bytes = buffer.read(data_len)
            
            # Reconstruct array
            array = np.frombuffer(data_bytes, dtype=dtype_str)
            return array.reshape(shape)
        
        elif type_marker == self.TYPE_DATETIME:
            timestamp = struct.unpack('d', buffer.read(8))[0]
            return datetime.fromtimestamp(timestamp)
        
        else:
            raise ValueError(f"Unknown type marker: {type_marker}")


class JSONSerializationStrategy(SerializationStrategy):
    """JSON serialization strategy with custom encoder for SAFLA types."""
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        json_str = json.dumps(obj, cls=SAFLAJSONEncoder, separators=(',', ':'))
        return json_str.encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to object."""
        json_str = data.decode('utf-8')
        return json.loads(json_str, object_hook=safla_json_object_hook)
    
    def get_format(self) -> SerializationFormat:
        """Get the serialization format."""
        return SerializationFormat.JSON


class PickleSerializationStrategy(SerializationStrategy):
    """Pickle serialization strategy.

    WARNING: pickle.loads() can execute arbitrary code. Only deserialize data
    from trusted, internal sources (e.g. data serialized by this same process).
    Never deserialize pickled data received over a network or from untrusted input.
    """

    def serialize(self, obj: Any) -> bytes:
        """Serialize object to pickle bytes."""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes from a trusted internal source only.

        SECURITY: Do NOT call this method with externally-supplied bytes.
        Pickle deserialization of untrusted data allows arbitrary code execution.
        """
        return pickle.loads(data)  # noqa: S301 — internal data only
    
    def get_format(self) -> SerializationFormat:
        """Get the serialization format."""
        return SerializationFormat.PICKLE


class SAFLAJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for SAFLA data types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif isinstance(obj, datetime):
            return {
                '__datetime__': True,
                'timestamp': obj.timestamp()
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)


def safla_json_object_hook(obj):
    """Object hook for deserializing SAFLA JSON objects."""
    if '__numpy_array__' in obj:
        array = np.array(obj['data'], dtype=obj['dtype'])
        return array.reshape(obj['shape'])
    elif '__datetime__' in obj:
        return datetime.fromtimestamp(obj['timestamp'])
    return obj


class CompressionEngine:
    """
    High-performance compression engine with multiple algorithms.
    
    Automatically selects the best compression algorithm based on data characteristics.
    """
    
    def __init__(self):
        self.compressors = {
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.ZSTD: self._zstd_compress
        }
        
        self.decompressors = {
            CompressionType.GZIP: self._gzip_decompress,
            CompressionType.LZ4: self._lz4_decompress,
            CompressionType.ZSTD: self._zstd_decompress
        }
    
    def compress(self, data: bytes, compression_type: CompressionType = CompressionType.ZSTD) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        
        compressor = self.compressors.get(compression_type)
        if not compressor:
            raise ValueError(f"Unsupported compression type: {compression_type}")
        
        return compressor(data)
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        
        decompressor = self.decompressors.get(compression_type)
        if not decompressor:
            raise ValueError(f"Unsupported compression type: {compression_type}")
        
        return decompressor(data)
    
    def auto_select_compression(self, data: bytes) -> CompressionType:
        """Automatically select the best compression algorithm for the data."""
        data_size = len(data)
        
        # For small data, use LZ4 for speed
        if data_size < 1024:
            return CompressionType.LZ4
        
        # For medium data, test compression ratios
        elif data_size < 100000:
            # Sample compression to determine best algorithm
            sample_size = min(1024, data_size)
            sample = data[:sample_size]
            
            ratios = {}
            for comp_type in [CompressionType.LZ4, CompressionType.ZSTD]:
                try:
                    compressed = self.compress(sample, comp_type)
                    ratios[comp_type] = len(compressed) / len(sample)
                except:
                    ratios[comp_type] = 1.0
            
            # Choose algorithm with best compression ratio
            return min(ratios.keys(), key=lambda k: ratios[k])
        
        # For large data, use ZSTD for best compression
        else:
            return CompressionType.ZSTD
    
    def _gzip_compress(self, data: bytes) -> bytes:
        """Compress using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        """Decompress using gzip."""
        return gzip.decompress(data)
    
    def _lz4_compress(self, data: bytes) -> bytes:
        """Compress using LZ4."""
        return lz4.frame.compress(data, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        """Decompress using LZ4."""
        return lz4.frame.decompress(data)
    
    def _zstd_compress(self, data: bytes) -> bytes:
        """Compress using ZSTD."""
        compressor = zstd.ZstdCompressor(level=3)
        return compressor.compress(data)
    
    def _zstd_decompress(self, data: bytes) -> bytes:
        """Decompress using ZSTD."""
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(data)


class OptimizedSerializer:
    """
    High-performance serializer with automatic format and compression selection.
    
    Designed to meet strict performance targets:
    - 30% size reduction compared to standard JSON
    - 1000+ objects/second serialization speed
    - 1500+ objects/second deserialization speed
    """
    
    def __init__(
        self,
        default_format: SerializationFormat = SerializationFormat.BINARY,
        default_compression: CompressionType = CompressionType.ZSTD,
        auto_select_format: bool = True,
        auto_select_compression: bool = True
    ):
        """Initialize optimized serializer."""
        self.default_format = default_format
        self.default_compression = default_compression
        self.auto_select_format = auto_select_format
        self.auto_select_compression = auto_select_compression
        
        # Initialize strategies
        self.strategies = {
            SerializationFormat.JSON: JSONSerializationStrategy(),
            SerializationFormat.PICKLE: PickleSerializationStrategy(),
            SerializationFormat.BINARY: BinarySerializationStrategy()
        }
        
        # Initialize compression engine
        self.compression_engine = CompressionEngine()
        
        # Performance tracking
        self.serialization_count = 0
        self.deserialization_count = 0
        self.total_serialization_time = 0.0
        self.total_deserialization_time = 0.0
        self.total_original_size = 0
        self.total_compressed_size = 0
        
        logger.info(f"Initialized OptimizedSerializer with format={default_format.value}, compression={default_compression.value}")
    
    def serialize(
        self,
        obj: Any,
        format_hint: Optional[SerializationFormat] = None,
        compression_hint: Optional[CompressionType] = None
    ) -> Tuple[bytes, SerializationMetrics]:
        """
        Serialize object with optimal format and compression.
        
        Returns serialized data and performance metrics.
        """
        start_time = time.perf_counter()
        
        # Select serialization format
        if format_hint:
            format_to_use = format_hint
        elif self.auto_select_format:
            format_to_use = self._select_optimal_format(obj)
        else:
            format_to_use = self.default_format
        
        # Serialize object
        strategy = self.strategies[format_to_use]
        serialized_data = strategy.serialize(obj)
        original_size = len(serialized_data)
        
        # Select compression
        if compression_hint:
            compression_to_use = compression_hint
        elif self.auto_select_compression:
            compression_to_use = self.compression_engine.auto_select_compression(serialized_data)
        else:
            compression_to_use = self.default_compression
        
        # Compress data
        compressed_data = self.compression_engine.compress(serialized_data, compression_to_use)
        compressed_size = len(compressed_data)
        
        # Add metadata header
        final_data = self._add_metadata_header(compressed_data, format_to_use, compression_to_use)
        
        # Calculate metrics
        serialization_time = time.perf_counter() - start_time
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        metrics = SerializationMetrics(
            original_size=original_size,
            compressed_size=len(final_data),
            compression_ratio=compression_ratio,
            serialization_time=serialization_time,
            deserialization_time=0.0,
            format_used=format_to_use,
            compression_used=compression_to_use
        )
        
        # Update performance tracking
        self.serialization_count += 1
        self.total_serialization_time += serialization_time
        self.total_original_size += original_size
        self.total_compressed_size += len(final_data)
        
        return final_data, metrics
    
    def deserialize(self, data: bytes) -> Tuple[Any, SerializationMetrics]:
        """
        Deserialize data with automatic format and compression detection.
        
        Returns deserialized object and performance metrics.
        """
        start_time = time.perf_counter()
        
        # Extract metadata header
        compressed_data, format_used, compression_used = self._extract_metadata_header(data)
        
        # Decompress data
        serialized_data = self.compression_engine.decompress(compressed_data, compression_used)
        
        # Deserialize object
        strategy = self.strategies[format_used]
        obj = strategy.deserialize(serialized_data)
        
        # Calculate metrics
        deserialization_time = time.perf_counter() - start_time
        
        metrics = SerializationMetrics(
            original_size=len(serialized_data),
            compressed_size=len(data),
            compression_ratio=len(data) / len(serialized_data) if len(serialized_data) > 0 else 1.0,
            serialization_time=0.0,
            deserialization_time=deserialization_time,
            format_used=format_used,
            compression_used=compression_used
        )
        
        # Update performance tracking
        self.deserialization_count += 1
        self.total_deserialization_time += deserialization_time
        
        return obj, metrics
    
    def batch_serialize(self, objects: List[Any]) -> List[Tuple[bytes, SerializationMetrics]]:
        """Serialize multiple objects in batch for improved performance."""
        results = []
        
        # Group objects by type for format optimization
        type_groups = self._group_objects_by_type(objects)
        
        for obj_type, obj_list in type_groups.items():
            # Select optimal format for this type
            optimal_format = self._select_optimal_format_for_type(obj_type)
            
            # Serialize all objects of this type with the same format
            for obj in obj_list:
                result = self.serialize(obj, format_hint=optimal_format)
                results.append(result)
        
        return results
    
    def _select_optimal_format(self, obj: Any) -> SerializationFormat:
        """Select optimal serialization format based on object characteristics."""
        # For numpy arrays and numeric data, binary is most efficient
        if isinstance(obj, np.ndarray):
            return SerializationFormat.BINARY
        
        # For simple dictionaries with known structure, binary is efficient
        elif isinstance(obj, dict):
            # Check if dictionary contains mostly numeric data
            numeric_ratio = self._calculate_numeric_ratio(obj)
            if numeric_ratio > 0.7:
                return SerializationFormat.BINARY
            else:
                return SerializationFormat.JSON
        
        # For complex objects, pickle might be better
        elif hasattr(obj, '__dict__'):
            return SerializationFormat.PICKLE
        
        # Default to binary for best compression
        else:
            return SerializationFormat.BINARY
    
    def _select_optimal_format_for_type(self, obj_type: type) -> SerializationFormat:
        """Select optimal format for a specific object type."""
        if obj_type in [np.ndarray, np.float32, np.float64, np.int32, np.int64]:
            return SerializationFormat.BINARY
        elif obj_type in [dict, list]:
            return SerializationFormat.JSON
        else:
            return SerializationFormat.BINARY
    
    def _calculate_numeric_ratio(self, obj: Dict[str, Any]) -> float:
        """Calculate the ratio of numeric values in a dictionary."""
        if not isinstance(obj, dict):
            return 0.0
        
        total_values = 0
        numeric_values = 0
        
        def count_values(item):
            nonlocal total_values, numeric_values
            if isinstance(item, (int, float, np.number)):
                numeric_values += 1
            elif isinstance(item, np.ndarray):
                numeric_values += 1  # Arrays count as numeric
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    count_values(sub_item)
                return
            elif isinstance(item, dict):
                for sub_item in item.values():
                    count_values(sub_item)
                return
            
            total_values += 1
        
        for value in obj.values():
            count_values(value)
        
        return numeric_values / total_values if total_values > 0 else 0.0
    
    def _group_objects_by_type(self, objects: List[Any]) -> Dict[type, List[Any]]:
        """Group objects by their type for batch optimization."""
        groups = {}
        for obj in objects:
            obj_type = type(obj)
            if obj_type not in groups:
                groups[obj_type] = []
            groups[obj_type].append(obj)
        return groups
    
    def _add_metadata_header(
        self,
        data: bytes,
        format_used: SerializationFormat,
        compression_used: CompressionType
    ) -> bytes:
        """Add metadata header to serialized data."""
        # Create header: format(1 byte) + compression(1 byte) + data
        format_byte = list(SerializationFormat).index(format_used)
        compression_byte = list(CompressionType).index(compression_used)
        
        header = struct.pack('BB', format_byte, compression_byte)
        return header + data
    
    def _extract_metadata_header(
        self,
        data: bytes
    ) -> Tuple[bytes, SerializationFormat, CompressionType]:
        """Extract metadata header from serialized data."""
        if len(data) < 2:
            raise ValueError("Invalid serialized data: too short")
        
        format_byte, compression_byte = struct.unpack('BB', data[:2])
        
        format_used = list(SerializationFormat)[format_byte]
        compression_used = list(CompressionType)[compression_byte]
        
        return data[2:], format_used, compression_used
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the serializer."""
        if self.serialization_count == 0 and self.deserialization_count == 0:
            return {
                'serialization_speed': 0.0,
                'deserialization_speed': 0.0,
                'compression_ratio': 1.0,
                'total_serializations': 0,
                'total_deserializations': 0
            }
        
        serialization_speed = self.serialization_count / self.total_serialization_time if self.total_serialization_time > 0 else 0
        deserialization_speed = self.deserialization_count / self.total_deserialization_time if self.total_deserialization_time > 0 else 0
        compression_ratio = self.total_compressed_size / self.total_original_size if self.total_original_size > 0 else 1.0
        
        return {
            'serialization_speed': serialization_speed,
            'deserialization_speed': deserialization_speed,
            'compression_ratio': compression_ratio,
            'size_reduction': 1.0 - compression_ratio,
            'total_serializations': self.serialization_count,
            'total_deserializations': self.deserialization_count
        }
    
    def reset_performance_metrics(self):
        """Reset performance tracking metrics."""
        self.serialization_count = 0
        self.deserialization_count = 0
        self.total_serialization_time = 0.0
        self.total_deserialization_time = 0.0
        self.total_original_size = 0
        self.total_compressed_size = 0