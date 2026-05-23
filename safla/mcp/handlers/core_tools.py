"""Core system tool handlers for SAFLA MCP Server"""

import asyncio
import platform
import psutil
import time
from typing import Dict, Any
from safla.utils.validation import validate_installation, check_gpu_availability
from safla.utils.config import get_config
from safla.mcp.handler_registry import register_handler
import logging

logger = logging.getLogger(__name__)


@register_handler(
    "validate_installation",
    description="Validate SAFLA installation and configuration",
    category="core",
    requires_auth=False
)
async def validate_installation_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate SAFLA installation"""
    try:
        validation_result = validate_installation()
        
        return {
            "valid": validation_result["valid"],
            "system_info": validation_result.get("system_info", {}),
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", [])
        }
    except Exception as e:
        return {"error": str(e), "valid": False}


@register_handler(
    "get_system_info",
    description="Get system information including hardware and OS details",
    category="core",
    requires_auth=False
)
async def get_system_info_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Get system information"""
    try:
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        memory_info = psutil.virtual_memory()
        
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent": usage.percent
                })
            except PermissionError:
                continue
        
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "cpu": cpu_info,
            "memory": {
                "total_gb": memory_info.total / (1024**3),
                "available_gb": memory_info.available / (1024**3),
                "used_gb": memory_info.used / (1024**3),
                "percent": memory_info.percent
            },
            "disk": disk_info,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e)}


@register_handler(
    "check_gpu_status",
    description="Check GPU availability and status",
    category="core",
    requires_auth=False
)
async def check_gpu_status_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Check GPU status"""
    try:
        gpu_info = check_gpu_availability()
        
        if gpu_info.get("cuda_available", False):
            # Try to get detailed GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_details = []
                    for i in range(torch.cuda.device_count()):
                        gpu_details.append({
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                            "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                            "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                        })
                    gpu_info["devices"] = gpu_details
            except ImportError:
                pass
        
        return gpu_info
    except Exception as e:
        return {"error": str(e), "available": False}


@register_handler(
    "get_config_summary",
    description="Get current SAFLA configuration summary",
    category="core",
    requires_auth=True
)
async def get_config_summary_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Get configuration summary"""
    try:
        config = get_config()
        
        # Serialize config object
        def serialize_config(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        result[key] = serialize_config(value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [serialize_config(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize_config(value) for key, value in obj.items()}
            else:
                return obj
        
        config_dict = serialize_config(config)
        
        # Remove sensitive information
        if 'openai_api_key' in config_dict:
            config_dict['openai_api_key'] = "***" if config_dict['openai_api_key'] else None
        if 'anthropic_api_key' in config_dict:
            config_dict['anthropic_api_key'] = "***" if config_dict['anthropic_api_key'] else None
        
        return {
            "config": config_dict,
            "environment": config.environment,
            "version": getattr(config, 'version', '1.0.0')
        }
    except Exception as e:
        return {"error": str(e)}