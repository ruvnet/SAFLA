"""
System management handlers for SAFLA MCP server.

This module provides tools for system validation, configuration,
and general system information retrieval.
"""

import platform
import psutil
import time
from typing import Any, Dict, List, Optional
import logging

from .base import BaseHandler, ToolDefinition
from safla.utils.validation import validate_installation, check_gpu_availability

logger = logging.getLogger(__name__)


class SystemHandler(BaseHandler):
    """Handler for system management tools."""
    
    def _initialize_tools(self) -> None:
        """Initialize system management tools."""
        tools = [
            ToolDefinition(
                name="validate_installation",
                description="Validate SAFLA installation and dependencies",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_validate_installation"
            ),
            ToolDefinition(
                name="get_system_info",
                description="Get comprehensive system information",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_get_system_info"
            ),
            ToolDefinition(
                name="check_gpu_status",
                description="Check GPU availability and status",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_check_gpu_status"
            ),
            ToolDefinition(
                name="get_config_summary",
                description="Get current SAFLA configuration summary",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_get_config_summary"
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    async def _validate_installation(self) -> Dict[str, Any]:
        """Validate SAFLA installation and dependencies."""
        try:
            validation_results = validate_installation()
            
            # Check critical components
            critical_checks = {
                "core_modules": self._check_core_modules(),
                "vector_store": self._check_vector_store(),
                "ml_models": self._check_ml_models(),
                "mcp_connectivity": await self._check_mcp_connectivity()
            }
            
            return {
                "status": "healthy" if all(critical_checks.values()) else "unhealthy",
                "validation_results": validation_results,
                "critical_checks": critical_checks,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Installation validation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            # CPU information
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "per_cpu_percent": psutil.cpu_percent(interval=1, percpu=True)
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "free": memory.free
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
            
            # Network information
            net_io = psutil.net_io_counters()
            network_info = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
            
            # Process information
            current_process = psutil.Process()
            process_info = {
                "pid": current_process.pid,
                "cpu_percent": current_process.cpu_percent(),
                "memory_percent": current_process.memory_percent(),
                "memory_info": current_process.memory_info()._asdict(),
                "num_threads": current_process.num_threads()
            }
            
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
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info,
                "process": process_info,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}")
            raise
    
    async def _check_gpu_status(self) -> Dict[str, Any]:
        """Check GPU availability and status."""
        try:
            gpu_info = check_gpu_availability()
            gpu_available = gpu_info.get("cuda_available", False)

            result = {
                "gpu_available": gpu_available,
                "gpu_info": gpu_info,
                "timestamp": time.time()
            }

            # Additional GPU checks if available
            if gpu_available:
                try:
                    import torch
                    result["pytorch_cuda"] = {
                        "available": torch.cuda.is_available(),
                        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    }
                except ImportError:
                    result["pytorch_cuda"] = {"available": False, "error": "PyTorch not installed"}
            
            return result
        except Exception as e:
            logger.error(f"GPU status check failed: {str(e)}")
            return {
                "gpu_available": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _get_config_summary(self) -> Dict[str, Any]:
        """Get current SAFLA configuration summary."""
        try:
            config_dict = self._serialize_config(self.config)
            
            # Add computed values
            config_dict["computed"] = {
                "total_memory_tiers": len(getattr(self.config.memory, "tiers", [])),
                "ml_models_configured": len(getattr(self.config.ml_models, "models", {})),
                "mcp_agents_configured": len(getattr(self.config.mcp, "agents", [])),
                "safety_constraints_active": getattr(self.config.safety, "enabled", False)
            }
            
            return {
                "config": config_dict,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get config summary: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_core_modules(self) -> bool:
        """Check if core SAFLA modules are importable."""
        core_modules = [
            "safla.core.hybrid_memory",
            "safla.core.meta_cognitive_engine",
            "safla.core.mcp_orchestration",
            "safla.core.safety_validation",
            "safla.core.delta_evaluation"
        ]
        
        for module in core_modules:
            try:
                __import__(module)
            except ImportError:
                logger.error(f"Core module not available: {module}")
                return False
        return True
    
    def _check_vector_store(self) -> bool:
        """Check if vector store is operational."""
        try:
            # Simple check - try to import vector store components
            from safla.core.ml_neural_embedding_engine import MLNeuralEmbeddingEngine
            return True
        except Exception:
            return False
    
    def _check_ml_models(self) -> bool:
        """Check if ML models are available."""
        try:
            # Check if model files exist or can be loaded
            # This is a placeholder - implement actual model checking
            return True
        except Exception:
            return False
    
    async def _check_mcp_connectivity(self) -> bool:
        """Check MCP connectivity."""
        try:
            # Simple connectivity check
            # In real implementation, would test actual MCP protocols
            return True
        except Exception:
            return False
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Serialize config object to dictionary."""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._serialize_config(value)
                elif isinstance(value, (list, tuple)):
                    result[key] = [self._serialize_config(item) if hasattr(item, '__dict__') else item for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: self._serialize_config(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                else:
                    result[key] = value
            return result
        return config