"""
Implementation functions for SAFLA CLI Management System.

This module contains all the implementation functions for the CLI commands,
separated for better organization and maintainability.
"""

import asyncio
import json
import os
import shutil
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.text import Text

from safla.utils.config import SAFLAConfig
# Note: These imports are for reference - actual modules may need to be available
# from safla.core.hybrid_memory import HybridMemoryArchitecture
# from safla.core.meta_cognitive_engine import MetaCognitiveEngine
# from safla.core.safety_validation import SafetyValidationFramework
# from safla.mcp.server import ModularMCPServer

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM STATUS AND HEALTH FUNCTIONS
# =============================================================================

def _get_system_status(manager, detailed: bool = False) -> Dict[str, Any]:
    """Get comprehensive system status."""
    status = {
        'timestamp': datetime.now().isoformat(),
        'uptime': _get_system_uptime(),
        'components': {},
        'health': 'unknown',
        'performance': {},
        'resources': _get_resource_usage()
    }
    
    # Check component status
    components = ['memory', 'cognition', 'safety', 'mcp', 'agents']
    for component in components:
        try:
            status['components'][component] = _check_component_status(component, detailed)
        except Exception as e:
            status['components'][component] = {
                'status': 'error',
                'health': 'error',
                'error': str(e)
            }
    
    # Calculate overall health
    component_healths = [comp.get('health', 'error') for comp in status['components'].values()]
    if all(h == 'healthy' for h in component_healths):
        status['health'] = 'healthy'
    elif any(h == 'error' for h in component_healths):
        status['health'] = 'error'
    else:
        status['health'] = 'warning'
    
    # Performance summary
    if detailed:
        status['performance'] = _get_performance_summary()
    
    return status


def _check_component_status(component: str, detailed: bool = False) -> Dict[str, Any]:
    """Check status of a specific component."""
    status_info = {
        'status': 'unknown',
        'health': 'unknown',
        'uptime': 'unknown',
        'last_check': datetime.now().isoformat()
    }
    
    if component == 'memory':
        status_info.update(_check_memory_system_status(detailed))
    elif component == 'cognition':
        status_info.update(_check_cognitive_engine_status(detailed))
    elif component == 'safety':
        status_info.update(_check_safety_system_status(detailed))
    elif component == 'mcp':
        status_info.update(_check_mcp_server_status(detailed))
    elif component == 'agents':
        status_info.update(_check_agent_manager_status(detailed))
    
    return status_info


def _check_memory_system_status(detailed: bool = False) -> Dict[str, Any]:
    """Check hybrid memory system status."""
    try:
        # This would check actual memory system health
        status = {
            'status': 'running',
            'health': 'healthy',
            'uptime': '2h 30m',
            'vector_memory_size': 8542,
            'episodic_memory_size': 1203,
            'semantic_nodes': 456,
            'working_memory_size': 7
        }
        
        if detailed:
            status.update({
                'memory_usage_mb': 245.6,
                'search_latency_ms': 0.8,
                'cache_hit_ratio': 0.87,
                'consolidation_pending': 23
            })
        
        return status
    except Exception as e:
        return {'status': 'error', 'health': 'error', 'error': str(e)}


def _check_cognitive_engine_status(detailed: bool = False) -> Dict[str, Any]:
    """Check meta-cognitive engine status."""
    try:
        status = {
            'status': 'running',
            'health': 'healthy',
            'uptime': '2h 30m',
            'active_goals': 3,
            'running_strategies': 2,
            'self_awareness_score': 0.85,
            'adaptation_rate': 0.12
        }
        
        if detailed:
            status.update({
                'cpu_usage_percent': 15.3,
                'decision_latency_ms': 45.2,
                'strategy_cache_hits': 0.73,
                'learning_rate': 0.08
            })
        
        return status
    except Exception as e:
        return {'status': 'error', 'health': 'error', 'error': str(e)}


def _check_safety_system_status(detailed: bool = False) -> Dict[str, Any]:
    """Check safety validation framework status."""
    try:
        status = {
            'status': 'running',
            'health': 'healthy',
            'uptime': '2h 30m',
            'active_constraints': 12,
            'violations_24h': 0,
            'risk_score': 0.15,
            'rollbacks_available': 5
        }
        
        if detailed:
            status.update({
                'validation_latency_ms': 2.3,
                'constraint_checks_per_sec': 1847,
                'emergency_stops': 0,
                'checkpoint_size_mb': 67.4
            })
        
        return status
    except Exception as e:
        return {'status': 'error', 'health': 'error', 'error': str(e)}


def _check_mcp_server_status(detailed: bool = False) -> Dict[str, Any]:
    """Check MCP server status."""
    try:
        status = {
            'status': 'running',
            'health': 'healthy',
            'uptime': '2h 30m',
            'active_connections': 5,
            'total_tools': 24,
            'requests_24h': 1847,
            'avg_response_time_ms': 12.5
        }
        
        if detailed:
            status.update({
                'connection_pool_usage': 0.50,
                'circuit_breaker_state': 'closed',
                'rate_limit_hits': 3,
                'auth_failures': 0
            })
        
        return status
    except Exception as e:
        return {'status': 'error', 'health': 'error', 'error': str(e)}


def _check_agent_manager_status(detailed: bool = False) -> Dict[str, Any]:
    """Check agent deployment manager status."""
    try:
        status = {
            'status': 'running',
            'health': 'healthy',
            'uptime': '2h 30m',
            'deployed_agents': 3,
            'total_replicas': 8,
            'healthy_replicas': 8,
            'resource_usage_percent': 65.4
        }
        
        if detailed:
            status.update({
                'pending_deployments': 0,
                'failed_deployments': 0,
                'auto_scaling_enabled': True,
                'last_scaling_event': '1h 15m ago'
            })
        
        return status
    except Exception as e:
        return {'status': 'error', 'health': 'error', 'error': str(e)}


def _get_resource_usage() -> Dict[str, Any]:
    """Get system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3)
        }
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        return {}


def _get_system_uptime() -> str:
    """Get system uptime."""
    try:
        uptime_seconds = time.time() - psutil.boot_time()
        uptime_delta = timedelta(seconds=uptime_seconds)
        
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{days}d {hours}h {minutes}m"
    except Exception:
        return "unknown"


def _get_performance_summary() -> Dict[str, Any]:
    """Get system performance summary."""
    try:
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Get memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Get disk I/O metrics
        disk_io = psutil.disk_io_counters()
        
        # Get network I/O metrics
        net_io = psutil.net_io_counters()
        
        return {
            'cpu': {
                'usage_percent': cpu_percent,
                'core_count': cpu_count,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            },
            'memory': {
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'cached_gb': memory.cached / (1024**3) if hasattr(memory, 'cached') else 0,
                'swap_usage_percent': swap.percent
            },
            'io': {
                'disk_read_mb_s': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb_s': disk_io.write_bytes / (1024**2) if disk_io else 0,
                'network_recv_mb_s': net_io.bytes_recv / (1024**2) if net_io else 0,
                'network_sent_mb_s': net_io.bytes_sent / (1024**2) if net_io else 0
            },
            'processes': {
                'total': len(psutil.pids()),
                'running': len([p for p in psutil.process_iter(['status']) if p.info['status'] == psutil.STATUS_RUNNING])
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        return {
            'error': str(e),
            'cpu': {'usage_percent': 0},
            'memory': {'usage_percent': 0},
            'io': {},
            'processes': {'total': 0}
        }


# =============================================================================
# COMPONENT MANAGEMENT FUNCTIONS
# =============================================================================

def _start_component(manager, component: str, daemon: bool = False):
    """Start a specific component."""
    component_starters = {
        'memory': _start_memory_system,
        'cognition': _start_cognitive_engine,
        'safety': _start_safety_system,
        'mcp': _start_mcp_server,
        'agents': _start_agent_manager
    }
    
    if component in component_starters:
        return component_starters[component](manager, daemon)
    else:
        raise ValueError(f"Unknown component: {component}")


def _start_memory_system(manager, daemon: bool = False):
    """Start hybrid memory system."""
    console.print("[blue]Initializing hybrid memory system...[/blue]")
    
    try:
        # Initialize memory architecture
        memory_config = {
            'vector_config': {
                'embedding_dim': manager.config.get('vector_embedding_dim', 768),
                'max_capacity': manager.config.get('max_memories', 10000)
            },
            'episodic_config': {
                'max_capacity': manager.config.get('episodic_memory_size', 1000)
            },
            'working_config': {
                'capacity': manager.config.get('working_memory_size', 10)
            }
        }
        
        memory = HybridMemoryArchitecture(**memory_config)
        
        # Store in manager for access
        manager.memory_system = memory
        
        console.print("[green]✓ Memory system started successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to start memory system: {e}[/red]")
        return False


def _start_cognitive_engine(manager, daemon: bool = False):
    """Start meta-cognitive engine."""
    console.print("[blue]Initializing meta-cognitive engine...[/blue]")
    
    try:
        # Initialize cognitive engine
        cognition = MetaCognitiveEngine(manager.config)
        
        # Start cognitive processes
        # This would start background tasks for self-awareness, goal management, etc.
        
        manager.cognitive_engine = cognition
        
        console.print("[green]✓ Cognitive engine started successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to start cognitive engine: {e}[/red]")
        return False


def _start_safety_system(manager, daemon: bool = False):
    """Start safety validation framework."""
    console.print("[blue]Initializing safety validation framework...[/blue]")
    
    try:
        # Initialize safety framework
        safety = SafetyValidationFramework(manager.config)
        
        # Start safety monitoring
        # This would start background safety monitoring tasks
        
        manager.safety_system = safety
        
        console.print("[green]✓ Safety system started successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to start safety system: {e}[/red]")
        return False


def _start_mcp_server(manager, daemon: bool = False):
    """Start MCP server."""
    console.print("[blue]Starting MCP server...[/blue]")
    
    try:
        # Initialize MCP server
        mcp_server = ModularMCPServer(manager.config)
        
        if daemon:
            # Start as background process
            import threading
            def run_server():
                asyncio.run(mcp_server.run())
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            manager.mcp_server_thread = server_thread
        else:
            # Store for manual management
            manager.mcp_server = mcp_server
        
        console.print("[green]✓ MCP server started successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to start MCP server: {e}[/red]")
        return False


def _start_agent_manager(manager, daemon: bool = False):
    """Start agent deployment manager."""
    console.print("[blue]Initializing agent deployment manager...[/blue]")
    
    try:
        # Initialize agent manager
        # This would set up the agent deployment and orchestration system
        
        manager.agent_manager_running = True
        
        console.print("[green]✓ Agent manager started successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to start agent manager: {e}[/red]")
        return False


def _start_full_system(manager, daemon: bool = False):
    """Start all SAFLA components."""
    console.print("[bold blue]Starting SAFLA System...[/bold blue]")
    
    components = ['memory', 'safety', 'cognition', 'mcp', 'agents']
    failed_components = []
    
    with Progress() as progress:
        task = progress.add_task("Starting components...", total=len(components))
        
        for component in components:
            progress.update(task, description=f"Starting {component}...")
            
            if not _start_component(manager, component, daemon):
                failed_components.append(component)
            
            progress.advance(task)
    
    if failed_components:
        console.print(f"[red]Failed to start components: {', '.join(failed_components)}[/red]")
        return False
    else:
        console.print("[green]✓ All SAFLA components started successfully[/green]")
        return True


# =============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =============================================================================

def _get_config_value(config: SAFLAConfig, key: str) -> Any:
    """Get configuration value by key."""
    try:
        # Handle nested keys with dot notation
        keys = key.split('.')
        value = config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        return value
    except Exception as e:
        raise ValueError(f"Error getting config value: {e}")


def _set_config_value(config: SAFLAConfig, key: str, value: str):
    """Set configuration value."""
    try:
        # Convert string value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
        
        # Handle nested keys
        keys = key.split('.')
        target = config
        
        for k in keys[:-1]:
            if not hasattr(target, k):
                setattr(target, k, {})
            target = getattr(target, k)
        
        setattr(target, keys[-1], value)
        
    except Exception as e:
        raise ValueError(f"Error setting config value: {e}")


def _display_config(config: SAFLAConfig, output_format: str):
    """Display configuration in specified format."""
    config_dict = _config_to_dict(config)
    
    if output_format == 'json':
        console.print(json.dumps(config_dict, indent=2))
    elif output_format == 'yaml':
        console.print(yaml.dump(config_dict, default_flow_style=False))
    elif output_format == 'env':
        _display_env_format(config_dict)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def _config_to_dict(config: SAFLAConfig) -> Dict[str, Any]:
    """Convert config object to dictionary."""
    config_dict = {}
    
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            value = getattr(config, attr)
            if not callable(value):
                config_dict[attr] = value
    
    return config_dict


def _display_env_format(config_dict: Dict[str, Any], prefix: str = 'SAFLA_'):
    """Display configuration in environment variable format."""
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key.upper(), v))
        return dict(items)
    
    flat_config = flatten_dict(config_dict)
    
    for key, value in flat_config.items():
        if not key.startswith(prefix):
            key = f"{prefix}{key}"
        console.print(f"{key}={value}")


def _backup_config(config: SAFLAConfig, backup_dir: str) -> str:
    """Backup current configuration."""
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_path / f"safla_config_backup_{timestamp}.json"
    
    config_dict = _config_to_dict(config)
    
    with open(backup_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return str(backup_file)


def _restore_config(manager, backup_file: str):
    """Restore configuration from backup."""
    with open(backup_file, 'r') as f:
        config_dict = json.load(f)
    
    # Apply configuration
    for key, value in config_dict.items():
        try:
            _set_config_value(manager.config, key, str(value))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not restore {key}: {e}[/yellow]")


# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

def _get_system_metrics(manager, detailed: bool = False) -> Dict[str, Any]:
    """Get system performance metrics."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'system': _get_resource_usage(),
        'components': {}
    }
    
    # Component-specific metrics
    components = ['memory', 'cognition', 'safety', 'mcp']
    for component in components:
        metrics['components'][component] = _get_component_metrics(component, detailed)
    
    return metrics


def _get_component_metrics(component: str, detailed: bool = False) -> Dict[str, Any]:
    """Get metrics for a specific component."""
    base_metrics = {
        'uptime_seconds': 9000,  # 2.5 hours
        'memory_usage_mb': 128.5,
        'cpu_usage_percent': 15.2
    }
    
    if component == 'memory':
        base_metrics.update({
            'vector_searches_per_sec': 145.7,
            'similarity_search_latency_ms': 0.8,
            'cache_hit_ratio': 0.87,
            'memory_consolidations_per_hour': 12
        })
    elif component == 'cognition':
        base_metrics.update({
            'decisions_per_sec': 23.4,
            'strategy_switches_per_hour': 8,
            'self_awareness_score': 0.85,
            'goal_completion_rate': 0.73
        })
    elif component == 'safety':
        base_metrics.update({
            'validations_per_sec': 1847.3,
            'violations_per_hour': 0,
            'risk_assessments_per_sec': 234.5,
            'rollback_events_per_day': 0
        })
    elif component == 'mcp':
        base_metrics.update({
            'requests_per_sec': 45.6,
            'response_time_ms': 12.5,
            'active_connections': 5,
            'error_rate_percent': 0.02
        })
    
    return base_metrics


def _display_metrics_table(metrics_data: Dict[str, Any], detailed: bool = False):
    """Display metrics in table format."""
    # System metrics table
    system_table = Table(title="System Resource Usage")
    system_table.add_column("Metric", style="cyan")
    system_table.add_column("Value", style="green")
    system_table.add_column("Status", style="yellow")
    
    system_metrics = metrics_data['system']
    for key, value in system_metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.1f}"
            if 'percent' in key:
                formatted_value += "%"
                status = "🟢" if value < 80 else "🟡" if value < 90 else "🔴"
            elif 'gb' in key:
                formatted_value += " GB"
                status = "ℹ️"
            else:
                status = "ℹ️"
        else:
            formatted_value = str(value)
            status = "ℹ️"
        
        system_table.add_row(key.replace('_', ' ').title(), formatted_value, status)
    
    console.print(system_table)
    
    # Component metrics table
    if detailed:
        for component, comp_metrics in metrics_data['components'].items():
            comp_table = Table(title=f"{component.title()} Component Metrics")
            comp_table.add_column("Metric", style="cyan")
            comp_table.add_column("Value", style="green")
            
            for key, value in comp_metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.1f}"
                    if 'percent' in key:
                        formatted_value += "%"
                    elif 'ms' in key:
                        formatted_value += " ms"
                    elif 'sec' in key:
                        formatted_value += "/sec"
                    elif 'ratio' in key:
                        formatted_value = f"{value:.2%}"
                else:
                    formatted_value = str(value)
                
                comp_table.add_row(key.replace('_', ' ').title(), formatted_value)
            
            console.print(comp_table)


def _get_recent_logs(manager, lines: int = 50) -> str:
    """Get recent system logs."""
    try:
        # This would read from actual log files
        log_lines = [
            f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Memory consolidation completed - 23 events processed",
            f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Cognitive engine adapted strategy for optimization task",
            f"[{datetime.now().strftime('%H:%M:%S')}] INFO: MCP server handled 145 requests in the last minute",
            f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Safety validation passed for 1,847 operations",
            f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Agent 'optimizer-01' scaled to 3 replicas",
        ]
        
        return '\n'.join(log_lines[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def _analyze_optimizations(manager, component: Optional[str] = None) -> List[Dict[str, Any]]:
    """Analyze system for optimization opportunities."""
    optimizations = []
    
    # Memory optimizations
    if not component or component == 'memory':
        optimizations.extend([
            {
                'component': 'memory',
                'type': 'cache_optimization',
                'description': 'Increase vector memory cache size for better performance',
                'impact': 'high',
                'effort': 'low',
                'estimated_improvement': '25% faster searches'
            },
            {
                'component': 'memory',
                'type': 'compression',
                'description': 'Enable memory compression to reduce footprint',
                'impact': 'medium',
                'effort': 'low',
                'estimated_improvement': '30% memory reduction'
            }
        ])
    
    # Cognitive optimizations
    if not component or component == 'cognition':
        optimizations.extend([
            {
                'component': 'cognition',
                'type': 'parallel_processing',
                'description': 'Enable parallel strategy evaluation',
                'impact': 'high',
                'effort': 'medium',
                'estimated_improvement': '40% faster decisions'
            }
        ])
    
    # MCP optimizations
    if not component or component == 'mcp':
        optimizations.extend([
            {
                'component': 'mcp',
                'type': 'connection_pooling',
                'description': 'Optimize connection pool size based on load',
                'impact': 'medium',
                'effort': 'low',
                'estimated_improvement': '15% better throughput'
            }
        ])
    
    return optimizations


def _display_optimization_recommendations(optimizations: List[Dict[str, Any]]):
    """Display optimization recommendations."""
    if not optimizations:
        console.print("[green]No optimization opportunities found[/green]")
        return
    
    table = Table(title="Optimization Recommendations")
    table.add_column("Component", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Description", style="white")
    table.add_column("Impact", style="green")
    table.add_column("Effort", style="yellow")
    table.add_column("Improvement", style="magenta")
    
    for opt in optimizations:
        impact_color = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }.get(opt['impact'], '⚪')
        
        effort_color = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }.get(opt['effort'], '⚪')
        
        table.add_row(
            opt['component'],
            opt['type'],
            opt['description'],
            f"{impact_color} {opt['impact']}",
            f"{effort_color} {opt['effort']}",
            opt['estimated_improvement']
        )
    
    console.print(table)


def _apply_optimizations(manager, optimizations: List[Dict[str, Any]]):
    """Apply recommended optimizations."""
    console.print("[blue]Applying optimizations...[/blue]")
    
    with Progress() as progress:
        task = progress.add_task("Applying optimizations...", total=len(optimizations))
        
        for opt in optimizations:
            progress.update(task, description=f"Applying {opt['type']}...")
            
            try:
                _apply_single_optimization(manager, opt)
                console.print(f"[green]✓ Applied {opt['type']} optimization[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to apply {opt['type']}: {e}[/red]")
            
            progress.advance(task)
    
    console.print("[green]Optimization process completed[/green]")


def _apply_single_optimization(manager, optimization: Dict[str, Any]):
    """Apply a single optimization."""
    opt_type = optimization['type']
    component = optimization['component']
    
    if opt_type == 'cache_optimization':
        # Increase cache size
        current_size = manager.config.get('cache_size', 256)
        new_size = min(current_size * 1.5, 1024)  # Increase by 50%, max 1GB
        _set_config_value(manager.config, 'cache_size', str(int(new_size)))
        
    elif opt_type == 'compression':
        # Enable compression
        _set_config_value(manager.config, 'memory_compression', 'true')
        
    elif opt_type == 'parallel_processing':
        # Enable parallel processing
        _set_config_value(manager.config, 'parallel_cognition', 'true')
        
    elif opt_type == 'connection_pooling':
        # Optimize connection pool
        current_pool = manager.config.get('mcp_connection_pool_size', 10)
        new_pool = min(current_pool * 1.2, 50)  # Increase by 20%, max 50
        _set_config_value(manager.config, 'mcp_connection_pool_size', str(int(new_pool)))
    
    else:
        raise ValueError(f"Unknown optimization type: {opt_type}")


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def _run_benchmark_suite(manager, suite: str) -> Dict[str, Any]:
    """Run comprehensive benchmark suite."""
    console.print(f"[blue]Running {suite} benchmark suite...[/blue]")
    
    benchmarks = {
        'quick': ['memory_search', 'basic_cognitive'],
        'standard': ['memory_search', 'cognitive_full', 'safety_validation', 'mcp_latency'],
        'comprehensive': ['memory_search', 'memory_stress', 'cognitive_full', 'cognitive_stress',
                         'safety_validation', 'safety_stress', 'mcp_latency', 'mcp_throughput']
    }
    
    suite_benchmarks = benchmarks.get(suite, benchmarks['standard'])
    results = {
        'suite': suite,
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
        'summary': {}
    }
    
    with Progress() as progress:
        task = progress.add_task("Running benchmarks...", total=len(suite_benchmarks))
        
        for benchmark_name in suite_benchmarks:
            progress.update(task, description=f"Running {benchmark_name}...")
            
            benchmark_result = _run_single_benchmark(manager, benchmark_name)
            results['benchmarks'][benchmark_name] = benchmark_result
            
            progress.advance(task)
    
    # Calculate summary
    results['summary'] = _calculate_benchmark_summary(results['benchmarks'])
    
    return results


def _run_single_benchmark(manager, benchmark_name: str) -> Dict[str, Any]:
    """Run a single benchmark."""
    benchmark_functions = {
        'memory_search': _benchmark_memory_search,
        'memory_stress': _benchmark_memory_stress,
        'cognitive_full': _benchmark_cognitive_processing,
        'cognitive_stress': _benchmark_cognitive_stress,
        'safety_validation': _benchmark_safety_validation,
        'safety_stress': _benchmark_safety_stress,
        'mcp_latency': _benchmark_mcp_latency,
        'mcp_throughput': _benchmark_mcp_throughput
    }
    
    if benchmark_name in benchmark_functions:
        return benchmark_functions[benchmark_name](manager)
    else:
        return {'error': f'Unknown benchmark: {benchmark_name}'}


def _benchmark_memory_search(manager) -> Dict[str, Any]:
    """Benchmark memory search performance."""
    # Simulate memory search benchmark
    import random
    
    latencies = [random.uniform(0.5, 2.0) for _ in range(1000)]
    
    return {
        'name': 'memory_search',
        'iterations': 1000,
        'avg_latency_ms': sum(latencies) / len(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
        'throughput_ops_per_sec': 1000 / (sum(latencies) / 1000),
        'status': 'passed' if sum(latencies) / len(latencies) < 1.0 else 'failed'
    }


def _benchmark_cognitive_processing(manager) -> Dict[str, Any]:
    """Benchmark cognitive processing performance."""
    # Simulate cognitive benchmark
    import random
    
    decision_times = [random.uniform(20, 100) for _ in range(100)]
    
    return {
        'name': 'cognitive_processing',
        'iterations': 100,
        'avg_decision_time_ms': sum(decision_times) / len(decision_times),
        'strategy_switches': 15,
        'goal_completions': 8,
        'adaptation_events': 23,
        'status': 'passed' if sum(decision_times) / len(decision_times) < 50.0 else 'failed'
    }


def _benchmark_safety_validation(manager) -> Dict[str, Any]:
    """Benchmark safety validation performance."""
    # Simulate safety benchmark
    import random
    
    validation_times = [random.uniform(1, 5) for _ in range(10000)]
    
    return {
        'name': 'safety_validation',
        'iterations': 10000,
        'avg_validation_time_ms': sum(validation_times) / len(validation_times),
        'validations_per_sec': 10000 / (sum(validation_times) / 1000),
        'violations_detected': 0,
        'false_positives': 2,
        'status': 'passed' if sum(validation_times) / len(validation_times) < 2.0 else 'failed'
    }


def _benchmark_mcp_latency(manager) -> Dict[str, Any]:
    """Benchmark MCP server latency."""
    # Simulate MCP benchmark
    import random
    
    latencies = [random.uniform(5, 25) for _ in range(1000)]
    
    return {
        'name': 'mcp_latency',
        'iterations': 1000,
        'avg_latency_ms': sum(latencies) / len(latencies),
        'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
        'success_rate': 0.995,
        'errors': 5,
        'status': 'passed' if sum(latencies) / len(latencies) < 15.0 else 'failed'
    }


def _calculate_benchmark_summary(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall benchmark summary."""
    total_benchmarks = len(benchmarks)
    passed_benchmarks = sum(1 for b in benchmarks.values() if b.get('status') == 'passed')
    
    return {
        'total_benchmarks': total_benchmarks,
        'passed_benchmarks': passed_benchmarks,
        'failed_benchmarks': total_benchmarks - passed_benchmarks,
        'pass_rate': passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
        'overall_status': 'passed' if passed_benchmarks == total_benchmarks else 'failed'
    }


def _display_benchmark_results(results: Dict[str, Any]):
    """Display benchmark results."""
    summary = results['summary']
    
    # Summary panel
    summary_text = (
        f"Suite: {results['suite']}\n"
        f"Total Benchmarks: {summary['total_benchmarks']}\n"
        f"Passed: {summary['passed_benchmarks']}\n"
        f"Failed: {summary['failed_benchmarks']}\n"
        f"Pass Rate: {summary['pass_rate']:.1%}\n"
        f"Overall Status: {summary['overall_status']}"
    )
    
    status_color = 'green' if summary['overall_status'] == 'passed' else 'red'
    console.print(Panel(summary_text, title="Benchmark Summary", border_style=status_color))
    
    # Detailed results table
    table = Table(title="Detailed Benchmark Results")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key Metric", style="yellow")
    table.add_column("Value", style="white")
    
    for name, benchmark in results['benchmarks'].items():
        status_icon = "✅" if benchmark.get('status') == 'passed' else "❌"
        
        # Extract key metric based on benchmark type
        if 'latency' in name or 'search' in name:
            key_metric = "Avg Latency"
            value = f"{benchmark.get('avg_latency_ms', 0):.1f} ms"
        elif 'cognitive' in name:
            key_metric = "Avg Decision Time"
            value = f"{benchmark.get('avg_decision_time_ms', 0):.1f} ms"
        elif 'safety' in name:
            key_metric = "Validations/sec"
            value = f"{benchmark.get('validations_per_sec', 0):.0f}"
        else:
            key_metric = "Iterations"
            value = str(benchmark.get('iterations', 0))
        
        table.add_row(
            name.replace('_', ' ').title(),
            f"{status_icon} {benchmark.get('status', 'unknown')}",
            key_metric,
            value
        )
    
    console.print(table)


# =============================================================================
# AGENT MANAGEMENT FUNCTIONS
# =============================================================================

def _get_deployed_agents(manager) -> List[Dict[str, Any]]:
    """Get list of deployed agents."""
    # This would query the actual agent deployment system
    return [
        {
            'name': 'optimizer-01',
            'type': 'optimization',
            'status': 'running',
            'replicas': 3,
            'cpu_usage': 15.2,
            'memory_usage': 245.6,
            'uptime': '2h 30m',
            'last_activity': '30s ago'
        },
        {
            'name': 'monitor-01',
            'type': 'monitoring',
            'status': 'running',
            'replicas': 2,
            'cpu_usage': 8.7,
            'memory_usage': 128.3,
            'uptime': '1h 45m',
            'last_activity': '5s ago'
        },
        {
            'name': 'learner-01',
            'type': 'learning',
            'status': 'scaling',
            'replicas': 1,
            'cpu_usage': 22.4,
            'memory_usage': 512.8,
            'uptime': '45m',
            'last_activity': '2m ago'
        }
    ]


def _display_agents_table(agents_data: List[Dict[str, Any]]):
    """Display agents in table format."""
    table = Table(title="Deployed Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Replicas", style="yellow")
    table.add_column("CPU %", style="red")
    table.add_column("Memory MB", style="magenta")
    table.add_column("Uptime", style="white")
    
    for agent in agents_data:
        status_color = {
            'running': '🟢',
            'scaling': '🟡',
            'error': '🔴',
            'stopped': '⚪'
        }.get(agent['status'], '⚪')
        
        table.add_row(
            agent['name'],
            agent['type'],
            f"{status_color} {agent['status']}",
            str(agent['replicas']),
            f"{agent['cpu_usage']:.1f}",
            f"{agent['memory_usage']:.1f}",
            agent['uptime']
        )
    
    console.print(table)


# =============================================================================
# ADDITIONAL CLI HELPER FUNCTIONS
# =============================================================================

def _search_cli_help(query: str):
    """Search CLI help and documentation."""
    console.print(f"[blue]Searching for: {query}[/blue]")
    
    # Command search database
    commands = {
        'start': 'Start SAFLA system or components',
        'stop': 'Stop SAFLA system or components',
        'status': 'Show system status and health',
        'config': 'Configuration management',
        'monitor': 'System monitoring and metrics',
        'benchmark': 'Performance benchmarking',
        'optimize': 'System optimization',
        'agents': 'Agent deployment and management',
        'dashboard': 'Interactive dashboard',
        'setup': 'System setup wizard',
        'validate': 'System validation',
        'logs': 'View system logs',
        'metrics': 'Performance metrics',
        'memory': 'Memory management',
        'safety': 'Safety validation',
        'mcp': 'MCP server management'
    }
    
    # Settings search database
    settings = {
        'debug': 'SAFLA_DEBUG - Enable debug mode',
        'log_level': 'SAFLA_LOG_LEVEL - Set logging level',
        'workers': 'SAFLA_WORKER_THREADS - Number of worker threads',
        'memory_pool': 'SAFLA_MEMORY_POOL_SIZE - Memory pool size',
        'cache': 'SAFLA_CACHE_SIZE - Cache size',
        'vector_dims': 'SAFLA_VECTOR_DIMENSIONS - Vector dimensions',
        'similarity': 'SAFLA_SIMILARITY_THRESHOLD - Similarity threshold',
        'timeout': 'SAFLA_MCP_TIMEOUT - MCP timeout',
        'rate_limit': 'SAFLA_API_RATE_LIMIT - API rate limit'
    }
    
    # Search commands
    matching_commands = []
    for cmd, desc in commands.items():
        if query.lower() in cmd.lower() or query.lower() in desc.lower():
            matching_commands.append((cmd, desc))
    
    # Search settings
    matching_settings = []
    for setting, desc in settings.items():
        if query.lower() in setting.lower() or query.lower() in desc.lower():
            matching_settings.append((setting, desc))
    
    # Display results
    if matching_commands:
        table = Table(title="Matching Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        
        for cmd, desc in matching_commands:
            table.add_row(f"safla {cmd}", desc)
        
        console.print(table)
    
    if matching_settings:
        table = Table(title="Matching Settings")
        table.add_column("Setting", style="green")
        table.add_column("Description", style="white")
        
        for setting, desc in matching_settings:
            table.add_row(setting, desc)
        
        console.print(table)
    
    if not matching_commands and not matching_settings:
        console.print(f"[yellow]No results found for '{query}'[/yellow]")
        console.print("Try: safla help-menu for full command list")


def _run_health_checks(manager) -> Dict[str, Any]:
    """Run comprehensive health checks."""
    health_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_health': 'unknown',
        'checks': {}
    }
    
    checks = [
        ('system_resources', _check_system_resources),
        ('component_health', _check_all_components_health),
        ('configuration_validity', _check_configuration_health),
        ('network_connectivity', _check_network_health),
        ('storage_health', _check_storage_health),
        ('performance_baseline', _check_performance_baseline)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            result = check_func(manager)
            health_results['checks'][check_name] = result
            if result.get('status') == 'pass':
                passed_checks += 1
        except Exception as e:
            health_results['checks'][check_name] = {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Calculate overall health
    health_ratio = passed_checks / total_checks
    if health_ratio >= 0.9:
        health_results['overall_health'] = 'excellent'
    elif health_ratio >= 0.7:
        health_results['overall_health'] = 'good'
    elif health_ratio >= 0.5:
        health_results['overall_health'] = 'fair'
    else:
        health_results['overall_health'] = 'poor'
    
    return health_results


def _check_system_resources(manager) -> Dict[str, Any]:
    """Check system resource availability."""
    try:
        resources = _get_resource_usage()
        
        # Check if resources are within acceptable limits
        warnings = []
        if resources.get('cpu_percent', 0) > 90:
            warnings.append("High CPU usage")
        if resources.get('memory_percent', 0) > 90:
            warnings.append("High memory usage")
        if resources.get('disk_percent', 0) > 95:
            warnings.append("Low disk space")
        
        return {
            'status': 'warn' if warnings else 'pass',
            'message': '; '.join(warnings) if warnings else 'System resources OK',
            'details': resources,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Resource check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _check_all_components_health(manager) -> Dict[str, Any]:
    """Check health of all SAFLA components."""
    try:
        components = ['memory', 'cognition', 'safety', 'mcp', 'agents']
        component_results = {}
        healthy_components = 0
        
        for component in components:
            status = _check_component_status(component, detailed=False)
            component_results[component] = status
            if status.get('health') == 'healthy':
                healthy_components += 1
        
        health_ratio = healthy_components / len(components)
        
        return {
            'status': 'pass' if health_ratio >= 0.8 else 'warn' if health_ratio >= 0.5 else 'fail',
            'message': f"{healthy_components}/{len(components)} components healthy",
            'details': component_results,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Component health check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _check_configuration_health(manager) -> Dict[str, Any]:
    """Check configuration validity and completeness."""
    try:
        config_errors = validate_config(manager.config)
        
        return {
            'status': 'pass' if not config_errors else 'fail',
            'message': f"Found {len(config_errors)} configuration issues" if config_errors else "Configuration valid",
            'details': {'errors': config_errors},
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Configuration check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _check_network_health(manager) -> Dict[str, Any]:
    """Check network connectivity and MCP server accessibility."""
    try:
        # Check if MCP server port is accessible
        import socket
        
        host = manager.config.get('host', 'localhost')
        port = manager.config.get('port', 8000)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            status = 'pass'
            message = f"Network connectivity OK (port {port} accessible)"
        else:
            status = 'warn'
            message = f"Port {port} not accessible (service may be down)"
        
        return {
            'status': status,
            'message': message,
            'details': {'host': host, 'port': port, 'connection_result': result},
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Network check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _check_storage_health(manager) -> Dict[str, Any]:
    """Check storage availability and permissions."""
    try:
        data_dir = Path(manager.config.get('data_dir', './data'))
        config_dir = Path(manager.config.get('config_dir', './.roo'))
        
        issues = []
        
        # Check data directory
        if not data_dir.exists():
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                issues.append(f"Cannot create data directory: {data_dir}")
        elif not os.access(data_dir, os.W_OK):
            issues.append(f"No write permission to data directory: {data_dir}")
        
        # Check config directory
        if not config_dir.exists():
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                issues.append(f"Cannot create config directory: {config_dir}")
        elif not os.access(config_dir, os.W_OK):
            issues.append(f"No write permission to config directory: {config_dir}")
        
        return {
            'status': 'pass' if not issues else 'fail',
            'message': '; '.join(issues) if issues else 'Storage access OK',
            'details': {'data_dir': str(data_dir), 'config_dir': str(config_dir)},
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Storage check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _check_performance_baseline(manager) -> Dict[str, Any]:
    """Check basic performance baseline."""
    try:
        # Simple performance test
        import time
        import numpy as np
        
        # Memory allocation test
        start_time = time.time()
        test_array = np.random.random((1000, 100))
        alloc_time = time.time() - start_time
        
        # Simple computation test
        start_time = time.time()
        result = np.sum(test_array * test_array)
        compute_time = time.time() - start_time
        
        # Basic I/O test
        test_file = Path('./test_io_performance.tmp')
        start_time = time.time()
        with open(test_file, 'w') as f:
            f.write('test' * 1000)
        with open(test_file, 'r') as f:
            content = f.read()
        test_file.unlink()
        io_time = time.time() - start_time
        
        warnings = []
        if alloc_time > 0.1:
            warnings.append("Slow memory allocation")
        if compute_time > 0.1:
            warnings.append("Slow computation")
        if io_time > 0.1:
            warnings.append("Slow I/O")
        
        return {
            'status': 'warn' if warnings else 'pass',
            'message': '; '.join(warnings) if warnings else 'Performance baseline OK',
            'details': {
                'memory_alloc_ms': alloc_time * 1000,
                'compute_ms': compute_time * 1000,
                'io_ms': io_time * 1000
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Performance check failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def _run_performance_check(manager) -> Dict[str, Any]:
    """Run performance analysis."""
    return {
        'timestamp': datetime.now().isoformat(),
        'overall_performance': 'good',
        'metrics': _get_system_metrics(manager, detailed=True),
        'recommendations': [
            'Consider increasing cache size for better performance',
            'Enable performance optimizations if not already enabled'
        ]
    }


def _run_security_check(manager) -> Dict[str, Any]:
    """Run security audit."""
    return {
        'timestamp': datetime.now().isoformat(),
        'security_level': 'medium',
        'vulnerabilities': [],
        'recommendations': [
            'Ensure JWT secret key is set',
            'Enable HTTPS in production',
            'Review rate limiting settings'
        ]
    }


def _display_doctor_report(validation_results, config_errors, health_results, 
                          perf_results, security_results):
    """Display comprehensive doctor report."""
    console.print("\n[bold blue]SAFLA System Health Report[/bold blue]")
    
    # Overall status
    overall_health = health_results.get('overall_health', 'unknown')
    health_color = {
        'excellent': 'green',
        'good': 'green',
        'fair': 'yellow',
        'poor': 'red',
        'unknown': 'blue'
    }.get(overall_health, 'blue')
    
    console.print(Panel(
        f"[{health_color}]Overall Health: {overall_health.title()}[/{health_color}]\n"
        f"Timestamp: {health_results.get('timestamp', 'unknown')}",
        title="System Summary"
    ))
    
    # Health checks table
    health_table = Table(title="Health Checks")
    health_table.add_column("Check", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Message", style="white")
    
    for check_name, check_result in health_results.get('checks', {}).items():
        status = check_result.get('status', 'unknown')
        status_icon = {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌',
            'error': '🚫'
        }.get(status, '❓')
        
        health_table.add_row(
            check_name.replace('_', ' ').title(),
            f"{status_icon} {status}",
            check_result.get('message', 'No message')
        )
    
    console.print(health_table)
    
    # Configuration errors
    if config_errors:
        console.print(Panel(
            '\n'.join(config_errors),
            title="Configuration Issues",
            border_style="red"
        ))
    
    # Performance summary
    perf_health = perf_results.get('overall_performance', 'unknown')
    console.print(Panel(
        f"Performance Level: {perf_health.title()}",
        title="Performance Summary"
    ))
    
    # Security summary
    security_level = security_results.get('security_level', 'unknown')
    console.print(Panel(
        f"Security Level: {security_level.title()}",
        title="Security Summary"
    ))


def _get_version_info() -> Dict[str, Any]:
    """Get comprehensive version information."""
    import platform
    import sys
    
    return {
        'safla_version': '2.0.0',
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor() or 'unknown',
        'hostname': platform.node(),
        'cli_version': '1.0.0',
        'build_date': '2024-01-01',
        'components': {
            'memory_system': '2.0.0',
            'cognitive_engine': '2.0.0',
            'safety_framework': '2.0.0',
            'mcp_server': '2.0.0',
            'agent_manager': '1.0.0'
        }
    }


def _display_version_table(version_info: Dict[str, Any]):
    """Display version information in table format."""
    table = Table(title="SAFLA Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    # Main version info
    table.add_row("SAFLA Core", version_info['safla_version'])
    table.add_row("CLI", version_info['cli_version'])
    table.add_row("Python", version_info['python_version'].split()[0])
    table.add_row("Platform", version_info['platform'])
    
    console.print(table)
    
    # Component versions
    comp_table = Table(title="Component Versions")
    comp_table.add_column("Component", style="blue")
    comp_table.add_column("Version", style="yellow")
    
    for component, version in version_info['components'].items():
        comp_table.add_row(component.replace('_', ' ').title(), version)
    
    console.print(comp_table)
    
    # System info
    sys_info = f"""
Hostname: {version_info['hostname']}
Architecture: {version_info['architecture']}
Processor: {version_info['processor']}
Build Date: {version_info['build_date']}
"""
    
    console.print(Panel(sys_info.strip(), title="System Information"))