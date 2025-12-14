import os.path
import time
import asyncio
from typing import Optional, Callable, Any

import typer

from service import registry


def get_service_defaults(script_name: str) -> tuple[int, int]:
    """Get default ports for a service from the registry."""
    service_name = os.path.splitext(os.path.basename(script_name))[0]
    grpc_port = registry[service_name]['grpc']
    rest_port = registry.get(service_name, {}).get('rest', 8000)
    return grpc_port, rest_port


def create_service_app(script_name: str) -> typer.Typer:
    """Create a typer app with common service options.
    
    Args:
        script_name: Name of the service script to get default ports from registry
        
    Returns:
        Configured typer.Typer instance
    """
    app = typer.Typer()
    
    # Store defaults for use by service commands
    default_grpc, default_rest = get_service_defaults(script_name)
    app.info = typer.main.TyperInfo(
        name=os.path.splitext(os.path.basename(script_name))[0]
    )
    app.default_grpc_port = default_grpc
    app.default_rest_port = default_rest
    
    return app


class ServiceArgs:
    """Container for service arguments to maintain compatibility with existing code."""
    
    def __init__(self, grpc_port: int, rest_port: int):
        self.grpc_port = grpc_port
        self.rest_port = rest_port


async def main_loop(grpc_handler: Callable, args: ServiceArgs) -> None:
    """
    Start async gRPC server and run until interrupted.
    
    Args:
        grpc_handler: Async function that creates and starts the gRPC server
        args: ServiceArgs object with grpc_port and rest_port
    """
    server = await grpc_handler(port=args.grpc_port)
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(grace=5)

