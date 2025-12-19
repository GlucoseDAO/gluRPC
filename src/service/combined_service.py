"""
Combined async gRPC + REST service for GluRPC.
Runs both async gRPC (grpc.aio) and FastAPI in the same process, sharing the same core logic and models.

DISCONNECT HANDLING ARCHITECTURE:
---------------------------------
When running both gRPC and REST services simultaneously, each uses its own disconnect detection mechanism:

1. **gRPC Disconnect Detection** (glurpc_service.py):
   - Uses context.cancelled() polling in AsyncGlucosePredictionServicer._create_disconnect_future()
   - Creates a disconnect_future + watcher_task per request
   - Watcher task properly cancelled in finally blocks
   - Works with gRPC streaming context cancellation

2. **REST Disconnect Detection** (middleware.py + app.py):
   - DisconnectMiddleware creates per-request disconnect_event (asyncio.Event)
   - Polls request.is_disconnected() for FastAPI requests
   - link_disconnect_event() bridges event to disconnect_future
   - Watcher task cancelled in request cleanup

3. **Unified Tracking** (state.py):
   - DisconnectTracker singleton manages both gRPC and REST disconnects
   - Per-request futures: individual request cancellation
   - Shared futures: all requests for (handle, index) must disconnect
   - Thread-safe with asyncio.Lock
   - Properly handles duplicate requests with request_id sequencing

ISOLATION:
----------
- gRPC and REST use different request contexts (no interference)
- Each creates independent disconnect_future per request
- DisconnectTracker handles both types transparently
- No shared state between gRPC context.cancelled() and REST request.is_disconnected()

SAFETY:
-------
✓ Watcher tasks are properly tracked and cancelled
✓ Disconnect futures are per-request (no cross-contamination)
✓ Cleanup happens in finally blocks
✓ Both can run simultaneously without conflicts
"""

import logging
import asyncio

import grpc
from grpc import aio
import uvicorn

import service.common
from service.service_spec import glurpc_pb2_grpc
from service.glurpc_service import AsyncGlucosePredictionServicer

# Import the existing FastAPI app from glurpc.app instead of reimplementing it
from glurpc.app import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger("glurpc.combined")


# --- Async gRPC server --------------------------------------------------------

async def create_grpc_server(port: int = 7003) -> aio.Server:
    """
    Create and start async gRPC server.
    
    Args:
        port: Port to bind gRPC service to
        
    Returns:
        Started async gRPC server
    """
    server = aio.server()
    servicer = AsyncGlucosePredictionServicer()
    
    # Initialize servicer (models + background processor)
    await servicer._initialize()
    
    glurpc_pb2_grpc.add_GlucosePredictionServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info(f"Async gRPC server started on port {port}")
    return server


# --- Main entry point ---------------------------------------------------------

async def main_combined(grpc_port: int = 7003, rest_port: int = 8000):
    """Main async entry point to run both gRPC and REST."""
    # Start gRPC server
    grpc_server = await create_grpc_server(port=grpc_port)
    logger.info(f"gRPC server started on :{grpc_port}")

    # Create uvicorn config for REST
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=rest_port,
        log_level="info",
        loop="asyncio"
    )
    rest_server = uvicorn.Server(config)
    
    # Run REST server in background task
    rest_task = asyncio.create_task(rest_server.serve())
    logger.info(f"REST server started on :{rest_port}")

    # Setup signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()
    
    # Register signal handlers
    import signal as signal_module
    loop = asyncio.get_event_loop()
    for sig in (signal_module.SIGTERM, signal_module.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    # Wait for shutdown signal or gRPC termination
    try:
        # Wait for either shutdown signal or gRPC termination
        termination_task = asyncio.create_task(grpc_server.wait_for_termination())
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        done, pending = await asyncio.wait(
            [termination_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received interrupt, shutting down...")
    finally:
        # Graceful shutdown of both servers
        logger.info("Shutting down combined service...")
        await grpc_server.stop(grace=5)
        rest_server.should_exit = True
        try:
            await asyncio.wait_for(rest_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("REST server shutdown timeout, forcing exit")
        logger.info("Combined service shutdown complete")


if __name__ == "__main__":
    """
    Run the combined async gRPC + REST server.
    """
    import typer
    
    cli_app = service.common.create_service_app(__file__)
    
    @cli_app.command()
    def main(
        grpc_port: int = typer.Option(
            cli_app.default_grpc_port,
            "--grpc-port",
            help="Port to bind gRPC service to"
        ),
        rest_port: int = typer.Option(
            cli_app.default_rest_port,
            "--rest-port",
            help="Port to bind REST service to"
        )
    ) -> None:
        """Run the combined async gRPC + REST server."""
        asyncio.run(main_combined(grpc_port=grpc_port, rest_port=rest_port))
    
    cli_app()
