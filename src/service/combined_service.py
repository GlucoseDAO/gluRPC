"""
Combined async gRPC + REST service for GluRPC.
Runs both async gRPC (grpc.aio) and FastAPI in the same process, sharing the same core logic and models.
"""

import sys
import logging
import asyncio
import base64
import statistics
from contextlib import asynccontextmanager
from typing import Optional, List

import grpc
from grpc import aio
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, Query, Request
import uvicorn

import service.common
from service.service_spec import glurpc_pb2_grpc
from service.glurpc_service import AsyncGlucosePredictionServicer

# Import all FastAPI dependencies
from glurpc.config import ENABLE_API_KEYS
from glurpc.core import (
    convert_to_unified_action,
    parse_and_schedule,
    generate_plot_from_handle,
    quick_plot_action,
    verify_api_key
)
from glurpc.engine import ModelManager, BackgroundProcessor, check_queue_overload, TaskRegistry
from glurpc.state import InferenceCache, PlotCache, DisconnectTracker
from glurpc.data_classes import RequestTimeStats
from glurpc.middleware import RequestCounterMiddleware, DisconnectMiddleware
from glurpc.schemas import (
    UnifiedResponse,
    PlotRequest,
    QuickPlotResponse,
    ConvertResponse,
    HealthResponse,
    ProcessRequest,
    CacheManagementResponse,
    RequestMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger("glurpc.combined")


# --- Core business logic (shared between gRPC and REST) -----------------------
# The ModelManager and BackgroundProcessor are singletons, so both services
# will automatically share the same instances


def get_request_time_stats(metrics: RequestMetrics) -> RequestTimeStats:
    """Calculate request time statistics from middleware metrics."""
    request_times: List[float] = metrics.request_times
    if not request_times:
        return RequestTimeStats(
            avg_request_time_ms=0.0,
            median_request_time_ms=0.0,
            min_request_time_ms=0.0,
            max_request_time_ms=0.0,
            total_requests=0
        )

    return RequestTimeStats(
        avg_request_time_ms=statistics.mean(request_times),
        median_request_time_ms=statistics.median(request_times),
        min_request_time_ms=min(request_times),
        max_request_time_ms=max(request_times),
        total_requests=len(request_times)
    )


def link_disconnect_event(
    disconnect_event: Optional[asyncio.Event],
    disconnect_future: Optional[asyncio.Future],
    log_label: str,
) -> Optional[asyncio.Task]:
    """Bridge middleware disconnect_event to an existing future."""
    if not disconnect_event or not disconnect_future:
        return None

    async def _link() -> None:
        try:
            await disconnect_event.wait()
            if not disconnect_future.done():
                disconnect_future.set_result(True)
                logger.info(log_label)
        except asyncio.CancelledError:
            return

    return asyncio.create_task(_link())


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


# --- REST API (FastAPI) -------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown."""
    # Startup
    logger.info("Starting up FastAPI REST server...")
    model_manager = ModelManager()
    bg_processor = BackgroundProcessor()
    
    try:
        await model_manager.initialize()
        await bg_processor.start()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model or start processor on startup: {e}")
        raise e
    yield
    # Shutdown
    logger.info("Shutting down FastAPI REST server...")
    await bg_processor.cancel_all_background_tasks()
    await bg_processor.stop()


app = FastAPI(
    title="GluRPC Combined Service",
    description="Glucose Prediction Service (gRPC + REST)",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(DisconnectMiddleware)
app.add_middleware(RequestCounterMiddleware)


# --- API Key Dependency ---

async def require_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Dependency to verify API key from header."""
    if not ENABLE_API_KEYS:
        return None
    
    if not x_api_key:
        logger.warning("API key missing in request")
        raise HTTPException(status_code=401, detail="API key required")
    
    if not verify_api_key(x_api_key):
        logger.warning(f"Invalid API key provided: {x_api_key[:8]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    logger.info(f"API key verified: {x_api_key[:8]}...")
    return x_api_key


# --- REST Endpoints ---

@app.post("/convert_to_unified", response_model=ConvertResponse)
async def convert_to_unified(file: UploadFile = File(...)):
    logger.info(f"REST: /convert_to_unified - filename={file.filename}")
    try:
        content = await file.read()
        content_base64 = base64.b64encode(content).decode()
        result = await convert_to_unified_action(content_base64)
        if result.error:
            logger.info(f"REST: /convert_to_unified - error={result.error}")
        else:
            logger.info(f"REST: /convert_to_unified - success, csv_length={len(result.csv_content) if result.csv_content else 0}")
        return result
    except Exception as e:
        logger.error(f"REST: Convert failed: {e}")
        return ConvertResponse(error=str(e))


@app.post("/process_unified", response_model=UnifiedResponse)
async def process_unified(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """Process unified CSV and cache for plotting (requires auth)."""
    logger.info(f"REST: /process_unified - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"REST: Rejecting /process_unified due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    result = await parse_and_schedule(request.csv_base64, force_calculate=request.force_calculate)
    if result.error:
        logger.info(f"REST: /process_unified - error={result.error}")
    else:
        logger.info(f"REST: /process_unified - handle={result.handle}, has_warnings={result.warnings.has_warnings}")
    return result


@app.post("/draw_a_plot")
async def draw_a_plot(request: PlotRequest, api_key: str = Depends(require_api_key), http_request: Request = None):
    """Generate plot from cached dataset (requires auth)."""
    logger.info(f"REST: /draw_a_plot - handle={request.handle}, index={request.index}, force={request.force_calculate}")
    
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"REST: Rejecting /draw_a_plot due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    disconnect_tracker = DisconnectTracker()
    request_id = await disconnect_tracker.register_request(request.handle, request.index)
    
    bg_processor = BackgroundProcessor()
    await bg_processor.update_latest_request_id(request.handle, request.index, request_id)
    
    disconnect_future = await disconnect_tracker.get_disconnect_future(request.handle, request.index, request_id)
    disconnect_event = getattr(http_request.state, "disconnect_event", None) if http_request else None
    disconnect_link_task = link_disconnect_event(
        disconnect_event,
        disconnect_future,
        f"Client disconnected: {request.handle[:8]}:{request.index}:{request_id}",
    )
    
    try:
        plot_dict = await generate_plot_from_handle(
            request.handle, 
            request.index, 
            force_calculate=request.force_calculate,
            request_id=request_id,
            disconnect_future=disconnect_future
        )
        logger.info(f"REST: /draw_a_plot - handle={request.handle}, index={request.index}, plot_keys={list(plot_dict.keys())}")
        return plot_dict
    except asyncio.CancelledError:
        logger.info(f"REST: Request cancelled: {request.handle[:8]}:{request.index}:{request_id}")
        await TaskRegistry().cancel_request(request.handle, request.index, request_id, "Client disconnected")
        raise HTTPException(status_code=499, detail="Client closed request")
    except asyncio.TimeoutError:
        logger.info(f"REST: Request timeout: {request.handle[:8]}:{request.index}:{request_id}")
        await TaskRegistry().cancel_request(request.handle, request.index, request_id, "Request timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except ValueError as e:
        logger.info(f"REST: /draw_a_plot - handle={request.handle}, index={request.index}, error={str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"REST: Plot failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if disconnect_link_task:
            disconnect_link_task.cancel()
        await disconnect_tracker.unregister_request(request.handle, request.index, request_id)


@app.post("/quick_plot", response_model=QuickPlotResponse)
async def quick_plot(request: ProcessRequest, api_key: str = Depends(require_api_key), http_request: Request = None):
    """Quick plot: process + plot in one call (requires auth)."""
    logger.info(f"REST: /quick_plot - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"REST: Rejecting /quick_plot due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    import uuid
    request_id = hash(str(uuid.uuid4())) & 0x7FFFFFFF
    
    try:
        disconnect_event = getattr(http_request.state, "disconnect_event", None) if http_request else None
        disconnect_future = None
        disconnect_link_task = None
        if disconnect_event:
            loop = asyncio.get_running_loop()
            disconnect_future = loop.create_future()
            disconnect_link_task = link_disconnect_event(
                disconnect_event,
                disconnect_future,
                f"Client disconnected for quick_plot req_id={request_id}",
            )

        try:
            result = await quick_plot_action(
                request.csv_base64, 
                force_calculate=request.force_calculate,
                request_id=request_id,
                disconnect_future=disconnect_future
            )
            
            if result.error:
                logger.info(f"REST: /quick_plot - error={result.error}")
            else:
                warnings = result.warnings.has_warnings if hasattr(result.warnings, 'has_warnings') else result.warnings.get('has_warnings', False)
                logger.info(f"REST: /quick_plot - success, plot_keys={list(result.plot_data.keys())}, has_warnings={warnings}")
            return result
        finally:
            if disconnect_link_task:
                disconnect_link_task.cancel()

    except asyncio.CancelledError:
        logger.info(f"REST: Quick plot cancelled: req_id={request_id}")
        return QuickPlotResponse(plot_data={}, warnings={}, error="Request cancelled")


@app.post("/cache_management", response_model=CacheManagementResponse)
async def cache_management(
    action: str = Query(..., description="Action to perform: 'flush', 'info', 'delete', 'save', 'load'"),
    handle: Optional[str] = Query(None, description="Handle for delete/load/save operations"),
    api_key: str = Depends(require_api_key)
):
    """Manage cache: flush, info, delete, save, load (requires auth)."""
    logger.info(f"REST: /cache_management - action={action}, handle={handle}")
    inf_cache = InferenceCache()
    plot_cache = PlotCache()
    
    if action == "flush":
        await inf_cache.clear()
        await plot_cache.clear()
        return CacheManagementResponse(
            success=True,
            message="Cache flushed successfully",
            cache_size=0,
            persisted_count=0,
            items_affected=None
        )
    
    elif action == "info":
        size = await inf_cache.get_size()
        plot_size = await plot_cache.get_size()
        return CacheManagementResponse(
            success=True,
            message=f"Cache info retrieved (Inference: {size}, Plots: {plot_size})",
            cache_size=size,
            persisted_count=size + plot_size,
            items_affected=None
        )
    
    elif action == "delete":
        if not handle:
            raise HTTPException(status_code=400, detail="Handle parameter required for delete action")
        
        exists = await inf_cache.contains(handle)
        await inf_cache.delete(handle)
        size = await inf_cache.get_size()
        
        return CacheManagementResponse(
            success=exists,
            message=f"Handle {handle} deleted successfully" if exists else f"Handle {handle} not found in cache",
            cache_size=size,
            persisted_count=size,
            items_affected=1 if exists else 0
        )
    
    elif action == "save":
        size = await inf_cache.get_size()
        return CacheManagementResponse(
            success=True,
            message="Cache is automatically persisted (No-op)",
            cache_size=size,
            persisted_count=size,
            items_affected=0
        )
    
    elif action == "load":
        if not handle:
            raise HTTPException(status_code=400, detail="Handle parameter required for load action")
        
        data = await inf_cache.get(handle)
        size = await inf_cache.get_size()
        
        if data:
            return CacheManagementResponse(
                success=True,
                message=f"Handle {handle} loaded/verified in memory",
                cache_size=size,
                persisted_count=size,
                items_affected=1
            )
        else:
            return CacheManagementResponse(
                success=False,
                message=f"Handle {handle} not found on disk",
                cache_size=size,
                persisted_count=size,
                items_affected=0
            )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}. Valid actions are: flush, info, delete, save, load")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (public, no auth)."""
    logger.info("REST: /health")
    model_manager = ModelManager()
    bg_processor = BackgroundProcessor()
    inf_cache = InferenceCache()
    metrics = getattr(app.state, "request_metrics", RequestMetrics())
    
    stats = model_manager.get_stats()
    calc_stats = bg_processor.get_calc_stats()
    cache_size = await inf_cache.get_size()
    request_stats = get_request_time_stats(metrics)
    
    _, load_status, _, _ = check_queue_overload()
    
    response = HealthResponse(
        status="ok" if model_manager.initialized else "degraded",
        load_status=load_status,
        cache_size=cache_size,
        models_initialized=model_manager.initialized,
        available_priority_models=stats.available_priority_models,
        available_general_models=stats.available_general_models,
        avg_fulfillment_time_ms=stats.avg_fulfillment_time_ms,
        vmem_usage_mb=stats.vmem_usage_mb,
        device=stats.device,
        total_http_requests=metrics.total_http_requests,
        total_http_errors=metrics.total_http_errors,
        avg_request_time_ms=request_stats.avg_request_time_ms,
        median_request_time_ms=request_stats.median_request_time_ms,
        min_request_time_ms=request_stats.min_request_time_ms,
        max_request_time_ms=request_stats.max_request_time_ms,
        inference_requests_by_priority=stats.inference_requests_by_priority,
        total_inference_errors=stats.total_inference_errors,
        total_calc_runs=calc_stats.total_calc_runs,
        total_calc_errors=calc_stats.total_calc_errors,
        inference_queue_size=calc_stats.inference_queue_size,
        inference_queue_capacity=calc_stats.inference_queue_capacity,
        calc_queue_size=calc_stats.calc_queue_size,
        calc_queue_capacity=calc_stats.calc_queue_capacity
    )
    logger.info(f"REST: /health - status={response.status}, load_status={response.load_status}")
    return response


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

    # Wait for gRPC server (main lifecycle)
    try:
        await grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down combined service...")
        await grpc_server.stop(grace=5)
        rest_server.should_exit = True
        await rest_task


if __name__ == "__main__":
    """
    Run the combined async gRPC + REST server.
    """
    parser = service.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(main_combined(grpc_port=args.grpc_port, rest_port=args.rest_port))
