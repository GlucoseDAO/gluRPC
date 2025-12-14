import base64
import logging
import signal
import sys
import statistics
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, Query, Request
from fastapi.responses import Response, JSONResponse

# Dependencies from glurpc
from glurpc.config import (
    ENABLE_API_KEYS, 
    LOG_LEVEL_ROOT, LOG_LEVEL_LOGIC, LOG_LEVEL_ENGINE,
    LOG_LEVEL_CORE, LOG_LEVEL_APP, LOG_LEVEL_STATE, LOG_LEVEL_CACHE,
    LOG_LEVEL_LOCKS
)
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

# Setup logging with configured levels
# Pad logger names to 18 chars and level names to 8 chars for aligned output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s'
)

# Configure logger levels from environment
logging.getLogger("glurpc").setLevel(LOG_LEVEL_ROOT)
logging.getLogger("glurpc.logic").setLevel(LOG_LEVEL_LOGIC)
logging.getLogger("glurpc.engine").setLevel(LOG_LEVEL_ENGINE)
logging.getLogger("glurpc.core").setLevel(LOG_LEVEL_CORE)
logging.getLogger("glurpc.app").setLevel(LOG_LEVEL_APP)
logging.getLogger("glurpc.state").setLevel(LOG_LEVEL_STATE)
logging.getLogger("glurpc.cache").setLevel(LOG_LEVEL_CACHE)
logging.getLogger("glurpc.locks").setLevel(LOG_LEVEL_LOCKS)  # App-wide lock logger

logger = logging.getLogger("glurpc.app")

def get_request_time_stats(metrics: RequestMetrics) -> RequestTimeStats:
    """
    Calculate request time statistics from middleware metrics.
    """
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
    """
    Bridge middleware disconnect_event to an existing future.
    Returns a watcher task or None if inputs are missing.
    """
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up GluRPC server...")
    logger.info(f"Logger levels: ROOT={logging.getLevelName(LOG_LEVEL_ROOT)}, LOGIC={logging.getLevelName(LOG_LEVEL_LOGIC)}, ENGINE={logging.getLevelName(LOG_LEVEL_ENGINE)}, CORE={logging.getLevelName(LOG_LEVEL_CORE)}, APP={logging.getLevelName(LOG_LEVEL_APP)}, STATE={logging.getLevelName(LOG_LEVEL_STATE)}, CACHE={logging.getLevelName(LOG_LEVEL_CACHE)}, LOCKS={logging.getLevelName(LOG_LEVEL_LOCKS)}")
    model_manager = ModelManager()
    bg_processor = BackgroundProcessor()
    
    try:
        await model_manager.initialize()
        await bg_processor.start()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model or start processor on startup: {e}")
        raise e # Model failure is critical, terminate app
    yield
    # Shutdown
    logger.info("Shutting down GluRPC server...")
    await bg_processor.cancel_all_background_tasks()
    await bg_processor.stop()

app = FastAPI(
    title="GluRPC",
    description="Glucose Prediction Service",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(DisconnectMiddleware)
app.add_middleware(RequestCounterMiddleware)

# --- API Key Dependency ---

async def require_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Dependency to verify API key from header."""
    if not ENABLE_API_KEYS:
        # API keys disabled, allow all requests
        return None
    
    if not x_api_key:
        logger.warning("API key missing in request")
        raise HTTPException(status_code=401, detail="API key required")
    
    if not verify_api_key(x_api_key):
        logger.warning(f"Invalid API key provided: {x_api_key[:8]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    logger.info(f"API key verified: {x_api_key[:8]}...")
    return x_api_key

# --- Endpoints ---

@app.post("/convert_to_unified", response_model=ConvertResponse)
async def convert_to_unified(file: UploadFile = File(...)):
    logger.info(f"Request: /convert_to_unified - filename={file.filename}, content_type={file.content_type}")
    try:
        content = await file.read()
        content_base64 = base64.b64encode(content).decode()
        logger.debug (f"In /convert_to_unified got={len(content)} bytes, decoded to {len(content_base64)} bytes")
        result = await convert_to_unified_action(content_base64)
        if result.error:
            logger.info(f"Response: /convert_to_unified - error={result.error}")
        else:
            logger.info(f"Response: /convert_to_unified - success, csv_length={len(result.csv_content) if result.csv_content else 0}")
        return result
    except Exception as e:
        logger.error(f"Convert failed: {e}")
        return ConvertResponse(error=str(e))

@app.post("/process_unified", response_model=UnifiedResponse)
async def process_unified(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """
    Upload a CSV (base64 encoded) to process and cache for plotting.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /process_unified - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    
    # DEBUG: Check for truncated data and dump
    expected_size = 2310888  # Expected full size
    if len(request.csv_base64) < expected_size:
        try:
            import tempfile
            import datetime
            import base64
            import os
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = os.path.join(tempfile.gettempdir(), "glurpc_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            logger.error(f"REST: TRUNCATED DATA DETECTED! Expected {expected_size}, got {len(request.csv_base64)}")
            
            base64_file = os.path.join(debug_dir, f"rest_truncated_base64_{timestamp}.txt")
            with open(base64_file, 'w') as f:
                f.write(request.csv_base64)
            logger.error(f"REST: Truncated input dumped to: {base64_file}")
        except Exception as dump_err:
            logger.error(f"REST: Failed to dump truncated input: {dump_err}")
    
    # Check for overload before processing
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"Rejecting /process_unified request due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    result = await parse_and_schedule(request.csv_base64, force_calculate=request.force_calculate)
    if result.error:
        logger.info(f"Response: /process_unified - error={result.error}")
    else:
        logger.info(f"Response: /process_unified - handle={result.handle}, has_warnings={result.warnings.has_warnings}")
    return result

@app.post("/draw_a_plot")
async def draw_a_plot(request: PlotRequest, api_key: str = Depends(require_api_key), http_request: Request = None):
    """
    Generate a Plotly JSON plot for a cached dataset and index.
    Returns Plotly figure as JSON dict (compatible with Gradio gr.Plot).
    Requires valid API key in X-API-Key header.
    
    Multiple concurrent requests for the same (handle, index) will wait for the same result.
    Implements disconnect detection and cancellation.
    """
    logger.info(f"Request: /draw_a_plot - handle={request.handle}, index={request.index}, force={request.force_calculate}")
    
    # Check for overload before processing
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"Rejecting /draw_a_plot request due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    # Register request and get request_id
    disconnect_tracker = DisconnectTracker()
    request_id = await disconnect_tracker.register_request(request.handle, request.index)
    
    # Update latest request_id for this (handle, index)
    bg_processor = BackgroundProcessor()
    await bg_processor.update_latest_request_id(request.handle, request.index, request_id)
    
    # Get per-request disconnect future (not the shared one)
    # This allows individual request disconnect detection
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
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, plot_keys={list(plot_dict.keys())}")
        return plot_dict
    except asyncio.CancelledError:
        logger.info(f"Request cancelled: {request.handle[:8]}:{request.index}:{request_id}")
        # Use the unified cancellation hook
        await TaskRegistry().cancel_request(request.handle, request.index, request_id, "Client disconnected")
        raise HTTPException(status_code=499, detail="Client closed request")
    except asyncio.TimeoutError:
        logger.info(f"Request timeout: {request.handle[:8]}:{request.index}:{request_id}")
        # Use the unified cancellation hook
        await TaskRegistry().cancel_request(request.handle, request.index, request_id, "Request timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except ValueError as e:
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, error={str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Plot failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if disconnect_link_task:
            disconnect_link_task.cancel()
        # Unregister request (decrease counter)
        await disconnect_tracker.unregister_request(request.handle, request.index, request_id)

@app.post("/quick_plot", response_model=QuickPlotResponse)
async def quick_plot(request: ProcessRequest, api_key: str = Depends(require_api_key), http_request: Request = None):
    """
    Upload CSV, process, and immediately get the Plotly JSON plot for the last sample.
    Returns a Plotly figure as JSON dict (compatible with Gradio gr.Plot).
    Requires valid API key in X-API-Key header.
    Implements disconnect detection and cancellation.
    """
    logger.info(f"Request: /quick_plot - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    
    # Check for overload before processing
    is_overloaded, load_status, _, _ = check_queue_overload()
    if is_overloaded:
        logger.warning(f"Rejecting /quick_plot request due to overload (status={load_status})")
        raise HTTPException(
            status_code=503,
            detail=f"Service overloaded. Queue is {load_status}. Please retry later.",
            headers={"Retry-After": "30"}
        )
    
    # For quick_plot, we don't know the handle/index yet, so we'll assign a temporary request_id
    # The actual registration will happen in generate_plot_from_handle
    import uuid
    request_id = hash(str(uuid.uuid4())) & 0x7FFFFFFF  # Positive int
    
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
                logger.info(f"Response: /quick_plot - error={result.error}")
            else:
                warnings = result.warnings.has_warnings if hasattr(result.warnings, 'has_warnings') else result.warnings.get('has_warnings', False)
                logger.info(f"Response: /quick_plot - success, plot_keys={list(result.plot_data.keys())}, has_warnings={warnings}")
            return result
        finally:
            if disconnect_link_task:
                disconnect_link_task.cancel()

    except asyncio.CancelledError:
        logger.info(f"Quick plot request cancelled: req_id={request_id}")
        return QuickPlotResponse(plot_data={}, warnings={}, error="Request cancelled")

@app.post("/cache_management", response_model=CacheManagementResponse)
async def cache_management(
    action: str = Query(..., description="Action to perform: 'flush', 'info', 'delete', 'save', 'load'"),
    handle: Optional[str] = Query(None, description="Handle for delete/load/save operations"),
    api_key: str = Depends(require_api_key)
):
    """
    Manage the cache (Flush, Info, Delete, Save, Load).
    Actions:
    - flush: Clear all cache (memory and disk)
    - info: Get cache statistics
    - delete: Delete a specific handle (requires handle parameter)
    - save: Save cache to disk (optional handle parameter for specific entry) - Auto-persisted in new engine
    - load: Load a handle from disk to memory (requires handle parameter) - Auto-loaded in new engine
    
    Requires valid API key.
    """
    logger.info(f"Request: /cache_management - action={action}, handle={handle}")
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
        
        if exists:
            return CacheManagementResponse(
                success=True,
                message=f"Handle {handle} deleted successfully",
                cache_size=size,
                persisted_count=size,
                items_affected=1
            )
        else:
            return CacheManagementResponse(
                success=False,
                message=f"Handle {handle} not found in cache",
                cache_size=size,
                persisted_count=size,
                items_affected=0
            )
    
    elif action == "save":
        # New engine has auto-persistence
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
        
        # 'get' automatically loads from disk if present
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
    logger.info("Request: /health")
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
    logger.info(f"Response: /health - status={response.status}, load_status={response.load_status}, health={response.model_dump_json()}")
    return response

def start_server():
    """Start the uvicorn server with proper signal handling."""
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run("glurpc.app:app", host="0.0.0.0", port=8000, reload=False)
