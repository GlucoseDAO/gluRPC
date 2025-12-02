import base64
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, Query
from fastapi.responses import Response, JSONResponse

# Dependencies from glurpc
from glurpc.config import ENABLE_API_KEYS
from glurpc.core import (
    convert_to_unified_action,
    process_and_cache,
    generate_plot_from_handle,
    quick_plot_action,
    verify_api_key
)
from glurpc.engine import ModelManager, BackgroundProcessor
from glurpc.state import DataCache
from glurpc.schemas import (
    UnifiedResponse,
    PlotRequest,
    QuickPlotResponse,
    ConvertResponse,
    HealthResponse,
    ProcessRequest,
    CacheManagementResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up GluRPC server...")
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
    await bg_processor.stop()

app = FastAPI(
    title="GluRPC",
    description="Glucose Prediction Service",
    lifespan=lifespan
)

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
        ModelManager().increment_errors() 
        return ConvertResponse(error=str(e))

@app.post("/process_unified", response_model=UnifiedResponse)
async def process_unified(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """
    Upload a CSV (base64 encoded) to process and cache for plotting.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /process_unified - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    result = await process_and_cache(request.csv_base64, force_calculate=request.force_calculate)
    if result.error:
        logger.info(f"Response: /process_unified - error={result.error}")
    else:
        logger.info(f"Response: /process_unified - handle={result.handle}, has_warnings={result.warnings.get('has_warnings', False)}")
    return result

@app.post("/draw_a_plot")
async def draw_a_plot(request: PlotRequest, api_key: str = Depends(require_api_key)):
    """
    Generate a PNG plot for a cached dataset and index.
    Returns raw PNG bytes.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /draw_a_plot - handle={request.handle}, index={request.index}")
    model_manager = ModelManager()
    
    try:
        png_bytes = await generate_plot_from_handle(request.handle, request.index)
        model_manager.increment_requests()
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, png_size={len(png_bytes)} bytes")
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        model_manager.increment_errors()
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, error={str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        model_manager.increment_errors()
        logger.error(f"Plot failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/quick_plot", response_model=QuickPlotResponse)
async def quick_plot(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """
    Upload CSV, process, and immediately get the plot for the last sample.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /quick_plot - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
    result = await quick_plot_action(request.csv_base64, force_calculate=request.force_calculate)
    if result.error:
        logger.info(f"Response: /quick_plot - error={result.error}")
    else:
        warnings = result.warnings.get('has_warnings', False) if result.warnings else False
        logger.info(f"Response: /quick_plot - success, plot_base64_length={len(result.plot_base64)}, has_warnings={warnings}")
    return result

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
    - save: Save cache to disk (optional handle parameter for specific entry)
    - load: Load a handle from disk to memory (requires handle parameter)
    
    Requires valid API key.
    """
    logger.info(f"Request: /cache_management - action={action}, handle={handle}")
    data_cache = DataCache()
    
    if action == "flush":
        await data_cache.clear_cache()
        return CacheManagementResponse(
            success=True,
            message="Cache flushed successfully",
            cache_size=0,
            persisted_count=0,
            items_affected=None
        )
    
    elif action == "info":
        size = await data_cache.get_size()
        return CacheManagementResponse(
            success=True,
            message="Cache info retrieved",
            cache_size=size,
            persisted_count=size,
            items_affected=None
        )
    
    elif action == "delete":
        if not handle:
            raise HTTPException(status_code=400, detail="Handle parameter required for delete action")
        
        deleted = await data_cache.delete_handle(handle)
        size = await data_cache.get_size()
        
        if deleted:
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
        saved_count = await data_cache.save_to_disk(handle)
        size = await data_cache.get_size()
        
        if handle:
            message = f"Handle {handle} saved to disk" if saved_count > 0 else f"Handle {handle} not found in memory"
        else:
            message = f"Saved {saved_count} entries to disk"
        
        return CacheManagementResponse(
            success=saved_count > 0,
            message=message,
            cache_size=size,
            persisted_count=size,
            items_affected=saved_count
        )
    
    elif action == "load":
        if not handle:
            raise HTTPException(status_code=400, detail="Handle parameter required for load action")
        
        loaded = await data_cache.load_from_disk(handle)
        size = await data_cache.get_size()
        
        if loaded:
            return CacheManagementResponse(
                success=True,
                message=f"Handle {handle} loaded from disk to memory",
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
    data_cache = DataCache()
    
    stats = model_manager.get_stats()
    cache_size = await data_cache.get_size()
    
    response = HealthResponse(
        status="ok" if model_manager.initialized else "degraded",
        cache_size=cache_size,
        models_initialized=model_manager.initialized,
        queue_length=stats["queue_length"],
        avg_fulfillment_time_ms=stats["avg_fulfillment_time_ms"],
        vmem_usage_mb=stats["vmem_usage_mb"],
        device=stats["device"],
        total_requests_processed=stats["total_requests_processed"],
        total_errors=stats["total_errors"]
    )
    logger.info(f"Response: /health - status={response.status}, health={response.model_dump_json()}")
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
