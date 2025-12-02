import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.responses import Response, JSONResponse

# Dependencies from glurpc
from glurpc.core import (
    convert_to_unified_action,
    process_and_cache,
    generate_plot_from_handle,
    quick_plot_action,
    verify_api_key,
    ENABLE_API_KEYS
)
from glurpc.engine import MODEL_MANAGER, BACKGROUND_PROCESSOR
from glurpc.schemas import (
    UnifiedResponse,
    PlotRequest,
    QuickPlotResponse,
    ConvertResponse,
    HealthResponse,
    ProcessRequest
)
from glurpc.state import DATA_CACHE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up GluRPC server...")
    try:
        await MODEL_MANAGER.initialize()
        await BACKGROUND_PROCESSOR.start()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model or start processor on startup: {e}")
        raise e # Model failure is critical, terminate app
    yield
    # Shutdown
    logger.info("Shutting down GluRPC server...")
    BACKGROUND_PROCESSOR.stop()

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
        MODEL_MANAGER.increment_errors() 
        return ConvertResponse(error=str(e))

@app.post("/process_unified", response_model=UnifiedResponse)
async def process_unified(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """
    Upload a CSV (base64 encoded) to process and cache for plotting.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /process_unified - csv_base64_length={len(request.csv_base64)}")
    result = await process_and_cache(request.csv_base64)
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
    try:
        png_bytes = await generate_plot_from_handle(request.handle, request.index)
        MODEL_MANAGER.increment_requests()
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, png_size={len(png_bytes)} bytes")
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        MODEL_MANAGER.increment_errors()
        logger.info(f"Response: /draw_a_plot - handle={request.handle}, index={request.index}, error={str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        MODEL_MANAGER.increment_errors()
        logger.error(f"Plot failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/quick_plot", response_model=QuickPlotResponse)
async def quick_plot(request: ProcessRequest, api_key: str = Depends(require_api_key)):
    """
    Upload CSV, process, and immediately get the plot for the last sample.
    Requires valid API key in X-API-Key header.
    """
    logger.info(f"Request: /quick_plot - csv_base64_length={len(request.csv_base64)}")
    result = await quick_plot_action(request.csv_base64)
    if result.error:
        logger.info(f"Response: /quick_plot - error={result.error}")
    else:
        warnings = result.warnings.get('has_warnings', False) if result.warnings else False
        logger.info(f"Response: /quick_plot - success, plot_base64_length={len(result.plot_base64)}, has_warnings={warnings}")
    return result

@app.get("/health", response_model=HealthResponse)
async def health():
    logger.info("Request: /health")
    stats = MODEL_MANAGER.get_stats()
    response = HealthResponse(
        status="ok" if MODEL_MANAGER.initialized else "degraded",
        cache_size=len(DATA_CACHE),
        models_initialized=MODEL_MANAGER.initialized,
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
    uvicorn.run("glurpc.app:app", host="0.0.0.0", port=8000, reload=False)
