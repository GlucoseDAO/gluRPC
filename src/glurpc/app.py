import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
import base64
import uvicorn

from .core import (
    convert_to_unified_action,
    process_and_cache,
    generate_plot_from_handle,
    quick_plot_action,
    get_model,
    CACHE
)
from .schemas import (
    UnifiedResponse,
    PlotRequest,
    QuickPlotResponse,
    ConvertResponse
)
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GluRPC", description="Glucose Prediction Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up GluRPC server...")
    try:
        get_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # We don't crash, but subsequent requests might fail
        pass

@app.post("/convert_to_unified", response_model=ConvertResponse)
async def convert_to_unified(file: UploadFile = File(...)):
    try:
        content = await file.read()
        content_base64 = base64.b64encode(content).decode()
        result = convert_to_unified_action(content_base64)
        return result
    except Exception as e:
        logger.error(f"Convert failed: {e}")
        return ConvertResponse(error=str(e))

class ProcessRequest(BaseModel):
    csv_base64: str

@app.post("/process_unified", response_model=UnifiedResponse)
async def process_unified(request: ProcessRequest):
    """
    Upload a CSV (base64 encoded) to process and cache for plotting.
    """
    result = process_and_cache(request.csv_base64)
    if result.error:
        # Can return 400 or just the error object
        pass 
    return result

@app.post("/draw_a_plot")
async def draw_a_plot(request: PlotRequest):
    """
    Generate a PNG plot for a cached dataset and index.
    Returns raw PNG bytes.
    """
    try:
        png_bytes = generate_plot_from_handle(request.handle, request.index)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Plot failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/quick_plot", response_model=QuickPlotResponse)
async def quick_plot(request: ProcessRequest):
    """
    Upload CSV, process, and immediately get the plot for the last sample.
    """
    return quick_plot_action(request.csv_base64)

@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(CACHE)}

def start_server():
    uvicorn.run("glurpc.app:app", host="0.0.0.0", port=8000, reload=False)
