import os
import base64
import asyncio
import pytest
import random
import logging
import time
from pathlib import Path

from httpx import AsyncClient, ASGITransport
from glurpc.app import app
import pandas as pd

# Setup logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_test")

DATA_DIR = Path(__file__).parent.parent / "data"

def get_all_csvs():
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found at {DATA_DIR}")
    
    files = []
    for f in DATA_DIR.rglob("*.csv"):
        if "parsed" in str(f):
            continue
        files.append(f)
    return sorted(files)

async def run_load_test():
    csv_files = get_all_csvs()
    if not csv_files:
        pytest.skip("No CSV files found")
        
    logger.info(f"Found {len(csv_files)} CSV files for testing")
    
    valid_handles = []
    errors_count = 0
    start_time = time.time()
    
    # Stats Containers
    request_timestamps = []
    file_stats = []
    
    # Manually manage lifespan since ASGITransport doesn't automatically do it
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=120.0) as ac:
            
            # Phase 1: Process all CSVs concurrently
            logger.info("Phase 1: Processing all CSVs concurrently...")
            
            async def process_file(path):
                t0 = time.time()
                try:
                    content = path.read_bytes()
                    b64 = base64.b64encode(content).decode()
                    resp = await ac.post("/process_unified", json={"csv_base64": b64})
                    
                    duration = time.time() - t0
                    
                    if resp.status_code == 500:
                        return "500", None, duration
                    
                    if resp.status_code != 200:
                        return f"HTTP{resp.status_code}", None, duration
                        
                    data = resp.json()
                    if data.get("error"):
                        return "BusinessError", None, duration
                        
                    return "OK", data["handle"], duration
                except Exception as e:
                    logger.error(f"Request failed for {path.name}: {e}")
                    return "Exception", None, time.time() - t0

            tasks = [process_file(f) for f in csv_files]
            results = await asyncio.gather(*tasks)
            
            for i, (status, handle, duration) in enumerate(results):
                file_name = csv_files[i].name
                file_stats.append({"file": file_name, "status": status, "duration": duration})
                
                if status == "OK":
                    valid_handles.append(handle)
                elif status == "BusinessError":
                    errors_count += 1
                elif status == "500":
                    pytest.fail("Received 500 Internal Server Error")
                else:
                    if status.startswith("HTTP4"):
                        errors_count += 1
            
            logger.info(f"Processing complete. Valid handles: {len(valid_handles)}, Errors: {errors_count}")
            
            # Report Phase 1 Stats
            df_phase1 = pd.DataFrame(file_stats)
            logger.info("\n=== Phase 1 Statistics ===")
            logger.info(f"Total Files: {len(df_phase1)}")
            logger.info(f"Avg Duration: {df_phase1['duration'].mean():.2f}s")
            logger.info(f"Max Duration: {df_phase1['duration'].max():.2f}s")
            logger.info("Status Counts:\n" + str(df_phase1['status'].value_counts()))
            
            # Assertion: Exactly 2 processing errors
            assert errors_count == 2, f"Expected 2 processing errors, got {errors_count}"
            assert len(valid_handles) > 0
            
            # Phase 2: Stress Test Inference/Plotting
            selected_handles = valid_handles[:10]
            logger.info(f"Phase 2: Stress testing with {len(selected_handles)} handles")
            
            plot_results = []
            phase2_start = time.time()
            
            async def request_plot(handle, idx):
                try:
                    resp = await ac.post("/draw_a_plot", json={"handle": handle, "index": idx})
                    request_timestamps.append(time.time())
                    if resp.status_code == 200:
                        return resp.content
                    return None
                except Exception:
                    return None

            tasks = []
            for handle in selected_handles:
                for i in range(100):
                    idx = i % 20 
                    tasks.append(request_plot(handle, idx))
            
            logger.info(f"Firing {len(tasks)} plot requests...")
            responses = await asyncio.gather(*tasks)
            
            valid_pngs = [r for r in responses if r is not None and len(r) > 0]
            phase2_duration = time.time() - phase2_start
            
            logger.info(f"Received {len(valid_pngs)} valid PNGs out of {len(tasks)} requests")
            
            # Phase 2 Stats
            logger.info("\n=== Phase 2 Statistics ===")
            logger.info(f"Total Requests: {len(tasks)}")
            logger.info(f"Total Time: {phase2_duration:.2f}s")
            logger.info(f"Throughput: {len(tasks)/phase2_duration:.2f} req/s")
            
            # Histogram of requests over time (1 second bins)
            if request_timestamps:
                df_reqs = pd.DataFrame({'timestamp': request_timestamps})
                df_reqs['time_rel'] = df_reqs['timestamp'] - phase2_start
                df_reqs['bin'] = df_reqs['time_rel'].astype(int)
                hist = df_reqs['bin'].value_counts().sort_index()
                logger.info("\n=== Request Profile (Requests per Second) ===")
                logger.info(str(hist))
            
            assert len(valid_pngs) > 0
            
            if valid_pngs:
                random_png = random.choice(valid_pngs)
                out_path = "files/stress_test_random.png"
                with open(out_path, "wb") as f:
                    f.write(random_png)
                logger.info(f"Saved random plot to {out_path}")

def test_integration_load():
    """Sync wrapper for async test logic"""
    asyncio.run(run_load_test())
