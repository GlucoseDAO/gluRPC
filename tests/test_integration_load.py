import base64
import asyncio
import logging
import multiprocessing
import random
import socket
import time
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
import pytest
import pytest_asyncio
import uvicorn
from httpx import AsyncClient, Response

# Run all async tests in this module on the same event loop so the
# module-scoped AsyncClient remains valid across parameterizations.
pytestmark = pytest.mark.asyncio(scope="module")

# Setup logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_test")

# Ensure output directory exists
Path("test_outputs").mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data"

MAX_503_RETRIES = 10
RETRY_AFTER_FALLBACK_SECONDS = 30.0

# Test configuration
HAMMERING_ITERATIONS = 10  # Number of hammering cycles
PARALLEL_REQUESTS_PER_CYCLE = 100  # Total requests per cycle
TRACKED_REQUESTS_PER_CYCLE = 20  # Requests we wait for (other 80 are fire-and-forget)
FIRE_AND_FORGET_REQUESTS_PER_CYCLE = PARALLEL_REQUESTS_PER_CYCLE - TRACKED_REQUESTS_PER_CYCLE
CANCELLATION_PROBABILITY = 0.3  # 30% of fire-and-forget requests will be cancelled


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(host: str, port: int) -> None:
    """Run uvicorn server in a separate process for the module lifespan."""
    config = uvicorn.Config("glurpc.app:app", host=host, port=port, reload=False, log_level="info")
    server = uvicorn.Server(config)
    server.run()


@pytest.fixture(scope="module")
def live_server():
    """
    Start the FastAPI app once for the whole module in its own process.
    Keeps a single event loop/lifecycle for all load tests (no cache resets).
    """
    host = "127.0.0.1"
    port = _find_free_port()
    proc = multiprocessing.Process(target=_run_server, args=(host, port), daemon=True)
    proc.start()

    # Wait until /health is ready
    async def _wait_ready():
        for _ in range(40):
            try:
                async with AsyncClient(base_url=f"http://{host}:{port}", timeout=2.0) as ac:
                    resp = await ac.get("/health")
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Server did not become ready in time")

    # Create a dedicated loop for the readiness probe to avoid relying on
    # pytest-asyncio's function-scoped loop.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_wait_ready())
    finally:
        loop.close()
    yield f"http://{host}:{port}"
    proc.terminate()
    proc.join(timeout=5)


@pytest_asyncio.fixture(scope="module")
async def client(live_server: str):
    """
    Module-scoped HTTP client pointing to the live server process.
    """
    async with AsyncClient(base_url=live_server, timeout=120.0) as ac:
        yield ac

def get_all_csvs() -> List[Path]:
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found at {DATA_DIR}")
    
    files = []
    for f in DATA_DIR.rglob("*.csv"):
        if "parsed" in str(f):
            continue
        files.append(f)
    return sorted(files)


async def post_with_retry_on_503(
    client: AsyncClient,
    url: str,
    payload: dict,
    *,
    max_retries: int = MAX_503_RETRIES,
    default_delay: float = RETRY_AFTER_FALLBACK_SECONDS,
) -> Response:
    """
    Retry POST requests when the service is temporarily overloaded (HTTP 503).
    Respects the Retry-After header when present, otherwise falls back to the
    configured default delay.
    """
    last_response: Optional[Response] = None
    for attempt in range(max_retries + 1):
        resp = await client.post(url, json=payload)
        last_response = resp
        if resp.status_code != 503:
            return resp

        if attempt == max_retries:
            return resp

        retry_after_raw = resp.headers.get("Retry-After")
        try:
            delay = float(retry_after_raw) if retry_after_raw is not None else default_delay
        except (TypeError, ValueError):
            delay = default_delay

        logger.warning(
            f"{url} returned 503 (attempt {attempt + 1}/{max_retries}); "
            f"retrying in {delay:.0f}s"
        )
        await asyncio.sleep(delay)

    # Fallback in case the loop exits unexpectedly
    if last_response is None:
        raise RuntimeError("post_with_retry_on_503 finished without a response")
    return last_response


# ============================================================================
# Phase 1: Process CSV Files
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("csv_path", get_all_csvs(), ids=lambda p: p.name)
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
async def test_process_single_file(client: AsyncClient, csv_path: Path, force_calculate: bool):
    """
    Test processing a single CSV file.
    
    This tests the /process_unified endpoint with each input file.
    """
    file_name = csv_path.name
    logger.info(f"Processing {file_name} (force_calculate={force_calculate})...")
    
    t0 = time.time()
    try:
        content = csv_path.read_bytes()
        b64 = base64.b64encode(content).decode()
        resp = await client.post("/process_unified", json={"csv_base64": b64, "force_calculate": force_calculate})
        
        duration = time.time() - t0
        logger.info(f"Processed {file_name} in {duration:.2f}s, status: {resp.status_code}")
        
        # Assert no 500 errors
        assert resp.status_code != 500, f"Received 500 Internal Server Error for {file_name}"
        
        # If we got a handle, store it for later phases
        if resp.status_code == 200:
            data = resp.json()
            if not data.get("error"):
                handle = data["handle"]
                pytest.handle_storage = getattr(pytest, 'handle_storage', {})
                pytest.handle_storage[(file_name, force_calculate)] = handle
                
    except Exception as e:
        duration = time.time() - t0
        logger.error(f"Request failed for {file_name}: {e}")
        pytest.fail(f"Exception during processing: {e}")


# ============================================================================
# Phase 2: Sanity Check - Wait for First Plot
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
async def test_sanity_check_first_plot(client: AsyncClient, force_calculate: bool):
    """
    Wait for the first inference to complete by polling for a plot.
    
    This ensures the inference system is warmed up before hammering.
    """
    logger.info(f"Sanity check: waiting for first plot (force_calculate={force_calculate})...")
    
    # Get a valid handle from previous test
    handle_storage = getattr(pytest, 'handle_storage', {})
    valid_handles = [h for (fn, fc), h in handle_storage.items() if fc == force_calculate]
    
    if not valid_handles:
        pytest.skip("No valid handles available from processing phase")
    
    first_handle = valid_handles[0]
    
    t0 = time.time()
    sanity_check_success = False
    
    for attempt in range(3):  # Try up to 3 times with delays
        try:
            resp = await client.post("/draw_a_plot", json={"handle": first_handle, "index": 0})
            if resp.status_code == 200:
                sanity_check_success = True
                logger.info(f"First plot retrieved successfully after {time.time() - t0:.2f}s")
                break
            else:
                logger.warning(f"Attempt {attempt + 1}: Plot not ready yet (status {resp.status_code})")
                await asyncio.sleep(30)  # Wait 30s between attempts
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Exception {e}")
            await asyncio.sleep(30)
    
    if not sanity_check_success:
        pytest.skip("Sanity check failed - inference may still be running")
    
    # Store success flag for later phases
    pytest.sanity_check_passed = getattr(pytest, 'sanity_check_passed', {})
    pytest.sanity_check_passed[force_calculate] = True


# ============================================================================
# Phase 3: Hammering - Multiple Iterations with Tracking
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
@pytest.mark.parametrize("iteration", range(HAMMERING_ITERATIONS), ids=lambda i: f"iter{i}")
async def test_hammering_iteration(client: AsyncClient, force_calculate: bool, iteration: int):
    """
    Hammer the API with 100 parallel requests per iteration.
    
    - Track 20 requests (wait for them to complete)
    - Fire-and-forget 80 requests (don't wait for completion)
    
    This allows us to see progress per iteration rather than waiting 20 minutes.
    """
    logger.info(f"Hammering iteration {iteration} (force_calculate={force_calculate})...")
    
    # Check if sanity check passed
    sanity_check_passed = getattr(pytest, 'sanity_check_passed', {}).get(force_calculate, False)
    if not sanity_check_passed:
        pytest.skip("Sanity check did not pass, skipping hammering")
    
    # Get valid handles
    handle_storage = getattr(pytest, 'handle_storage', {})
    valid_handles = [h for (fn, fc), h in handle_storage.items() if fc == force_calculate]
    
    if not valid_handles:
        pytest.skip("No valid handles available")
    
    # Select a subset of handles for this iteration
    selected_handles = random.sample(valid_handles, min(10, len(valid_handles)))
    
    t0 = time.time()
    
    # Generate request tasks
    tracked_tasks = []
    fire_and_forget_tasks = []
    
    async def request_plot(handle: str, idx: int, is_tracked: bool):
        """Make a plot request."""
        payload = {"handle": handle, "index": idx}
        try:
            if is_tracked:
                resp = await post_with_retry_on_503(client, "/draw_a_plot", payload)
            else:
                resp = await client.post("/draw_a_plot", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data and "layout" in data:
                    return {"success": True, "handle": handle[:8], "index": idx, "is_tracked": is_tracked}
            return {"success": False, "handle": handle[:8], "index": idx, "is_tracked": is_tracked, "status": resp.status_code}
        except Exception as e:
            return {"success": False, "handle": handle[:8], "index": idx, "is_tracked": is_tracked, "error": str(e)}
    
    # Create TRACKED_REQUESTS_PER_CYCLE tracked requests
    for i in range(TRACKED_REQUESTS_PER_CYCLE):
        handle = random.choice(selected_handles)
        idx = random.randint(-20, 0)
        tracked_tasks.append(request_plot(handle, idx, True))
    
    # Create FIRE_AND_FORGET_REQUESTS_PER_CYCLE fire-and-forget requests
    for i in range(FIRE_AND_FORGET_REQUESTS_PER_CYCLE):
        handle = random.choice(selected_handles)
        idx = random.randint(-20, 0)
        fire_and_forget_tasks.append(request_plot(handle, idx, False))
    
    # Fire all requests in parallel
    logger.info(f"Firing {PARALLEL_REQUESTS_PER_CYCLE} requests ({TRACKED_REQUESTS_PER_CYCLE} tracked, {FIRE_AND_FORGET_REQUESTS_PER_CYCLE} fire-and-forget)...")
    
    # Start all tasks
    all_tasks = tracked_tasks + fire_and_forget_tasks
    all_coros = [asyncio.create_task(t) for t in all_tasks]
    
    # Randomly cancel some fire-and-forget requests to test cancellation
    fire_and_forget_coros = all_coros[TRACKED_REQUESTS_PER_CYCLE:]
    cancelled_count = 0
    cancel_delay = random.uniform(0.1, 0.5)  # Cancel after a short random delay
    
    async def cancel_random_requests():
        nonlocal cancelled_count
        await asyncio.sleep(cancel_delay)  # Let requests start first
        for coro in fire_and_forget_coros:
            if random.random() < CANCELLATION_PROBABILITY:
                coro.cancel()
                cancelled_count += 1
    
    # Start cancellation task
    cancel_task = asyncio.create_task(cancel_random_requests())
    
    # Wait only for tracked tasks
    tracked_results = await asyncio.gather(*all_coros[:TRACKED_REQUESTS_PER_CYCLE])
    
    # Wait for cancellation task to complete
    await cancel_task
    
    duration = time.time() - t0
    
    # Count successes in tracked requests
    tracked_success = sum(1 for r in tracked_results if r.get("success"))
    
    logger.info(f"Iteration {iteration} completed in {duration:.2f}s")
    logger.info(f"Tracked requests: {tracked_success}/{TRACKED_REQUESTS_PER_CYCLE} succeeded")
    logger.info(f"Fire-and-forget cancelled: {cancelled_count}/{FIRE_AND_FORGET_REQUESTS_PER_CYCLE} ({cancelled_count/FIRE_AND_FORGET_REQUESTS_PER_CYCLE*100:.1f}%)")
    logger.info(f"Throughput: {TRACKED_REQUESTS_PER_CYCLE/duration:.2f} req/s (tracked only)")
    
    # Assert at least some tracked requests succeeded
    assert tracked_success > 0, f"No tracked requests succeeded in iteration {iteration}"
    
    # Save iteration stats
    pytest.hammering_stats = getattr(pytest, 'hammering_stats', [])
    pytest.hammering_stats.append({
        "force_calculate": force_calculate,
        "iteration": iteration,
        "tracked_success": tracked_success,
        "tracked_total": TRACKED_REQUESTS_PER_CYCLE,
        "cancelled_count": cancelled_count,
        "fire_and_forget_total": FIRE_AND_FORGET_REQUESTS_PER_CYCLE,
        "duration": duration,
        "throughput": TRACKED_REQUESTS_PER_CYCLE / duration
    })


# ============================================================================
# Phase 4: Health Endpoint Hammering
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
async def test_health_endpoint_hammering(client: AsyncClient, force_calculate: bool):
    """
    Hammer the /health endpoint to ensure it remains responsive under load.
    
    This simulates aggressive UI polling.
    """
    logger.info(f"Health endpoint hammering (force_calculate={force_calculate})...")
    
    health_request_count = 200
    
    t0 = time.time()
    
    async def poll_health():
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False
    
    logger.info(f"Firing {health_request_count} concurrent health requests...")
    health_tasks = [poll_health() for _ in range(health_request_count)]
    health_results = await asyncio.gather(*health_tasks)
    
    health_success = sum(health_results)
    health_failures = health_request_count - health_success
    
    duration = time.time() - t0
    logger.info(f"Health requests: {health_request_count} total, {health_success} success, {health_failures} failures")
    logger.info(f"Health throughput: {health_request_count/duration:.2f} req/s")
    logger.info(f"Completed in {duration:.2f}s")
    
    assert health_success == health_request_count, f"Some health checks failed: {health_failures}/{health_request_count}"


# ============================================================================
# Phase 5: Random Plot Sampling (Save to Disk)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
async def test_random_plot_sampling(client: AsyncClient, force_calculate: bool):
    """
    Sample random plots and save them to disk for manual inspection.
    """
    logger.info(f"Random plot sampling (force_calculate={force_calculate})...")
    
    # Get valid handles
    handle_storage = getattr(pytest, 'handle_storage', {})
    valid_handles = [h for (fn, fc), h in handle_storage.items() if fc == force_calculate]
    
    if not valid_handles:
        pytest.skip("No valid handles available")
    
    sample_count = min(10, len(valid_handles))
    random_handles = random.sample(valid_handles, sample_count)
    
    output_dir = Path("test_outputs")
    suffix = "_forced" if force_calculate else "_cached"
    
    random_plot_samples = []
    
    for h_idx, handle in enumerate(random_handles):
        random_idx = random.randint(-20, 0)
        try:
            resp = await post_with_retry_on_503(
                client,
                "/draw_a_plot",
                {"handle": handle, "index": random_idx},
            )
            if resp.status_code == 200:
                plot_dict = resp.json()
                if isinstance(plot_dict, dict) and "data" in plot_dict:
                    random_plot_samples.append((h_idx, handle[:8], random_idx, plot_dict))
                    logger.info(f"Sampled plot from handle {handle[:8]}... at index {random_idx}")
        except Exception as e:
            logger.warning(f"Failed to sample plot from {handle[:8]}...: {e}")
    
    # Save random samples
    for h_idx, handle_short, idx, plot_dict in random_plot_samples:
        try:
            fig = go.Figure(plot_dict)
            fig.write_html(output_dir / f"load_test_random_h{h_idx}_idx{idx}{suffix}.html")
            fig.write_image(output_dir / f"load_test_random_h{h_idx}_idx{idx}{suffix}.svg")
        except Exception as e:
            logger.warning(f"Failed to save random plot {h_idx}: {e}")
    
    logger.info(f"Saved {len(random_plot_samples)} random plot samples")


# ============================================================================
# Phase 6: Mixed Valid/Invalid Index Hammering
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
async def test_mixed_index_hammering(client: AsyncClient, force_calculate: bool):
    """
    Hammer with a mix of valid and invalid indices to test error handling.
    
    Valid indices: 0 to -20
    Invalid indices: -200 to -100 (likely out of range)
    """
    logger.info(f"Mixed index hammering (force_calculate={force_calculate})...")
    
    # Check if sanity check passed
    sanity_check_passed = getattr(pytest, 'sanity_check_passed', {}).get(force_calculate, False)
    if not sanity_check_passed:
        pytest.skip("Sanity check did not pass, skipping hammering")
    
    # Get valid handles
    handle_storage = getattr(pytest, 'handle_storage', {})
    valid_handles = [h for (fn, fc), h in handle_storage.items() if fc == force_calculate]
    
    if not valid_handles:
        pytest.skip("No valid handles available")
    
    hammer_handles = random.sample(valid_handles, min(3, len(valid_handles)))
    
    valid_requests = []
    invalid_requests = []
    
    # Generate mix of valid and invalid index requests
    for handle in hammer_handles:
        # Add valid requests (indices 0 to -20)
        for _ in range(10):
            idx = random.randint(-20, 0)
            valid_requests.append((handle, idx, True))
        
        # Add invalid requests (indices -200 to -100, likely out of range)
        for _ in range(10):
            idx = random.randint(-200, -100)
            invalid_requests.append((handle, idx, False))
    
    all_hammer_requests = valid_requests + invalid_requests
    random.shuffle(all_hammer_requests)
    
    async def hammer_plot(handle: str, idx: int, expected_valid: bool):
        try:
            resp = await client.post("/draw_a_plot", json={"handle": handle, "index": idx})
            success = resp.status_code == 200
            return {
                "handle": handle[:8],
                "index": idx,
                "expected_valid": expected_valid,
                "status_code": resp.status_code,
                "success": success
            }
        except Exception as e:
            return {
                "handle": handle[:8],
                "index": idx,
                "expected_valid": expected_valid,
                "status_code": 0,
                "success": False,
                "error": str(e)
            }
    
    t0 = time.time()
    hammer_tasks = [hammer_plot(h, i, ev) for h, i, ev in all_hammer_requests]
    hammer_results = await asyncio.gather(*hammer_tasks)
    duration = time.time() - t0
    
    df_hammer = pd.DataFrame(hammer_results)
    
    # Separate results by expected validity
    valid_results = df_hammer[df_hammer['expected_valid'] == True]
    invalid_results = df_hammer[df_hammer['expected_valid'] == False]
    
    valid_success = valid_results['success'].sum()
    valid_total = len(valid_results)
    invalid_failures = (~invalid_results['success']).sum()
    invalid_total = len(invalid_results)
    
    logger.info(f"Valid index requests: {valid_success}/{valid_total} succeeded")
    logger.info(f"Invalid index requests: {invalid_failures}/{invalid_total} correctly failed")
    logger.info(f"Total hammer requests: {len(all_hammer_requests)}")
    logger.info(f"Hammer throughput: {len(all_hammer_requests)/duration:.2f} req/s")
    logger.info(f"Completed in {duration:.2f}s")
    
    # Status code distribution
    logger.info("\nStatus Code Distribution:")
    logger.info(str(df_hammer['status_code'].value_counts().sort_index()))
    
    # Assertions
    assert valid_success >= valid_total * 0.3, f"Too many valid requests failed: {valid_success}/{valid_total}"
    assert invalid_failures >= invalid_total * 0.8, f"Invalid requests should fail: only {invalid_failures}/{invalid_total} failed"


# ============================================================================
# Phase 7: ULTRAKILL - Chaotic Mixed Endpoint Hammering
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("force_calculate", [True, False], ids=["forced", "cached"])
@pytest.mark.parametrize("iteration", range(HAMMERING_ITERATIONS), ids=lambda i: f"ultrakill{i}")
async def test_ultrakill_mixed_endpoints(client: AsyncClient, force_calculate: bool, iteration: int):
    """
    ULTRAKILL: Randomly hammer ALL endpoints with mixed probabilities.
    
    Endpoint selection probabilities (relative weights):
    - Health: 5x (very common)
    - Draw plot (valid): 1x (normal)
    - Draw plot (invalid): 1x (normal)
    - Process forced: 0.2x (5x less common)
    
    Wait strategy:
    - 5% chance: Wait for response (tracked)
    - 95% chance: Fire-and-forget (don't wait)
    
    This creates maximum chaos to stress test the entire system.
    """
    logger.info(f"ULTRAKILL iteration {iteration} (force_calculate={force_calculate})...")
    
    # Check if sanity check passed
    sanity_check_passed = getattr(pytest, 'sanity_check_passed', {}).get(force_calculate, False)
    if not sanity_check_passed:
        pytest.skip("Sanity check did not pass, skipping ultrakill")
    
    # Get valid handles and CSV files
    handle_storage = getattr(pytest, 'handle_storage', {})
    valid_handles = [h for (fn, fc), h in handle_storage.items() if fc == force_calculate]
    
    if not valid_handles:
        pytest.skip("No valid handles available")
    
    csv_files = get_all_csvs()
    if not csv_files:
        pytest.skip("No CSV files found")
    
    # Endpoint weights (relative probabilities)
    WEIGHT_HEALTH = 5.0
    WEIGHT_PLOT_VALID = 1.0
    WEIGHT_PLOT_INVALID = 1.0
    WEIGHT_PROCESS_FORCED = 0.2
    
    total_weight = WEIGHT_HEALTH + WEIGHT_PLOT_VALID + WEIGHT_PLOT_INVALID + WEIGHT_PROCESS_FORCED
    
    # Probability of waiting for a request (vs fire-and-forget)
    WAIT_PROBABILITY = 0.05  # 5%
    
    t0 = time.time()
    
    tracked_tasks = []
    fire_and_forget_tasks = []
    request_types = []
    
    async def random_health_request(request_id: int, should_wait: bool):
        """Health endpoint request."""
        try:
            resp = await client.get("/health")
            return {
                "id": request_id,
                "type": "health",
                "success": resp.status_code == 200,
                "status_code": resp.status_code,
                "waited": should_wait
            }
        except Exception as e:
            return {
                "id": request_id,
                "type": "health",
                "success": False,
                "error": str(e),
                "waited": should_wait
            }
    
    async def random_plot_request(request_id: int, should_wait: bool, valid: bool):
        """Plot request with valid or invalid index."""
        handle = random.choice(valid_handles)
        if valid:
            idx = random.randint(-20, 0)
        else:
            idx = random.randint(-200, -100)
        
        try:
            payload = {"handle": handle, "index": idx}
            if should_wait:
                resp = await post_with_retry_on_503(client, "/draw_a_plot", payload)
            else:
                resp = await client.post("/draw_a_plot", json=payload)
            return {
                "id": request_id,
                "type": f"plot_{'valid' if valid else 'invalid'}",
                "success": resp.status_code == 200,
                "status_code": resp.status_code,
                "handle": handle[:8],
                "index": idx,
                "waited": should_wait
            }
        except Exception as e:
            return {
                "id": request_id,
                "type": f"plot_{'valid' if valid else 'invalid'}",
                "success": False,
                "error": str(e),
                "handle": handle[:8],
                "index": idx,
                "waited": should_wait
            }
    
    async def random_process_forced_request(request_id: int, should_wait: bool):
        """Process CSV with forced calculation."""
        csv_file = random.choice(csv_files)
        try:
            content = csv_file.read_bytes()
            b64 = base64.b64encode(content).decode()
            payload = {"csv_base64": b64, "force_calculate": True}
            if should_wait:
                resp = await post_with_retry_on_503(client, "/process_unified", payload)
            else:
                resp = await client.post("/process_unified", json=payload)
            return {
                "id": request_id,
                "type": "process_forced",
                "success": resp.status_code == 200,
                "status_code": resp.status_code,
                "file": csv_file.name,
                "waited": should_wait
            }
        except Exception as e:
            return {
                "id": request_id,
                "type": "process_forced",
                "success": False,
                "error": str(e),
                "file": csv_file.name,
                "waited": should_wait
            }
    
    # Generate PARALLEL_REQUESTS_PER_CYCLE requests with random endpoints
    for request_id in range(PARALLEL_REQUESTS_PER_CYCLE):
        # Decide if we should wait for this request
        should_wait = random.random() < WAIT_PROBABILITY
        
        # Select endpoint based on weights
        rand = random.random() * total_weight
        
        if rand < WEIGHT_HEALTH:
            # Health endpoint (most common)
            task = random_health_request(request_id, should_wait)
            request_types.append("health")
        elif rand < WEIGHT_HEALTH + WEIGHT_PLOT_VALID:
            # Valid plot request
            task = random_plot_request(request_id, should_wait, valid=True)
            request_types.append("plot_valid")
        elif rand < WEIGHT_HEALTH + WEIGHT_PLOT_VALID + WEIGHT_PLOT_INVALID:
            # Invalid plot request
            task = random_plot_request(request_id, should_wait, valid=False)
            request_types.append("plot_invalid")
        else:
            # Forced process (least common)
            task = random_process_forced_request(request_id, should_wait)
            request_types.append("process_forced")
        
        if should_wait:
            tracked_tasks.append(task)
        else:
            fire_and_forget_tasks.append(task)
    
    # Fire all requests
    logger.info(f"ULTRAKILL firing {PARALLEL_REQUESTS_PER_CYCLE} random requests ({len(tracked_tasks)} tracked, {len(fire_and_forget_tasks)} fire-and-forget)...")
    logger.info(f"Request type distribution: health={request_types.count('health')}, "
               f"plot_valid={request_types.count('plot_valid')}, "
               f"plot_invalid={request_types.count('plot_invalid')}, "
               f"process_forced={request_types.count('process_forced')}")
    
    # Start all tasks
    all_tasks = tracked_tasks + fire_and_forget_tasks
    all_coros = [asyncio.create_task(t) for t in all_tasks]
    
    # Randomly cancel some fire-and-forget requests
    fire_and_forget_coros = all_coros[len(tracked_tasks):]
    cancelled_count = 0
    cancel_delay = random.uniform(0.05, 0.3)  # Cancel quickly for chaos
    
    async def cancel_random_ultrakill_requests():
        nonlocal cancelled_count
        await asyncio.sleep(cancel_delay)
        for coro in fire_and_forget_coros:
            if random.random() < CANCELLATION_PROBABILITY:
                coro.cancel()
                cancelled_count += 1
    
    # Start cancellation task
    cancel_task = asyncio.create_task(cancel_random_ultrakill_requests())
    
    # Wait only for tracked tasks
    if tracked_tasks:
        tracked_results = await asyncio.gather(*all_coros[:len(tracked_tasks)])
        tracked_success = sum(1 for r in tracked_results if r.get("success"))
    else:
        tracked_results = []
        tracked_success = 0
    
    # Wait for cancellation task
    await cancel_task
    
    duration = time.time() - t0
    
    logger.info(f"ULTRAKILL iteration {iteration} completed in {duration:.2f}s")
    logger.info(f"Tracked requests: {tracked_success}/{len(tracked_tasks)} succeeded")
    logger.info(f"Fire-and-forget cancelled: {cancelled_count}/{len(fire_and_forget_tasks)} ({cancelled_count/max(1, len(fire_and_forget_tasks))*100:.1f}%)")
    if len(tracked_tasks) > 0:
        logger.info(f"Throughput (tracked): {len(tracked_tasks)/duration:.2f} req/s")
    logger.info(f"Total throughput (all): {PARALLEL_REQUESTS_PER_CYCLE/duration:.2f} req/s (estimated)")
    
    # Assertions - be lenient since we're causing chaos
    if len(tracked_tasks) > 0:
        success_rate = tracked_success / len(tracked_tasks)
        assert success_rate > 0.1, f"Too many failures in ULTRAKILL: only {success_rate:.1%} succeeded"
    
    # Save ultrakill stats
    pytest.ultrakill_stats = getattr(pytest, 'ultrakill_stats', [])
    pytest.ultrakill_stats.append({
        "force_calculate": force_calculate,
        "iteration": iteration,
        "tracked_success": tracked_success,
        "tracked_total": len(tracked_tasks),
        "fire_and_forget_total": len(fire_and_forget_tasks),
        "cancelled_count": cancelled_count,
        "duration": duration,
        "request_types": {
            "health": request_types.count("health"),
            "plot_valid": request_types.count("plot_valid"),
            "plot_invalid": request_types.count("plot_invalid"),
            "process_forced": request_types.count("process_forced")
        }
    })
