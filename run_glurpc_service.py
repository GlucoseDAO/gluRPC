import sys
import os
import signal
import time
import subprocess
import logging
import pathlib
from typing import Optional, Any

import typer

from service import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s"
)
log = logging.getLogger("run_glurpc_service")

app = typer.Typer()


@app.command()
def main(
    grpc: bool = typer.Option(
        False,
        "--grpc",
        help="Enable gRPC service"
    ),
    rest: bool = typer.Option(
        False,
        "--rest",
        help="Enable REST service"
    ),
    combined: bool = typer.Option(
        False,
        "--combined",
        help="Enable both gRPC and REST services (alias for --grpc --rest)"
    ),
) -> None:
    """
    Run gluRPC REST/gRPC service.
    
    Examples:
        # Combined service (both gRPC and REST, default recommended)
        glurpc-combined --combined
        
        # REST only
        glurpc-combined --rest
        
        # gRPC only
        glurpc-combined --grpc
        
        # Both services separately
        glurpc-combined --grpc --rest
    """
    # Handle combined flag as alias for both grpc and rest
    if combined:
        grpc = True
        rest = True
    
    # Default behavior: if no flags specified, run combined mode
    if not grpc and not rest:
        log.info("No service flags specified, defaulting to combined mode (gRPC + REST)")
        grpc = True
        rest = True
        combined = True
    
    # Determine service mode
    run_combined = grpc and rest
    run_grpc_only = grpc and not rest
    run_rest_only = rest and not grpc
    
    log.info(f"Starting gluRPC service - gRPC: {grpc}, REST: {rest}, Combined: {run_combined}")
    
    root_path = pathlib.Path(__file__).absolute().parent
    
    # Choose service module based on mode
    if run_combined:
        service_modules = ["service.combined_service"]
    else:
        # Separate processes mode
        service_modules = ["service.glurpc_service"]
    
    # Start all services
    all_p = start_all_services(
        root_path,
        service_modules,
        run_grpc_only,
        run_rest_only,
        run_combined
    )
    
    # Flag to track if we're in shutdown mode
    shutdown_initiated = False
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_initiated
        if shutdown_initiated:
            log.warning("Shutdown already in progress, ignoring signal")
            return
        shutdown_initiated = True
        sig_name = signal.Signals(signum).name
        log.info(f"Received {sig_name}, initiating graceful shutdown...")
        kill_and_exit(all_p, exit_code=0, graceful=True)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Continuous checking all subprocess
    try:
        while True:
            for p in all_p:
                p.poll()
                if p.returncode is not None:
                    proc_type = getattr(p, '_process_type', 'unknown')
                    # If process exited, check if it was graceful (exit code 0, 130 for SIGINT, 143 for SIGTERM)
                    if p.returncode in [0, 130, 143]:
                        log.info(f"Process {proc_type} (PID {p.pid}) exited gracefully with code {p.returncode}")
                        # Process exited gracefully, likely due to signal we sent
                        # Don't treat this as an error
                        if not shutdown_initiated:
                            shutdown_initiated = True
                            log.info("Process exited, initiating graceful shutdown of remaining processes...")
                            kill_and_exit(all_p, exit_code=0, graceful=True)
                    else:
                        log.error(f"Process {proc_type} (PID {p.pid}) exited unexpectedly with code {p.returncode}")
                        if not shutdown_initiated:
                            shutdown_initiated = True
                            kill_and_exit(all_p, exit_code=1, graceful=False)
            time.sleep(1)
    except KeyboardInterrupt:
        if not shutdown_initiated:
            shutdown_initiated = True
            log.info("Received keyboard interrupt, shutting down...")
            kill_and_exit(all_p, exit_code=0, graceful=True)
        else:
            log.warning("Keyboard interrupt during shutdown, waiting for cleanup...")
    except Exception as e:
        log.error(f"Error in main loop: {e}")
        if not shutdown_initiated:
            shutdown_initiated = True
            kill_and_exit(all_p, exit_code=1, graceful=False)
        else:
            log.error("Exception during shutdown, forcing immediate exit")
            sys.exit(1)


def start_all_services(
    cwd: pathlib.Path,
    service_modules: list[str],
    grpc_only: bool,
    rest_only: bool,
    combined: bool
) -> list[subprocess.Popen]:
    """
    Loop through all service_modules and start them.
    """
    all_p = []
    for i, service_module in enumerate(service_modules):
        service_name = service_module.split(".")[-1]
        log.info(f"Launching {service_module} on gRPC port {registry[service_name]['grpc']}, REST port {registry[service_name]['rest']}")
        all_p += start_service(
            cwd,
            service_module,
            grpc_only,
            rest_only,
            combined
        )
    return all_p


def start_service(
    cwd: pathlib.Path,
    service_module: str,
    grpc_only: bool,
    rest_only: bool,
    combined: bool
) -> list[subprocess.Popen]:
    """
    Starts the gRPC service and/or REST service.
    
    Args:
        combined: If True, run both gRPC and REST in the same process (recommended)
    """
    all_p = []
    
    service_name = service_module.split(".")[-1]
    grpc_port = registry[service_name]["grpc"]
    rest_port = registry[service_name]["rest"]
    
    # Combined mode: run both gRPC and REST in the same process
    # This uses service.combined_service which includes both servers
    if combined:
        log.info(f"Starting combined gRPC+REST service on ports gRPC={grpc_port}, REST={rest_port}")
        p_combined = subprocess.Popen(
            [
                sys.executable,
                "-m",
                service_module,
                "--grpc-port",
                str(grpc_port),
                "--rest-port",
                str(rest_port)
            ],
            cwd=str(cwd),
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=0  # Unbuffered
        )
        p_combined._process_type = "combined_service"  # type: ignore
        all_p.append(p_combined)
        return all_p
    
    # Separate processes mode (original behavior)
    # Start gRPC service (unless rest_only)
    if not rest_only:
        log.info(f"Starting gRPC service on port {grpc_port}")
        p_grpc = subprocess.Popen(
            [sys.executable, "-m", service_module, "--grpc-port", str(grpc_port)],
            cwd=str(cwd),
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=0
        )
        p_grpc._process_type = "grpc_service"  # type: ignore
        all_p.append(p_grpc)
    
    # Start REST service (unless grpc_only)
    if not grpc_only:
        log.info(f"Starting REST service on port {rest_port}")
        # Use uvicorn to run the FastAPI app
        p_rest = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "glurpc.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(rest_port)
            ],
            cwd=str(cwd),
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=0
        )
        p_rest._process_type = "rest_service"  # type: ignore
        all_p.append(p_rest)
    
    return all_p


def kill_and_exit(all_p: list[subprocess.Popen], exit_code: int = 0, graceful: bool = True) -> None:
    """Gracefully shutdown all subprocesses."""
    if not all_p:
        sys.exit(exit_code)
    
    log.info(f"Shutting down {len(all_p)} process(es)...")
    
    # Log what we're shutting down
    for p in all_p:
        proc_type = getattr(p, '_process_type', 'unknown')
        log.info(f"  - {proc_type} (PID {p.pid})")
    
    # Send SIGTERM to all processes
    for p in all_p:
        if p.poll() is None:  # Process is still running
            try:
                proc_type = getattr(p, '_process_type', 'unknown')
                log.info(f"Sending SIGTERM to {proc_type} (PID {p.pid})")
                p.terminate()  # Sends SIGTERM
            except Exception as e:
                log.error(f"Error terminating process {p.pid}: {e}")
    
    # Wait for processes to terminate gracefully (up to 10 seconds)
    log.info("Waiting for processes to terminate gracefully...")
    deadline = time.time() + 10
    
    while time.time() < deadline:
        all_terminated = True
        for p in all_p:
            if p.poll() is None:
                all_terminated = False
                break
        
        if all_terminated:
            log.info("All processes terminated gracefully")
            # Exit with 0 if this was a graceful shutdown, otherwise use provided exit_code
            sys.exit(0 if graceful else exit_code)
        
        time.sleep(0.5)
    
    # Force kill any remaining processes
    log.warning("Timeout waiting for graceful shutdown, forcing kill...")
    for p in all_p:
        if p.poll() is None:
            try:
                proc_type = getattr(p, '_process_type', 'unknown')
                log.warning(f"Sending SIGKILL to {proc_type} (PID {p.pid})")
                p.kill()  # Sends SIGKILL
            except Exception as e:
                log.error(f"Error killing process {p.pid}: {e}")
    
    # Wait a bit more for forced kills
    time.sleep(1)
    
    # Final status check
    for p in all_p:
        status = "terminated" if p.poll() is not None else "still running!"
        proc_type = getattr(p, '_process_type', 'unknown')
        log.info(f"  {proc_type} (PID {p.pid}): {status}")
    
    # Exit with non-zero if we had to force kill (not graceful)
    sys.exit(1 if not graceful else 0)




if __name__ == "__main__":
    app()

