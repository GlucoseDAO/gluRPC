import sys
import os
import signal
import time
import subprocess
import logging
import pathlib
import glob
import json
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
    daemon: bool = typer.Option(
        False,
        "--daemon",
        help="Enable SNET daemon (requires --grpc or --combined)"
    ),
    daemon_config: Optional[str] = typer.Option(
        None,
        "--daemon-config",
        help="Path to SNET daemon configuration file (e.g., /app/snetd_configs/snetd.sepolia.json)"
    ),
    ssl: bool = typer.Option(
        False,
        "--ssl",
        help="Enable SSL for daemon (requires --daemon)"
    ),
) -> None:
    """
    Run gluRPC REST/gRPC service with optional SNET daemon.
    
    Examples:
        # Combined service (both gRPC and REST, default recommended)
        glurpc-combined --combined
        
        # REST only
        glurpc-combined --rest
        
        # gRPC only
        glurpc-combined --grpc
        
        # Both services separately
        glurpc-combined --grpc --rest
        
        # With SNET daemon
        glurpc-combined --combined --daemon --daemon-config /app/snetd_configs/snetd.sepolia.json
        
        # With SSL
        glurpc-combined --combined --daemon --daemon-config /app/snetd_configs/snetd.sepolia.json --ssl
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
    
    # Validate dependencies
    if daemon and not grpc:
        log.error("Error: --daemon requires --grpc or --combined (daemon needs gRPC service)")
        raise typer.Exit(code=1)
    
    if ssl and not daemon:
        log.error("Error: --ssl requires --daemon")
        raise typer.Exit(code=1)
    
    if daemon and not daemon_config:
        log.warning("Warning: --daemon specified without --daemon-config, will search for configs in standard locations")
    
    # Determine service mode
    run_combined = grpc and rest
    run_grpc_only = grpc and not rest
    run_rest_only = rest and not grpc
    
    log.info(f"Starting gluRPC service - gRPC: {grpc}, REST: {rest}, Combined: {run_combined}, Daemon: {daemon}")
    
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
        daemon,
        daemon_config,
        ssl,
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
    daemon: bool,
    daemon_config: Optional[str],
    ssl: bool,
    grpc_only: bool,
    rest_only: bool,
    combined: bool
) -> list[subprocess.Popen]:
    """
    Loop through all service_modules and start them.
    For each one, an instance of SNET Daemon "snetd" is created (if enabled).
    snetd will start with configs from "snetd_configs/*.json"
    """
    all_p = []
    for i, service_module in enumerate(service_modules):
        service_name = service_module.split(".")[-1]
        log.info(f"Launching {service_module} on gRPC port {registry[service_name]['grpc']}, REST port {registry[service_name]['rest']}")
        all_p += start_service(
            cwd,
            service_module,
            daemon,
            daemon_config,
            ssl,
            grpc_only,
            rest_only,
            combined
        )
    return all_p


def start_service(
    cwd: pathlib.Path,
    service_module: str,
    daemon: bool,
    daemon_config: Optional[str],
    ssl: bool,
    grpc_only: bool,
    rest_only: bool,
    combined: bool
) -> list[subprocess.Popen]:
    """
    Starts SNET Daemon ("snetd"), the gRPC service, and the REST service.
    
    Args:
        combined: If True, run both gRPC and REST in the same process (recommended)
        
    Paths:
        - Docker: /app/snetd_configs/*.json
        - Local: ./snetd_configs/*.json
        - SSL certs: /app/.certs/ (Docker) or /etc/letsencrypt/live/domain/ (host)
        - ETCD data: /app/etcd/{network}/ (Docker, persistent volume)
    """
    
    def add_ssl_configs(conf: str) -> None:
        """
        Add SSL certificate paths to snetd config.
        Paths are set for Docker environment (/app/.certs/).
        """
        with open(conf, "r") as f:
            snetd_configs = json.load(f)
            snetd_configs["ssl_cert"] = "/app/.certs/fullchain.pem"
            snetd_configs["ssl_key"] = "/app/.certs/privkey.pem"
        with open(conf, "w") as f:
            json.dump(snetd_configs, f, sort_keys=True, indent=4)
    
    def get_config_search_paths() -> list[str]:
        """
        Return possible paths to search for SNET daemon configs.
        Checks Docker paths first, then local paths.
        """
        docker_path = "/app/snetd_configs/*.json"
        local_path = "./snetd_configs/*.json"
        cwd_path = str(cwd / "snetd_configs" / "*.json")
        
        # Return paths that exist
        for search_path in [docker_path, local_path, cwd_path]:
            matches = glob.glob(search_path)
            if matches:
                log.info(f"Found SNET configs in: {search_path}")
                return matches
        
        log.warning("No SNET daemon config files found in standard locations")
        return []
    
    all_p = []
    
    # Start SNET Daemon if enabled
    if daemon:
        if daemon_config:
            # Use specified config file
            config_path = pathlib.Path(daemon_config)
            if not config_path.exists():
                log.error(f"Daemon config file not found: {daemon_config}")
                log.info(f"Looking for config at: {config_path.absolute()}")
            else:
                if ssl:
                    add_ssl_configs(daemon_config)
                all_p.append(start_snetd(str(cwd), daemon_config))
        else:
            # Auto-discover config files
            config_files = get_config_search_paths()
            if not config_files:
                log.warning("No daemon config files found. Daemon will not start.")
                log.info("Create config files in /app/snetd_configs/ (Docker) or ./snetd_configs/ (local)")
            else:
                for config_file in config_files:
                    if ssl:
                        add_ssl_configs(config_file)
                    all_p.append(start_snetd(str(cwd), config_file))
    
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


def start_snetd(cwd: str, config_file: Optional[str] = None) -> subprocess.Popen:
    """
    Starts the SNET Daemon "snetd".
    
    The daemon binary should be in PATH (installed via Dockerfile or manually).
    Config file should point to valid JSON configuration with:
    - blockchain settings
    - service endpoint (should match gRPC service port)
    - ETCD data_dir (e.g., /app/etcd/sepolia)
    - SSL certificates (if enabled)
    
    Args:
        cwd: Working directory for the process
        config_file: Path to SNET daemon JSON config (optional)
    
    Returns:
        subprocess.Popen instance of the running daemon
    """
    cmd = ["snetd", "serve"]
    if config_file:
        cmd.extend(["--config", config_file])
        log.info(f"Starting SNET daemon with config: {config_file}")
    else:
        log.info("Starting SNET daemon with default config (snetd.config.json)")
    
    proc = subprocess.Popen(
        cmd, 
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=0
    )
    # Tag the process so we can identify it during shutdown
    proc._process_type = "snetd"  # type: ignore
    proc._config_file = config_file  # type: ignore
    return proc


def kill_and_exit(all_p: list[subprocess.Popen], exit_code: int = 0, graceful: bool = True) -> None:
    """Gracefully shutdown all subprocesses including SNET daemon."""
    if not all_p:
        sys.exit(exit_code)
    
    log.info(f"Shutting down {len(all_p)} process(es)...")
    
    # Log what we're shutting down
    for p in all_p:
        proc_type = getattr(p, '_process_type', 'unknown')
        if proc_type == 'snetd':
            config = getattr(p, '_config_file', 'default')
            log.info(f"  - SNET daemon (PID {p.pid}, config: {config})")
        else:
            log.info(f"  - {proc_type} (PID {p.pid})")
    
    # Send SIGTERM to all processes (including daemon)
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
            log.info("All processes (including daemon) terminated gracefully")
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

