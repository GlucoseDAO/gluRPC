import sys
import os
import signal
import time
import subprocess
import logging
import pathlib
import glob
import json
from typing import Optional

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
    no_daemon: bool = typer.Option(
        False,
        "--no-daemon",
        help="Do not start the SNET daemon"
    ),
    daemon_config: Optional[str] = typer.Option(
        None,
        "--daemon-config",
        help="Path of SNET daemon configuration file, without config it won't be started"
    ),
    ssl: bool = typer.Option(
        False,
        "--ssl",
        help="Start the daemon with SSL"
    ),
    grpc_only: bool = typer.Option(
        False,
        "--grpc-only",
        help="Run only gRPC service (no REST)"
    ),
    rest_only: bool = typer.Option(
        False,
        "--rest-only",
        help="Run only REST service (no gRPC)"
    ),
    combined: bool = typer.Option(
        False,
        "--combined",
        help="Run both gRPC and REST in the same process (recommended)"
    )
) -> None:
    """Run gluRPC combined REST/gRPC service."""
    run_daemon = not no_daemon
    
    root_path = pathlib.Path(__file__).absolute().parent
    
    # All services modules go here
    service_modules = ["service.glurpc_service"]
    
    # Call for all the services listed in service_modules
    all_p = start_all_services(
        root_path,
        service_modules,
        run_daemon,
        daemon_config,
        ssl,
        grpc_only,
        rest_only,
        combined
    )
    
    # Continuous checking all subprocess
    try:
        while True:
            for p in all_p:
                p.poll()
                if p.returncode and p.returncode != 0:
                    log.error(f"Process {p.pid} exited with code {p.returncode}")
                    kill_and_exit(all_p)
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Received keyboard interrupt, shutting down...")
        kill_and_exit(all_p)
    except Exception as e:
        log.error(f"Error in main loop: {e}")
        kill_and_exit(all_p)
        raise


def start_all_services(
    cwd: pathlib.Path,
    service_modules: list[str],
    run_daemon: bool,
    daemon_config: Optional[str],
    run_ssl: bool,
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
            run_daemon,
            daemon_config,
            run_ssl,
            grpc_only,
            rest_only,
            combined
        )
    return all_p


def start_service(
    cwd: pathlib.Path,
    service_module: str,
    run_daemon: bool,
    daemon_config: Optional[str],
    run_ssl: bool,
    grpc_only: bool,
    rest_only: bool,
    combined: bool
) -> list[subprocess.Popen]:
    """
    Starts SNET Daemon ("snetd"), the gRPC service, and the REST service.
    
    Args:
        combined: If True, run both gRPC and REST in the same process (recommended)
    """
    
    def add_ssl_configs(conf):
        """Add SSL keys to snetd.config.json"""
        with open(conf, "r") as f:
            snetd_configs = json.load(f)
            snetd_configs["ssl_cert"] = "/opt/singnet/.certs/fullchain.pem"
            snetd_configs["ssl_key"] = "/opt/singnet/.certs/privkey.pem"
        with open(conf, "w") as f:
            json.dump(snetd_configs, f, sort_keys=True, indent=4)
    
    all_p = []
    
    # Start SNET Daemon if enabled
    if run_daemon:
        if daemon_config:
            all_p.append(start_snetd(str(cwd), daemon_config))
        else:
            for idx, config_file in enumerate(glob.glob("./snetd_configs/*.json")):
                if run_ssl:
                    add_ssl_configs(config_file)
                all_p.append(start_snetd(str(cwd), config_file))
    
    service_name = service_module.split(".")[-1]
    grpc_port = registry[service_name]["grpc"]
    rest_port = registry[service_name]["rest"]
    
    # Combined mode: run both gRPC and REST in the same process
    if combined:
        log.info(f"Starting combined gRPC+REST service on ports gRPC={grpc_port}, REST={rest_port}")
        p_combined = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "service.combined_service",
                "--grpc-port",
                str(grpc_port),
                "--rest-port",
                str(rest_port)
            ],
            cwd=str(cwd)
        )
        all_p.append(p_combined)
        return all_p
    
    # Separate processes mode (original behavior)
    # Start gRPC service (unless rest_only)
    if not rest_only:
        log.info(f"Starting gRPC service on port {grpc_port}")
        p_grpc = subprocess.Popen(
            [sys.executable, "-m", service_module, "--grpc-port", str(grpc_port)],
            cwd=str(cwd)
        )
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
            cwd=str(cwd)
        )
        all_p.append(p_rest)
    
    return all_p


def start_snetd(cwd: pathlib.Path, config_file: Optional[str] = None) -> subprocess.Popen:
    """
    Starts the SNET Daemon "snetd":
    """
    cmd = ["snetd", "serve"]
    if config_file:
        cmd = ["snetd", "serve", "--config", config_file]
    log.info(f"Starting SNET daemon with config: {config_file}")
    return subprocess.Popen(cmd, cwd=str(cwd))


def kill_and_exit(all_p: list[subprocess.Popen]) -> None:
    """Kill all subprocesses and exit."""
    for p in all_p:
        try:
            log.info(f"Terminating process {p.pid}")
            os.kill(p.pid, signal.SIGTERM)
        except Exception as e:
            log.error(f"Error killing process {p.pid}: {e}")
    exit(1)


if __name__ == "__main__":
    app()

