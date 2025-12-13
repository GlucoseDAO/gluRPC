import argparse
import os.path
import time

from service import registry


def common_parser(script_name: str) -> argparse.ArgumentParser:
    """Create argument parser for service scripts."""
    parser = argparse.ArgumentParser(prog=script_name)
    service_name = os.path.splitext(os.path.basename(script_name))[0]
    parser.add_argument(
        "--grpc-port",
        help="port to bind gRPC service to",
        default=registry[service_name]['grpc'],
        type=int,
        required=False
    )
    parser.add_argument(
        "--rest-port",
        help="port to bind REST service to (if applicable)",
        default=registry.get(service_name, {}).get('rest', 8000),
        type=int,
        required=False
    )
    return parser


def main_loop(grpc_handler, args):
    """
    Start gRPC server and run until interrupted.
    From gRPC docs: Because start() does not block you may need to 
    sleep-loop if there is nothing else for your code to do while serving.
    """
    server = grpc_handler(port=args.grpc_port)
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)

