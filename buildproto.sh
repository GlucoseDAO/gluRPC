#!/bin/bash
# Build protocol buffers for gluRPC service
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/glurpc.proto

