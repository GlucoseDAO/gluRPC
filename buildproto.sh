#!/bin/bash
# Build protocol buffers for gluRPC service

# Run protoc from src/ directory to avoid 'src' prefix in imports
cd src
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/glurpc.proto
cd ..

# Fix any remaining import issues (replace 'src.service' with 'service')
#sed -i 's/from src\.service\.service_spec import/from service.service_spec import/g' src/service/service_spec/glurpc_pb2_grpc.py
#sed -i 's/import src\.service\.service_spec/import service.service_spec/g' src/service/service_spec/glurpc_pb2_grpc.py
#sed -i 's/src_dot_service_dot_service__spec_dot_glurpc__pb2/service_dot_service__spec_dot_glurpc__pb2/g' src/service/service_spec/glurpc_pb2_grpc.py

echo "✅ Protocol buffers compiled successfully"

