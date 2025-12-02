# Deployment Configurations

This directory contains **untested** deployment configuration examples for various scenarios. These are provided as starting points and should be customized and tested for your specific environment.

## ⚠️ Warning

**These configurations have not been tested in production environments.** Please review, customize, and thoroughly test them before deploying to production.

## Directory Structure

```
deployment/
├── systemd/           # Linux systemd service configuration
├── nginx/             # Nginx reverse proxy configuration
├── docker/            # Docker and docker-compose files
├── kubernetes/        # Kubernetes manifests
└── aws/              # AWS ECS configuration
```

---

## Deployment Scenarios

### Scenario 1: Single-Server Production

**Use Case**: Small to medium production deployment on a single server

#### With Systemd

1. Copy the systemd service file from `systemd/glurpc.service` to `/etc/systemd/system/glurpc.service`
2. Customize the file for your environment (paths, user, environment variables)
3. Create API keys file:

```bash
# /opt/glurpc/api_keys_list
production-key-1-abc123xyz
production-key-2-def456uvw
```

4. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable glurpc
sudo systemctl start glurpc
sudo systemctl status glurpc
```

See [`systemd/glurpc.service`](systemd/glurpc.service) for the full configuration.

#### With Nginx Reverse Proxy

Copy the nginx configuration from `nginx/glurpc.conf` to `/etc/nginx/sites-available/glurpc` and customize for your domain.

```bash
sudo ln -s /etc/nginx/sites-available/glurpc /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

See [`nginx/glurpc.conf`](nginx/glurpc.conf) for the full configuration.

---

### Scenario 2: Docker Deployment

**Use Case**: Containerized deployment for portability and isolation

See the [`docker/`](docker/) directory for Dockerfile and docker-compose.yml.

#### Quick Start

```bash
cd docker

# Build
docker-compose build

# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Clean restart
docker-compose down -v && docker-compose up -d
```

**Note**: Customize the `docker-compose.yml` file with your environment variables and volume mounts before deploying.

---

### Scenario 3: Kubernetes Deployment

**Use Case**: Multi-node cluster with auto-scaling and high availability

See the [`kubernetes/`](kubernetes/) directory for all Kubernetes manifests:
- `deployment.yaml` - Main application deployment
- `service.yaml` - Service configuration
- `ingress.yaml` - Ingress with TLS

#### Quick Start

```bash
# Create namespace
kubectl create namespace production

# Create API keys secret
kubectl create secret generic glurpc-api-keys \
  --from-file=api_keys_list=./api_keys_list \
  --namespace=production

# Apply manifests
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml

# Check status
kubectl get pods -n production
kubectl get svc -n production
```

**Note**: Customize the manifests with your registry, resource limits, and domain before deploying.

---

### Scenario 4: Cloud Deployment (AWS)

**Use Case**: Managed cloud deployment with auto-scaling

#### Using AWS ECS (Fargate)

See [`aws/task-definition.json`](aws/task-definition.json) for the ECS task definition.

1. **Create ECR Repository**:
```bash
aws ecr create-repository --repository-name glurpc
```

2. **Build and Push Image**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -t glurpc -f docker/Dockerfile ..
docker tag glurpc:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/glurpc:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/glurpc:latest
```

3. **Register Task Definition**:
```bash
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json
```

4. **Create Service**:
```bash
aws ecs create-service \
  --cluster glurpc-cluster \
  --service-name glurpc-service \
  --task-definition glurpc \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=glurpc,containerPort=8000"
```

**Note**: Customize the task definition with your account ID, region, and secrets before deploying.

---

### Scenario 5: Multi-GPU Server

**Use Case**: High-performance server with multiple GPUs

**Configuration**:
```bash
# Set model copies per GPU
export NUM_COPIES_PER_DEVICE=3  # 3 copies per GPU

# With 4 GPUs: 4 * 3 = 12 total model copies
# Automatically distributes across GPUs

# Start server
uv run uvicorn glurpc.app:app --host 0.0.0.0 --port 8000
```

**Expected Behavior**:
- Models distributed round-robin across GPUs (cuda:0, cuda:1, cuda:2, cuda:3)
- Queue pool size = 12 models
- Throughput scales linearly with GPU count

---

## Customization Checklist

Before using any configuration file:

- [ ] Replace placeholder values (server names, account IDs, etc.)
- [ ] Adjust resource limits for your hardware
- [ ] Configure appropriate security settings
- [ ] Set up monitoring and logging
- [ ] Test in a staging environment first
- [ ] Review and adjust scaling parameters

## Contributing

If you've tested and improved any of these configurations, please consider contributing your changes back to the project

