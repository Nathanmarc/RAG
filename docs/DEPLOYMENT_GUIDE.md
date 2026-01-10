# Deployment Guide

## Table of Contents
1. [Local Development Deployment](#local-development)
2. [Docker & Docker Compose Deployment](#docker-deployment)
3. [AWS Deployment](#aws-deployment)
4. [GCP Deployment](#gcp-deployment)
5. [Azure Deployment](#azure-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Configuration Management](#configuration)
8. [Rollback Procedures](#rollback)
9. [Monitoring & Health Checks](#monitoring)

---

## Local Development Deployment

### Prerequisites
```bash
# Required
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional (for local LLM)
# Download from https://ollama.ai
ollama pull llama2
```

### Step-by-Step Setup

#### 1. Initialize Data Pipeline
```bash
# Extract PDF data
python extract_pdf.py

# Preprocess text
python preprocess.py

# Generate chunks (tests multiple configs)
python chunk_text.py

# Create embeddings
python embed.py

# Initialize vector database
python init_vectordb.py
```

#### 2. Build RAG System
```bash
# Build vector database from processed data
python rag_engine.py --build
```

#### 3. Start MLflow Server (Optional)
```bash
mlflow server --host 0.0.0.0 --port 5000
# Access at http://localhost:5000
```

#### 4. Run Interactive Queries
```bash
# Test via CLI
python rag_engine.py --query "How many YOLOv8 streams can T4 process?"
```

#### 5. Start REST API
```bash
# Terminal 1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Access Swagger docs: http://localhost:8000/docs
```

#### 6. Start Streamlit UI
```bash
# Terminal 2
streamlit run app.py --server.port 8501

# Access UI: http://localhost:8501
```

#### 7. Run MLflow Experiments
```bash
# Terminal 3
python mlflow_experiment.py
```

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Services Available
- RAG API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)
- ChromaDB: `http://localhost:8001`

### Custom Docker Build

```bash
# Build image with custom tag
docker build -t t4-consultant:latest .

# Run with volume mount
docker run \
  -p 8000:8000 \
  -v $(pwd)/rag_chroma_db:/app/rag_chroma_db \
  -v $(pwd)/data:/app/data \
  -e LOG_LEVEL=INFO \
  t4-consultant:latest

# Build with buildkit for better caching
DOCKER_BUILDKIT=1 docker build -t t4-consultant:latest .
```

### Multi-Stage Docker Build Optimization

The provided `Dockerfile` uses multi-stage builds:
- **Builder stage**: Installs dependencies
- **Runtime stage**: Copies only necessary artifacts
- Result: ~60% smaller final image

### Docker Compose Configuration

Edit `docker-compose.yml` to adjust:
```yaml
services:
  rag-api:
    environment:
      - LOG_LEVEL=INFO
      - TOP_K_RETRIEVAL=5
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

---

## AWS Deployment

### Option 1: AWS ECS (Recommended for Production)

#### Step 1: Create ECR Repository
```bash
aws ecr create-repository \
  --repository-name t4-consultant \
  --region us-east-1
```

#### Step 2: Build & Push Image
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t t4-consultant:latest .

# Tag image
docker tag t4-consultant:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/t4-consultant:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/t4-consultant:latest
```

#### Step 3: Create ECS Task Definition
```json
{
  "family": "t4-consultant",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "t4-consultant",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/t4-consultant:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/t4-consultant",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "mountPoints": [
        {
          "sourceVolume": "rag-data",
          "containerPath": "/app/rag_chroma_db"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "rag-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

#### Step 4: Create ECS Service
```bash
aws ecs create-service \
  --cluster t4-consultant \
  --service-name t4-api \
  --task-definition t4-consultant:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=t4-consultant,containerPort=8000
```

#### Step 5: Set Up Auto-Scaling
```bash
# Target tracking scaling policy
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/t4-consultant/t4-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --policy-name cpu-scaling \
  --service-namespace ecs \
  --resource-id service/t4-consultant/t4-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    TargetValue=70.0,PredefinedMetricSpecification='{PredefinedMetricType=ECSServiceAverageCPUUtilization}'
```

### Option 2: AWS Lambda (Serverless)

```bash
# Requires Docker image < 256MB or ZIP < 50MB
# Package and deploy
zip -r lambda-package.zip . -x "*.git*" "__pycache__/*" "*.pyc" "venv/*"

aws lambda create-function \
  --function-name t4-consultant \
  --runtime python3.11 \
  --role arn:aws:iam::...:role/lambda-role \
  --handler main.handler \
  --zip-file fileb://lambda-package.zip \
  --timeout 300 \
  --memory-size 3008
```

### Option 3: AWS SageMaker

```bash
# Create SageMaker endpoint
aws sagemaker create-model \
  --model-name t4-consultant \
  --primary-container Image=<account-id>.dkr.ecr.us-east-1.amazonaws.com/t4-consultant:latest

aws sagemaker create-endpoint-config \
  --endpoint-config-name t4-config \
  --production-variants VariantName=primary,ModelName=t4-consultant,InitialInstanceCount=1,InstanceType=ml.m5.large

aws sagemaker create-endpoint \
  --endpoint-name t4-endpoint \
  --endpoint-config-name t4-config
```

---

## GCP Deployment

### Option 1: Cloud Run (Recommended)

```bash
# Authenticate
gcloud auth login
gcloud config set project <project-id>

# Enable services
gcloud services enable run.googleapis.com

# Build and deploy
gcloud run deploy t4-consultant \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --set-env-vars LOG_LEVEL=INFO,TOP_K_RETRIEVAL=5
```

### Option 2: GKE (Kubernetes)

```bash
# Create GKE cluster
gcloud container clusters create t4-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --region us-central1

# Push image to Container Registry
gcloud builds submit --tag gcr.io/<project-id>/t4-consultant:latest

# Deploy to GKE (see Kubernetes section below)
```

### Option 3: Compute Engine (VMs)

```bash
# Create instance
gcloud compute instances create t4-vm \
  --image-family ubuntu-2204-lts \
  --image-project ubuntu-os-cloud \
  --machine-type n1-standard-4 \
  --zone us-central1-a

# SSH and deploy
gcloud compute ssh t4-vm --zone us-central1-a

# On VM:
git clone <repo>
cd ko
pip install -r requirements.txt
docker-compose up -d
```

---

## Azure Deployment

### Option 1: Azure Container Instances

```bash
# Create resource group
az group create --name t4-rg --location eastus

# Create container registry
az acr create \
  --resource-group t4-rg \
  --name t4registry \
  --sku Basic

# Build and push image
az acr build \
  --registry t4registry \
  --image t4-consultant:latest \
  .

# Deploy container instance
az container create \
  --resource-group t4-rg \
  --name t4-consultant \
  --image t4registry.azurecr.io/t4-consultant:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --registry-login-server t4registry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --environment-variables LOG_LEVEL=INFO
```

### Option 2: Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group t4-rg \
  --name t4-cluster \
  --node-count 3 \
  --vm-set-type VirtualMachineScaleSets \
  --load-balancer-sku standard

# Get credentials
az aks get-credentials \
  --resource-group t4-rg \
  --name t4-cluster

# Deploy (see Kubernetes section)
```

### Option 3: Azure App Service (Web Apps)

```bash
# Create App Service plan
az appservice plan create \
  --name t4-plan \
  --resource-group t4-rg \
  --sku B4 \
  --is-linux

# Create web app
az webapp create \
  --resource-group t4-rg \
  --plan t4-plan \
  --name t4-consultant \
  --deployment-container-image-name t4registry.azurecr.io/t4-consultant:latest
```

---

## Kubernetes Deployment

### Prerequisites
```bash
# Install kubectl
# Install helm (optional, for package management)

# Get cluster credentials
kubectl config current-context
```

### Deployment YAML

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: t4-consultant
  namespace: default
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: t4-consultant
  template:
    metadata:
      labels:
        app: t4-consultant
    spec:
      containers:
      - name: api
        image: <registry>/t4-consultant:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: TOP_K_RETRIEVAL
          value: "5"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: rag-data
          mountPath: /app/rag_chroma_db
      volumes:
      - name: rag-data
        persistentVolumeClaim:
          claimName: rag-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: t4-consultant-svc
spec:
  type: LoadBalancer
  selector:
    app: t4-consultant
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: t4-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: t4-consultant
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace t4-prod

# Create persistent volume (example for local development)
kubectl apply -f k8s/pvc.yaml

# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n t4-prod
kubectl logs -f deployment/t4-consultant -n t4-prod

# Port forward for testing
kubectl port-forward svc/t4-consultant-svc 8000:80
```

---

## Configuration Management

### Environment Variables

Create `.env` file:
```bash
# LLM Configuration
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# Vector Database
VECTORDB_PATH=rag_chroma_db
EMBEDDING_MODEL=intfloat/e5-large-v2
TOP_K_RETRIEVAL=5
RERANK_THRESHOLD=0.3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
RATE_LIMIT_PER_MINUTE=60

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=T4_RAG_Experiment

# Monitoring
PROMETHEUS_PORT=8001
ENABLE_MONITORING=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Feature Flags
ENABLE_CONFIDENCE_SCORING=true
ENABLE_SOURCE_CITATION=true
ENABLE_RERANKING=true
```

### Secrets Management

**AWS Secrets Manager:**
```bash
aws secretsmanager create-secret \
  --name t4-config \
  --secret-string file://config.json
```

**GCP Secret Manager:**
```bash
gcloud secrets create t4-config --data-file=config.json
```

**Azure Key Vault:**
```bash
az keyvault secret set \
  --vault-name t4-vault \
  --name t4-config \
  --file config.json
```

---

## Rollback Procedures

### Docker Compose Rollback
```bash
# List available images
docker images | grep t4-consultant

# Revert to previous version
docker-compose down
git checkout HEAD~1  # Go to previous commit
docker-compose up -d
```

### AWS ECS Rollback
```bash
# List task definitions
aws ecs list-task-definitions \
  --family-prefix t4-consultant \
  --sort DESC

# Update service to previous version
aws ecs update-service \
  --cluster t4-consultant \
  --service t4-api \
  --task-definition t4-consultant:1  # Previous version
```

### Kubernetes Rollback
```bash
# Check rollout history
kubectl rollout history deployment/t4-consultant

# Rollback to previous version
kubectl rollout undo deployment/t4-consultant

# Rollback to specific revision
kubectl rollout undo deployment/t4-consultant --to-revision=2
```

### GCP Cloud Run Rollback
```bash
# List revisions
gcloud run revisions list --service=t4-consultant

# Route traffic to previous revision
gcloud run services update-traffic t4-consultant \
  --to-revisions LATEST=0,<previous-revision>=100
```

---

## Monitoring & Health Checks

### Health Check Endpoint
```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-11-22T10:30:45Z",
  "version": "1.0.0"
}
```

### Prometheus Metrics
```bash
# Scrape metrics
curl http://localhost:8000/metrics

# Common metrics:
# - rag_queries_total
# - rag_query_duration_seconds
# - rag_query_errors_total
```

### Application Logging
```bash
# View logs
docker-compose logs -f rag-api

# View with tail
tail -f logs/app.log

# Stream to centralized logging
# (configure in logging.yaml)
```

### Performance Monitoring

**Latency Analysis:**
```bash
# Query latency histogram
curl http://localhost:9090/api/v1/query?query=histogram_quantile%280.95%2Crag_query_duration_seconds%29
```

**Error Tracking:**
```bash
# Check error rate
curl http://localhost:9090/api/v1/query?query=rate%28rag_query_errors_total%5B5m%5D%29
```

---

## Troubleshooting Deployment

### Issue: Container Won't Start
```bash
# Check logs
docker logs <container-id>

# Verify image
docker run -it <image> /bin/bash

# Test locally first
python -m venv test_venv
source test_venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Issue: Out of Memory
```bash
# Increase memory allocation
docker run -m 8g <image>

# Or in docker-compose.yml
services:
  rag-api:
    deploy:
      resources:
        limits:
          memory: 8G
```

### Issue: Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
docker run -p 9000:8000 <image>
```

### Issue: Network Connectivity
```bash
# Test from container
docker exec <container-id> curl http://localhost:8000/health

# Check Docker network
docker network inspect t4_default

# Restart services
docker-compose restart
```

---

**Last Updated:** November 2025
