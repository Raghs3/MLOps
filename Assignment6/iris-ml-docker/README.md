# Containerizing a Machine Learning Model using Docker

## Aim

Containerize a trained Machine Learning model using Docker so that it can be
packaged, shipped, and run consistently across different environments, and
expose the model as a REST API service inside the container.

## Project structure

```
iris-ml-docker/
├── model/
│   └── train_model.py    # trains and saves the ML model
├── app.py                 # Flask app that serves the model
├── requirements.txt       # Python dependencies
├── Dockerfile              # instructions to build the image
├── iris_model.pkl          # generated after training
└── README.md
```

## Setup

This assignment uses the shared virtual environment at the repository root
(`MLOps/.venv`).

```powershell
# from the repository root
.venv\Scripts\activate
pip install -r Assignment6/iris-ml-docker/requirements.txt
```

## Step 1: Train and save the model

`model/train_model.py` trains a `RandomForestClassifier` (100 estimators) on
the full Iris dataset and saves it with `joblib` to `iris_model.pkl` in the
project root.

```powershell
cd Assignment6/iris-ml-docker
python model/train_model.py
```

Output:

```
Model trained and saved as iris_model.pkl
```

## Step 2: Flask REST API

`app.py` loads `iris_model.pkl` and exposes:

- `GET /` — health/welcome message
- `POST /predict` — accepts `{"features": [sepal_length, sepal_width, petal_length, petal_width]}` and returns `{"prediction": <class_id>}` (0 = setosa, 1 = versicolor, 2 = virginica)

## Step 3: requirements.txt

```
flask==3.0.3
scikit-learn==1.5.1
joblib==1.4.2
numpy==1.26.4
```

## Step 4: Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY iris_model.pkl .
EXPOSE 5000
CMD ["python", "app.py"]
```

| Instruction | Purpose |
|---|---|
| `FROM` | Specifies the base image (lightweight official Python 3.9 image) |
| `WORKDIR` | Sets the working directory inside the container (`/app`) |
| `COPY` | Copies files from the build context into the image |
| `RUN` | Executes commands during image build (installs dependencies) |
| `EXPOSE` | Documents the port the container listens on |
| `CMD` | Specifies the default command executed when the container starts |

`requirements.txt` is copied and installed **before** the application code so
that Docker's layer cache can reuse the dependency-install layer across
rebuilds whenever only `app.py` changes — this avoids re-downloading/
re-installing every package on every build.

## Step 5: Build the Docker image

```powershell
docker build -t iris-model-api:1.0 .
```

Verify:

```powershell
docker images
```

Actual output:

```
IMAGE                   ID             DISK USAGE   CONTENT SIZE
iris-model-api:1.0      175f978a1b1c        554MB          130MB
```

## Step 6: Run the container

```powershell
docker run -d -p 5000:5000 --name iris-container iris-model-api:1.0
```

- `-d` → detached (background) mode
- `-p 5000:5000` → maps host port 5000 to container port 5000
- `--name` → assigns a readable container name

```powershell
docker ps
```

Actual output:

```
CONTAINER ID   IMAGE                COMMAND           STATUS         PORTS
86357e23e02a   iris-model-api:1.0   "python app.py"   Up 2 seconds   0.0.0.0:5000->5000/tcp
```

## Step 7: Test the containerized API

```bash
curl http://localhost:5000/

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"
```

Actual responses:

```json
{"message":"Iris Model API is running"}
{"prediction":0}
```

A second sample was also tested:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [6.7, 3.0, 5.2, 2.3]}"
```

```json
{"prediction":2}
```

`docker logs iris-container` confirmed the Flask dev server started and
served all three requests with `200` status codes.

**Note on a warning observed in the logs:** the container logs an
`InconsistentVersionWarning` because the model was trained locally with a
newer scikit-learn (1.9.0) than the version pinned in `requirements.txt`
(1.5.1, per the assignment's pinned versions). Predictions were still
correct in this test, but in a real project this is fixed by training and
serving with the *same* scikit-learn version — e.g. by running
`train_model.py` inside the same Docker image before deploying, or pinning
the training environment to match `requirements.txt`.

## Step 8: Useful Docker management commands

| Command | Purpose |
|---|---|
| `docker ps -a` | List all containers (running + stopped) |
| `docker logs iris-container` | View container logs (for debugging) |
| `docker stop iris-container` | Stop the running container |
| `docker start iris-container` | Start a stopped container |
| `docker rm iris-container` | Remove a stopped container |
| `docker rmi iris-model-api:1.0` | Remove the image |
| `docker exec -it iris-container bash` | Open a shell inside the running container |

## Step 9 (optional, not performed): Push to Docker Hub

```bash
docker login
docker tag iris-model-api:1.0 <dockerhub-username>/iris-model-api:1.0
docker push <dockerhub-username>/iris-model-api:1.0
```

This was not performed as part of this assignment since it requires pushing
to a shared external registry under a personal Docker Hub account.

## Theory notes / Post-Lab (Viva) answers

**1. Difference between a Docker image and a Docker container?**
An image is a read-only, versioned template/blueprint containing the
application code, runtime, and dependencies. A container is a running
(or stopped) *instance* of an image — the same image can be used to start
many independent containers.

**2. Why is Docker preferred over a virtual machine for ML model deployment?**
Containers share the host OS kernel instead of running a full guest OS, so
they are lightweight (MBs vs GBs), start in seconds instead of minutes, and
are more portable across environments — while still isolating the
application's dependencies from the host.

**3. Role of the Dockerfile and its instructions?**
The Dockerfile is a text file of instructions Docker reads to build an
image layer by layer: `FROM` sets the base image, `WORKDIR` sets the
in-container working directory, `COPY` brings files from the build context
into the image, `RUN` executes build-time commands (e.g. installing
packages), `EXPOSE` documents the listening port, and `CMD` defines the
default process the container runs on startup.

**4. What is Docker layer caching, and why was `requirements.txt` copied
before the rest of the code?**
Docker builds an image as a stack of cached layers, one per instruction. If
a layer's inputs haven't changed, Docker reuses the cached layer instead of
re-running that step. By copying `requirements.txt` and running `pip
install` before copying the application code, the (slow) dependency install
layer is only invalidated when dependencies actually change — not on every
code edit — which makes rebuilds during development much faster.

**5. How would you persist logs or data generated by a container?**
Using Docker **volumes** (`docker run -v <host_path>:<container_path>` or a
named volume). A container's writable layer is ephemeral and destroyed with
the container; a volume decouples that data from the container's lifecycle
so it survives restarts/removal and can be shared between containers.

**6. How does containerization fit into a complete MLOps CI/CD pipeline?**
Typical pipeline: Train model → Serialize model (`.pkl`/`.joblib`) → Wrap in
a REST API (Flask/FastAPI) → Write Dockerfile → Build image → Run/Test
container → Push image to a registry (e.g. Docker Hub/ECR) → Deploy
(Cloud/Kubernetes). Containerizing the serving step guarantees the model
runs identically in dev, staging, and production, and the built image
becomes the CI/CD artifact that is promoted through environments.

**7. Limitations of Docker for ML workloads involving GPUs, and how are
they addressed?**
By default a container cannot access the host's GPU because it only shares
the OS kernel, not hardware drivers. This is addressed with the **NVIDIA
Container Toolkit** (formerly `nvidia-docker`), which exposes the host's
NVIDIA driver and GPU devices to the container (`docker run --gpus all ...`),
combined with a CUDA-enabled base image.

**8. How would you scale this containerized model to handle multiple
requests?**
Options include: running multiple worker processes with a production WSGI
server (e.g. `gunicorn -w 4 app:app` instead of Flask's dev server), running
multiple container replicas behind a load balancer via **Docker Compose**
(`--scale`) for a single host, or orchestrating many replicas with
auto-scaling and load balancing via **Kubernetes** for production-scale
deployments.

## Expected output (as achieved)

- Docker image `iris-model-api:1.0` was successfully built.
- A running container served the model as a REST API on port 5000.
- `POST /predict` with feature values returned the correct class prediction
  in JSON format (`0` for the setosa-like sample, `2` for the
  virginica-like sample).
- The container ran identically without requiring Python or any library to
  be installed on the host directly — only Docker.
