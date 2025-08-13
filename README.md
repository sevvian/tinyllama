# LLM-Powered Metadata Extractor

This   application uses a lightweight, quantized Large Language Model (`TinyLlama-1.1B-Chat`) to extract metadata from media file titles.

The application is served via a FastAPI backend and has a simple web UI for interaction. It is designed to run in a low-resource, CPU-only environment (< 2GB RAM) via Docker.

## Features

- **API Endpoint**: A `/api/extract` endpoint to process titles programmatically.
- **Web UI**: A simple interface to paste titles and see the extracted JSON output.
- **LLM-Powered**: Uses a quantized TinyLlama model for flexible and robust pattern recognition.
- **Dockerized**: Comes with a `Dockerfile` for easy deployment.
- **CI/CD**  : Includes a GitHub Actions workflow to automatically build and publish the Docker image to the GitHub Container Registry.

## How to Run Locally

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed and running.

### Steps

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd metadata-extractor-llm
    ```

2.  **Build the Docker image:**

    This command will build the image. The first build will take some time as it needs to download the LLM model (~670 MB).

    ```sh
    docker build -t metadata-extractor .
    ```

3.  **Run the Docker container:**
    ```sh
    docker run -d -p 8000:8000 --name metadata-app metadata-extractor
    ```

4.  **Access the application:**
    - **Web UI**: Open your browser to `http://localhost:8000`
    - **API Docs**: Access the auto-generated documentation at `http://localhost:8000/docs`

## GitHub Actions Workflow

The repository includes a workflow in `.github/workflows/docker-publish.yml`. This workflow will automatically:
1. Trigger on every push to the `main` branch.
2. Build the Docker image.
3. Push the image to your repository's GitHub Container Registry (GHCR).

You can find the published packages in the "Packages" section of your GitHub repository.
