# Watermark Remover

A web-based application for removing watermarks from videos using AI-powered inpainting technology.

## Features

- Web interface for easy video processing
- AI-powered watermark detection and removal
- Support for various video formats
- Docker support for containerized deployment

## Prerequisites

- Python 3.7 or higher
- Docker and Docker Compose (for containerized deployment)

## Installation

### Bare Metal Installation

1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python numpy torch torchvision simple-lama-inpainting pillow
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

### Docker Installation

#### macOS Setup

1. **Install Docker**
   ```bash
   brew install docker docker-compose
   brew install docker-credential-helper
   ```

2. **Build the Docker image**
   ```bash
   docker build -f Dockerfile -t watermarkremover .
   ```

3. **Run the container**
   ```bash
   docker run -p 0.0.0.0:5000:5000 watermarkremover
   ```

The application will be available at `http://localhost:5000`

> **Note:** Docker deployment may have slower performance compared to bare metal installation.

## Usage

1. Navigate to `http://localhost:5000` in your web browser
2. Upload your video file
3. Select the watermark region
4. Process and download the cleaned video

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Support

For issues and questions, please open an issue on the GitHub repository.


