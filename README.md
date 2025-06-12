# PDF to Markdown Converter

A modern web application that converts PDF documents to clean, formatted Markdown files using Azure OpenAI GPT-4.

## 🚀 Features

- **Simple Upload Interface**: Clean, modern UI with drag-and-drop support
- **AI-Powered Conversion**: Uses Azure OpenAI GPT-4 for intelligent content extraction and formatting
- **Instant Download**: Get your converted Markdown file immediately
- **Docker Ready**: Fully containerized for easy deployment
- **Cloud Deployable**: Ready for Azure Container Instances or App Service

## 📋 Requirements

- Python 3.11+
- Azure OpenAI API access
- Docker (optional, for containerized deployment)

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pdf-to-markdown
```

### 2. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### 3. Local Development

#### Option A: Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

#### Option B: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## 🎯 Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Upload a PDF file by clicking "Choose PDF File" or dragging and dropping
3. Click "Convert to Markdown"
4. Wait for processing (usually 10-30 seconds depending on PDF size)
5. Download your converted Markdown file

## 📁 Project Structure

```
pdf-to-markdown/
├── main.py                 # FastAPI backend application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── env.example           # Environment variables template
├── static/               # Frontend files
│   ├── index.html        # Main HTML page
│   ├── style.css         # CSS styles
│   └── script.js         # JavaScript functionality
├── outputs/              # Generated Markdown files (created automatically)
└── README.md            # This file
```

## 🔧 API Endpoints

- `GET /` - Serve the main web interface
- `POST /api/convert` - Convert PDF to Markdown
- `GET /download/{filename}` - Download converted Markdown files
- `GET /health` - Health check endpoint

## 🚀 Deployment

### Azure Container Instances

1. Build and push your Docker image to Azure Container Registry
2. Deploy using Azure CLI:

```bash
az container create \
    --resource-group myResourceGroup \
    --name pdf-to-markdown \
    --image myregistry.azurecr.io/pdf-to-markdown:latest \
    --environment-variables \
        AZURE_OPENAI_API_KEY=your_key \
        AZURE_OPENAI_ENDPOINT=your_endpoint \
    --ports 8000 \
    --protocol TCP
```

### Azure App Service

1. Push your code to a Git repository
2. Create an App Service and configure deployment from Git
3. Set environment variables in the App Service configuration

## 🛡️ Security Considerations

- Environment variables are used for sensitive configuration
- File uploads are validated for type and size
- Temporary files are cleaned up after processing
- CORS is configured for cross-origin requests

## 🔍 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Errors**
   - Verify your API key and endpoint are correct
   - Check that your deployment name matches your Azure OpenAI model

2. **File Upload Errors**
   - Ensure PDF files are under 50MB
   - Check that the file is a valid PDF

3. **Docker Issues**
   - Make sure Docker is running
   - Check that ports are not already in use

### Logs

Check application logs:
```bash
# Docker Compose
docker-compose logs -f

# Direct Python run
python main.py
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the application logs
- Open an issue in the repository 