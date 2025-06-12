import os
import tempfile
import uuid
import logging
import base64
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check required environment variables
required_env_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {missing_vars}")
    logger.info("Please check your .env file or environment configuration")
else:
    logger.info("All required environment variables are set")

app = FastAPI(title="PDF to Markdown Converter", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create outputs directory if it doesn't exist
OUTPUTS_DIR = Path("./outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Initialize Azure OpenAI client
def get_azure_openai_client():
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set")
        
        logger.info(f"Initializing Azure OpenAI client with endpoint: {endpoint}")
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Azure OpenAI client: {str(e)}")

def convert_pdf_with_ai(pdf_file_path: str) -> str:
    """Convert PDF to markdown using Azure OpenAI GPT-4 Vision by converting pages to images"""
    try:
        logger.info(f"Converting PDF to markdown using AI: {pdf_file_path}")
        
        # Import pdf2image for PDF to image conversion
        from pdf2image import convert_from_path
        from PIL import Image
        import io
        
        # Convert PDF pages to images
        logger.info("Converting PDF pages to images")
        try:
            images = convert_from_path(pdf_file_path, dpi=200, fmt='JPEG')
            logger.info(f"Successfully converted PDF to {len(images)} page images")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to convert PDF to images: {str(e)}")
        
        # Get Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
        all_markdown_content = []
        
        # Process each page image
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num} of {len(images)}")
            
            # Convert image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Send image to Azure OpenAI
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert at converting document images to well-formatted Markdown. Extract all text content and convert it to clean, properly structured Markdown with appropriate headers, formatting, lists, links, and tables. Preserve the document structure and hierarchy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Please convert this document page (page {page_num}) to well-formatted Markdown. Preserve all formatting including headers, bold/italic text, lists, tables, and links. Extract all text content accurately."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
                
                page_markdown = response.choices[0].message.content
                if page_markdown:
                    # Add page separator for multi-page documents
                    if page_num > 1:
                        all_markdown_content.append(f"\n\n---\n\n# Page {page_num}\n\n")
                    all_markdown_content.append(page_markdown)
                    logger.info(f"Successfully processed page {page_num}")
                else:
                    logger.warning(f"No content returned for page {page_num}")
                    
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {str(e)}")
                # Continue with other pages instead of failing completely
                all_markdown_content.append(f"\n\n[Error processing page {page_num}: {str(e)}]\n\n")
        
        # Combine all page content
        combined_markdown = "".join(all_markdown_content)
        
        if not combined_markdown.strip():
            logger.error("No markdown content was generated from any page")
            raise HTTPException(status_code=500, detail="Failed to extract any content from PDF")
        
        logger.info(f"Successfully converted PDF with {len(images)} pages to markdown")
        return combined_markdown
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"AI PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF with AI: {str(e)}")

def convert_to_markdown(text: str) -> str:
    """This function is no longer needed since we use AI for direct PDF conversion"""
    # This function is kept for compatibility but not used in the new flow
    return text

# All helper functions removed - now using pure AI conversion

@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

@app.post("/api/convert")
async def convert_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Convert uploaded PDF to Markdown"""
    
    logger.info(f"Starting conversion for file: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    original_name = Path(file.filename).stem
    output_filename = f"{original_name}_{file_id}.md"
    
    temp_pdf_path = None
    
    try:
        # Save uploaded PDF temporarily
        logger.info("Saving uploaded PDF temporarily")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
            logger.info(f"PDF saved to: {temp_pdf_path}, size: {len(content)} bytes")
        
        # Convert PDF directly to markdown using AI only
        logger.info("Converting PDF to markdown using AI")
        markdown_content = convert_pdf_with_ai(temp_pdf_path)
        
        if not markdown_content:
            logger.error("Azure OpenAI returned empty content")
            raise HTTPException(status_code=500, detail="Failed to generate markdown content")
        
        # Save markdown file
        logger.info(f"Saving markdown file: {output_filename}")
        output_path = OUTPUTS_DIR / output_filename
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        
        # Clean up temporary PDF
        if temp_pdf_path:
            os.unlink(temp_pdf_path)
            logger.info("Temporary PDF file cleaned up")
        
        logger.info(f"Conversion completed successfully: {output_filename}")
        return {
            "download_url": f"/download/{output_filename}",
            "filename": output_filename,
            "message": "PDF successfully converted to Markdown"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException occurred during conversion")
        raise
    except Exception as e:
        # Clean up on error
        logger.error(f"Unexpected error during conversion: {str(e)}")
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
                logger.info("Cleaned up temporary PDF after error")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temporary file: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.get("/download/{filename}")
async def download_markdown(filename: str):
    """Download the converted Markdown file"""
    file_path = OUTPUTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/markdown"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF to Markdown Converter"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 