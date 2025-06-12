import os
import tempfile
import uuid
import logging
import base64
import re
import asyncio
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI, OpenAI
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

# Initialize regular OpenAI client for direct PDF processing
def get_openai_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        logger.info("Initializing OpenAI client for direct PDF processing")
        
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {str(e)}")

def extract_markdown_from_response(response_text: str) -> str:
    """Extract markdown content from LLM response, handling various formats"""
    if not response_text:
        return ""
    
    logger.info("Extracting markdown content from LLM response")
    
    # First, try to extract from ```markdown code blocks
    markdown_pattern = r'```(?:markdown|md)?\s*\n(.*?)\n```'
    match = re.search(markdown_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        logger.info("Found markdown in code block, extracting...")
        return match.group(1).strip()
    
    # If no code block found, clean the response by removing conversational wrapper
    logger.info("No code block found, cleaning conversational wrapper...")
    
    # Remove common conversational elements
    cleaned = response_text
    
    # Remove intro phrases
    intro_patterns = [
        r'^.*?(?:certainly!?|here is|here\'s|the content|converted to markdown|below is|i\'ll convert).*?:?\s*',
        r'^.*?here you go.*?:?\s*',
        r'^.*?markdown.*?:?\s*'
    ]
    
    for pattern in intro_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove ending phrases
    ending_patterns = [
        r'\n\n.*?(?:preserves? the structure|let me know|feel free|if you need|any questions).*$',
        r'\n\n.*?(?:if there are any|hope this helps|anything else).*$'
    ]
    
    for pattern in ending_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    logger.info("Cleaned conversational wrapper from response")
    return cleaned

def check_pdf_size_and_pages(pdf_file_path: str) -> tuple[int, int]:
    """Check PDF file size and page count to determine if batching is needed"""
    try:
        from pdf2image import convert_from_path
        
        # Get file size
        file_size = os.path.getsize(pdf_file_path)
        
        # Get page count (quick check without converting)
        try:
            # Quick page count check
            images = convert_from_path(pdf_file_path, dpi=72, first_page=1, last_page=1)
            # Get total pages by checking with a library
            import PyPDF2
            with open(pdf_file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
        except:
            # Fallback: estimate based on file size
            page_count = max(1, file_size // 50000)  # Rough estimate: 50KB per page
        
        logger.info(f"PDF analysis: {file_size} bytes, estimated {page_count} pages")
        return file_size, page_count
        
    except Exception as e:
        logger.warning(f"Could not analyze PDF size/pages: {e}")
        return os.path.getsize(pdf_file_path), 1

def convert_pdf_with_openai_batched(pdf_file_path: str, max_pages_per_batch: int = 10) -> str:
    """Convert PDF using OpenAI with intelligent batching for large documents"""
    try:
        file_size, total_pages = check_pdf_size_and_pages(pdf_file_path)
        
        # Define thresholds for batching
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        MAX_PAGES_PER_BATCH = max_pages_per_batch
        
        # Check if batching is needed
        needs_batching = file_size > MAX_FILE_SIZE or total_pages > MAX_PAGES_PER_BATCH
        
        if not needs_batching:
            logger.info("PDF is small enough for single processing")
            return convert_pdf_with_openai_single(pdf_file_path)
        
        logger.info(f"Large PDF detected ({file_size} bytes, {total_pages} pages) - using batched processing")
        
        # Import required libraries for PDF splitting
        from pdf2image import convert_from_path
        import tempfile
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import PyPDF2
        
        # Split PDF into batches
        all_markdown_content = []
        
        with open(pdf_file_path, 'rb') as input_file:
            reader = PyPDF2.PdfReader(input_file)
            total_pages = len(reader.pages)
            
            # Calculate number of batches
            num_batches = (total_pages + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH
            logger.info(f"Splitting PDF into {num_batches} batches of ~{MAX_PAGES_PER_BATCH} pages each")
            
            for batch_num in range(num_batches):
                start_page = batch_num * MAX_PAGES_PER_BATCH
                end_page = min((batch_num + 1) * MAX_PAGES_PER_BATCH, total_pages)
                
                logger.info(f"Processing batch {batch_num + 1}/{num_batches} (pages {start_page + 1}-{end_page})")
                
                # Create temporary PDF for this batch
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as batch_pdf:
                    writer = PyPDF2.PdfWriter()
                    
                    # Add pages to batch
                    for page_num in range(start_page, end_page):
                        writer.add_page(reader.pages[page_num])
                    
                    writer.write(batch_pdf)
                    batch_pdf_path = batch_pdf.name
                
                try:
                    # Process this batch
                    batch_markdown = convert_pdf_with_openai_single(batch_pdf_path)
                    
                    # Add batch separator and content
                    if batch_num > 0:
                        all_markdown_content.append(f"\n\n---\n\n# Batch {batch_num + 1} (Pages {start_page + 1}-{end_page})\n\n")
                    
                    all_markdown_content.append(batch_markdown)
                    logger.info(f"Successfully processed batch {batch_num + 1}")
                    
                except Exception as batch_error:
                    logger.error(f"Failed to process batch {batch_num + 1}: {batch_error}")
                    all_markdown_content.append(f"\n\n[Error processing pages {start_page + 1}-{end_page}: {batch_error}]\n\n")
                
                finally:
                    # Clean up temporary batch file
                    try:
                        os.unlink(batch_pdf_path)
                    except:
                        pass
        
        # Combine all batch results
        combined_markdown = "".join(all_markdown_content)
        
        if not combined_markdown.strip():
            raise HTTPException(status_code=500, detail="Failed to convert any part of the PDF")
        
        logger.info(f"Successfully converted large PDF using {num_batches} batches")
        
        # Add batching info to the beginning of the document
        batching_info = f"<!-- Converted using {num_batches} batches due to document size ({file_size:,} bytes, {total_pages} pages) -->\n\n"
        return batching_info + combined_markdown
        
    except Exception as e:
        logger.error(f"Batched PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF in batches: {str(e)}")

def convert_pdf_with_openai_single(pdf_file_path: str) -> str:
    """Convert single PDF file using OpenAI API (original function)"""
    try:
        logger.info(f"Converting PDF directly with OpenAI: {pdf_file_path}")
        
        # Get OpenAI client
        client = get_openai_client()
        
        logger.info("Uploading PDF file to OpenAI...")
        
        # Upload PDF file to OpenAI
        with open(pdf_file_path, 'rb') as pdf_file:
            file_upload = client.files.create(
                file=pdf_file,
                purpose="user_data"
            )
        
        file_id = file_upload.id
        logger.info(f"PDF uploaded successfully with file ID: {file_id}")
        
        # Send file to OpenAI for processing
        logger.info("Processing PDF with OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o which supports file inputs
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at converting PDF documents to well-formatted Markdown. Extract all text content and convert it to clean, properly structured Markdown with appropriate headers, formatting, lists, links, and tables. Preserve the document structure and hierarchy."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_id": file_id
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please convert this PDF document to well-formatted Markdown. Preserve all formatting including headers, bold/italic text, lists, tables, and links. Maintain the document structure and hierarchy."
                        }
                    ]
                }
            ],
            max_tokens=8000,  # Increased for better coverage
            temperature=0.1
        )
        
        raw_content = response.choices[0].message.content
        
        # Clean up: delete the uploaded file
        try:
            client.files.delete(file_id)
            logger.info(f"Cleaned up uploaded file: {file_id}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup uploaded file {file_id}: {cleanup_error}")
        
        if not raw_content:
            logger.error("OpenAI returned empty content")
            raise HTTPException(status_code=500, detail="Failed to generate markdown content")
        
        # Extract clean markdown from the response
        markdown_content = extract_markdown_from_response(raw_content)
        
        if not markdown_content:
            logger.error("Failed to extract markdown content from response")
            raise HTTPException(status_code=500, detail="Failed to parse markdown content from response")
        
        logger.info("Successfully received and parsed markdown from OpenAI")
        return markdown_content
        
    except Exception as e:
        logger.error(f"OpenAI direct PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF with OpenAI: {str(e)}")

async def convert_pdf_with_openai_single_async(pdf_file_path: str, batch_num: int, page_range: str) -> tuple[int, str]:
    """Async version of convert_pdf_with_openai_single for parallel batch processing"""
    try:
        logger.info(f"Starting async conversion for batch {batch_num} ({page_range}): {pdf_file_path}")
        
        # Run the synchronous OpenAI call in a thread pool to make it non-blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, convert_pdf_with_openai_single, pdf_file_path)
        
        logger.info(f"Completed async conversion for batch {batch_num}")
        return batch_num, result
        
    except Exception as e:
        logger.error(f"Async batch {batch_num} conversion failed: {str(e)}")
        # Return error message for this batch
        return batch_num, f"\n\n[Error processing batch {batch_num} ({page_range}): {str(e)}]\n\n"

def convert_pdf_with_openai_direct(pdf_file_path: str) -> str:
    """Convert PDF to markdown using OpenAI API with intelligent batching for large files"""
    return convert_pdf_with_openai_batched(pdf_file_path)

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
async def convert_pdf(
    file: UploadFile = File(...), 
    backend: str = Form("openai")  # Default to OpenAI, options: "openai" or "azure"
) -> Dict[str, Any]:
    """Convert uploaded PDF to Markdown using specified backend"""
    
    logger.info(f"Starting conversion for file: {file.filename} using backend: {backend}")
    
    # Validate backend parameter
    if backend not in ["openai", "azure"]:
        raise HTTPException(status_code=400, detail="Backend must be either 'openai' or 'azure'")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename with backend identifier
    file_id = str(uuid.uuid4())
    original_name = Path(file.filename).stem
    output_filename = f"{original_name}_{backend}_{file_id}.md"
    
    temp_pdf_path = None
    
    try:
        # Save uploaded PDF temporarily
        logger.info("Saving uploaded PDF temporarily")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
            logger.info(f"PDF saved to: {temp_pdf_path}, size: {len(content)} bytes")
        
        # Convert PDF to markdown using selected backend
        if backend == "openai":
            logger.info("Converting PDF to markdown using OpenAI direct API")
            markdown_content = convert_pdf_with_openai_direct(temp_pdf_path)
        else:  # backend == "azure"
            logger.info("Converting PDF to markdown using Azure OpenAI (PDF→Images→AI)")
            markdown_content = convert_pdf_with_ai(temp_pdf_path)
        
        if not markdown_content:
            logger.error(f"{backend.upper()} returned empty content")
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
        
        # Check if batching was used (look for batch comments in the markdown)
        batching_used = "<!-- Converted using" in markdown_content and "batches" in markdown_content
        batching_info = ""
        if batching_used:
            # Extract batching info from the comment
            import re
            match = re.search(r'<!-- Converted using (\d+) batches due to document size \(([^)]+)\) -->', markdown_content)
            if match:
                num_batches, file_size = match.groups()
                batching_info = f" (processed in {num_batches} batches due to size: {file_size})"
        
        return {
            "download_url": f"/download/{output_filename}",
            "filename": output_filename,
            "message": f"PDF successfully converted to Markdown using {backend.upper()} API{batching_info}",
            "backend": backend,
            "batched": batching_used
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

@app.post("/api/convert-openai")
async def convert_pdf_openai_direct(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Convert uploaded PDF to Markdown using OpenAI direct PDF support"""
    
    logger.info(f"Starting OpenAI direct conversion for file: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename with openai prefix
    file_id = str(uuid.uuid4())
    original_name = Path(file.filename).stem
    output_filename = f"{original_name}_openai_{file_id}.md"
    
    temp_pdf_path = None
    
    try:
        # Save uploaded PDF temporarily
        logger.info("Saving uploaded PDF temporarily")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
            logger.info(f"PDF saved to: {temp_pdf_path}, size: {len(content)} bytes")
        
        # Convert PDF directly using OpenAI
        logger.info("Converting PDF to markdown using OpenAI direct API")
        markdown_content = convert_pdf_with_openai_direct(temp_pdf_path)
        
        if not markdown_content:
            logger.error("OpenAI returned empty content")
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
        
        logger.info(f"OpenAI direct conversion completed successfully: {output_filename}")
        return {
            "download_url": f"/download/{output_filename}",
            "filename": output_filename,
            "message": "PDF successfully converted to Markdown using OpenAI direct API",
            "method": "openai-direct"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException occurred during OpenAI conversion")
        raise
    except Exception as e:
        # Clean up on error
        logger.error(f"Unexpected error during OpenAI conversion: {str(e)}")
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
                logger.info("Cleaned up temporary PDF after error")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temporary file: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"OpenAI conversion failed: {str(e)}")

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

@app.get("/api/backends")
async def get_backends():
    """Get available backends and their configuration status"""
    backends = {
        "openai": {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "description": "OpenAI direct PDF processing",
            "method": "Direct PDF to AI conversion"
        },
        "azure": {
            "available": bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")),
            "description": "Azure OpenAI with image conversion",
            "method": "PDF to Images to AI conversion"
        }
    }
    
    default_backend = "openai"
    
    return {
        "backends": backends,
        "default": default_backend,
        "recommendation": "Use 'openai' for direct PDF processing or 'azure' for image-based processing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF to Markdown Converter"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 