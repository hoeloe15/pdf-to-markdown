#!/usr/bin/env python3
"""
Create a large test PDF to test the batching functionality
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import os

def create_large_test_pdf(filename="Data/large_test_document.pdf", num_pages=25):
    """Create a large multi-page PDF for testing batching"""
    
    # Ensure Data directory exists
    os.makedirs("Data", exist_ok=True)
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title page
    story.append(Paragraph("Large Test Document for Batching", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"This document contains {num_pages} pages to test the PDF batching functionality.", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Each page contains substantial content to ensure the document is large enough to trigger batching.", styles['Normal']))
    story.append(PageBreak())
    
    # Generate multiple pages with content
    for page_num in range(1, num_pages):
        story.append(Paragraph(f"Chapter {page_num}: Content Overview", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(f"Introduction to Chapter {page_num}", styles['Heading2']))
        story.append(Spacer(1, 6))
        
        # Add substantial text content
        for para_num in range(1, 8):
            text = f"""
            This is paragraph {para_num} of chapter {page_num}. This paragraph contains substantial 
            text content to make the PDF large enough to trigger the batching functionality. 
            The batching system should automatically detect that this PDF is large and split it 
            into smaller chunks for processing. Each chunk will be processed separately by the 
            OpenAI API, and then the results will be combined into a single markdown document.
            
            This approach has several benefits:
            1. Avoids hitting OpenAI's context window limits
            2. Prevents output token truncation
            3. Improves processing speed through parallel processing
            4. Provides better error handling for large documents
            5. Gives users feedback about the processing progress
            
            The content in this paragraph is designed to be meaningful while also taking up 
            sufficient space to make the overall document large. This helps ensure that the 
            batching functionality is properly tested with realistic document sizes.
            """
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Paragraph(f"Summary of Chapter {page_num}", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Chapter {page_num} covered important concepts related to testing PDF batching functionality. The next chapter will continue with additional content.", styles['Normal']))
        
        if page_num < num_pages - 1:
            story.append(PageBreak())
    
    # Build the PDF
    doc.build(story)
    
    # Get file size
    file_size = os.path.getsize(filename)
    
    print(f"Created large test PDF: {filename}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"Number of pages: {num_pages}")
    print(f"This should trigger batching if > 10MB or > 10 pages")

if __name__ == "__main__":
    create_large_test_pdf() 