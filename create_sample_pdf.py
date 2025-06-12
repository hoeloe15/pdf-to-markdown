#!/usr/bin/env python3
"""
Script to create sample PDF files for testing the PDF to Markdown converter.
Creates various types of PDFs with different content and formatting.
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas

def create_simple_text_pdf(filename="Data/simple_text.pdf"):
    """Create a simple PDF with basic text content"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Simple Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Content paragraphs
    content = [
        "This is a simple test PDF document created for testing the PDF to Markdown converter.",
        "It contains multiple paragraphs with different types of content.",
        "The converter should be able to extract this text and convert it to proper Markdown format.",
        "This paragraph tests basic text extraction capabilities."
    ]
    
    for paragraph in content:
        p = Paragraph(paragraph, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 6))
    
    doc.build(story)
    print(f"Created: {filename}")

def create_formatted_pdf(filename="Data/formatted_document.pdf"):
    """Create a PDF with various formatting (headers, bold, italic, lists)"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Formatted Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Heading 1
    h1 = Paragraph("Introduction", styles['Heading1'])
    story.append(h1)
    story.append(Spacer(1, 6))
    
    intro_text = """This document contains various formatting elements to test the 
    PDF to Markdown converter's ability to preserve formatting. The converter should 
    be able to identify headers, bold text, italic text, and lists."""
    
    p1 = Paragraph(intro_text, styles['Normal'])
    story.append(p1)
    story.append(Spacer(1, 12))
    
    # Heading 2
    h2 = Paragraph("Formatting Examples", styles['Heading2'])
    story.append(h2)
    story.append(Spacer(1, 6))
    
    # Bold and italic text
    formatted_text = """This paragraph contains <b>bold text</b> and <i>italic text</i>. 
    It also has <b><i>bold italic text</i></b> to test multiple formatting combinations."""
    
    p2 = Paragraph(formatted_text, styles['Normal'])
    story.append(p2)
    story.append(Spacer(1, 12))
    
    # Bullet points (simulated)
    h3 = Paragraph("Key Features:", styles['Heading3'])
    story.append(h3)
    story.append(Spacer(1, 6))
    
    bullet_points = [
        "• Text extraction from PDF documents",
        "• Conversion to Markdown format", 
        "• Preservation of formatting elements",
        "• Support for multiple page documents"
    ]
    
    for bullet in bullet_points:
        p = Paragraph(bullet, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 3))
    
    doc.build(story)
    print(f"Created: {filename}")

def create_table_pdf(filename="Data/table_document.pdf"):
    """Create a PDF with a table to test table extraction"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Table Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Introduction
    intro = Paragraph("This document contains a table to test table extraction capabilities.", styles['Normal'])
    story.append(intro)
    story.append(Spacer(1, 12))
    
    # Table data
    data = [
        ['Name', 'Age', 'City', 'Occupation'],
        ['John Doe', '30', 'New York', 'Engineer'],
        ['Jane Smith', '25', 'Los Angeles', 'Designer'],
        ['Bob Johnson', '35', 'Chicago', 'Manager'],
        ['Alice Brown', '28', 'Houston', 'Developer']
    ]
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Conclusion
    conclusion = Paragraph("The table above should be converted to Markdown table format.", styles['Normal'])
    story.append(conclusion)
    
    doc.build(story)
    print(f"Created: {filename}")

def create_multipage_pdf(filename="Data/multipage_document.pdf"):
    """Create a multi-page PDF to test page handling"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Page 1
    title = Paragraph("Multi-Page Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    page1_content = """This is the first page of a multi-page document. The PDF to Markdown 
    converter should be able to handle multiple pages and combine them into a single 
    Markdown document with appropriate page separators."""
    
    p1 = Paragraph(page1_content, styles['Normal'])
    story.append(p1)
    
    # Add page break
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    
    # Page 2
    h1 = Paragraph("Second Page", styles['Heading1'])
    story.append(h1)
    story.append(Spacer(1, 12))
    
    page2_content = """This is the second page of the document. It contains different 
    content to verify that the converter properly processes multiple pages and maintains 
    the content structure across page boundaries."""
    
    p2 = Paragraph(page2_content, styles['Normal'])
    story.append(p2)
    story.append(Spacer(1, 12))
    
    # Add some numbered items
    h2 = Paragraph("Numbered List:", styles['Heading2'])
    story.append(h2)
    story.append(Spacer(1, 6))
    
    numbered_items = [
        "1. First item in the list",
        "2. Second item in the list", 
        "3. Third item in the list",
        "4. Fourth item in the list"
    ]
    
    for item in numbered_items:
        p = Paragraph(item, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 3))
    
    # Page 3
    story.append(PageBreak())
    
    h1_page3 = Paragraph("Third Page - Conclusion", styles['Heading1'])
    story.append(h1_page3)
    story.append(Spacer(1, 12))
    
    conclusion = """This is the final page of the multi-page test document. 
    The converter should successfully process all three pages and create a 
    coherent Markdown document with proper page organization."""
    
    p3 = Paragraph(conclusion, styles['Normal'])
    story.append(p3)
    
    doc.build(story)
    print(f"Created: {filename}")

def create_minimal_pdf(filename="Data/minimal.pdf"):
    """Create a very simple PDF with minimal content"""
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Add simple text
    c.drawString(100, 750, "Minimal Test PDF")
    c.drawString(100, 730, "This is a very simple PDF with just a few lines of text.")
    c.drawString(100, 710, "Perfect for quick testing.")
    
    c.save()
    print(f"Created: {filename}")

def main():
    """Create all sample PDFs"""
    # Create Data directory if it doesn't exist
    os.makedirs("Data", exist_ok=True)
    
    print("Creating sample PDF files for testing...")
    print("=" * 50)
    
    try:
        # Check if reportlab is available
        import reportlab
        
        # Create different types of PDFs
        create_minimal_pdf()
        create_simple_text_pdf()
        create_formatted_pdf()
        create_table_pdf()
        create_multipage_pdf()
        
        print("=" * 50)
        print("All sample PDFs created successfully!")
        print("\nCreated files in Data/ folder:")
        for filename in ["Data/minimal.pdf", "Data/simple_text.pdf", "Data/formatted_document.pdf", 
                        "Data/table_document.pdf", "Data/multipage_document.pdf"]:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                basename = os.path.basename(filename)
                print(f"  {basename} ({size} bytes)")
        
        print("\nYou can now test these PDFs with your converter:")
        print("  curl -X POST -F 'file=@Data/minimal.pdf' http://localhost:8000/api/convert")
        print("  curl -X POST -F 'file=@Data/minimal.pdf' -F 'backend=azure' http://localhost:8000/api/convert")
        
    except ImportError:
        print("Error: reportlab is not installed.")
        print("Install it with: pip install reportlab")
        print("\nCreating minimal PDF with basic canvas method...")
        create_minimal_pdf()

if __name__ == "__main__":
    main()
