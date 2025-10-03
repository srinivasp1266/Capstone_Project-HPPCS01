"""
Document extraction module for CV Builder.
Handles PDF and Word document parsing and text extraction.
"""

import io
import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Handles extraction of text from various document formats."""
    
    def __init__(self):
        """Initialize the document extractor."""
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_word(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def extract_from_bytes(self, file_bytes: bytes, file_extension: str) -> str:
        """
        Extract text from file bytes.
        
        Args:
            file_bytes: File content as bytes
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            Extracted text content
        """
        file_extension = file_extension.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf_bytes(file_bytes)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_word_bytes(file_bytes)
            elif file_extension == '.txt':
                return self._extract_from_txt_bytes(file_bytes)
        except Exception as e:
            logger.error(f"Error extracting text from bytes: {str(e)}")
            raise
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file with robust error handling."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf = pdfplumber.open(file)
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                pdf.close()
                
                if not text.strip():
                    self.logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return "Unable to extract text from this PDF document."
                
                return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting PDF text from {pdf_path}: {str(e)}")
            return f"Error reading PDF file: {str(e)}"
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        return self.extract_pdf_text(str(file_path))
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {str(e)}")
            raise
    
    def _extract_from_txt_bytes(self, file_bytes: bytes) -> str:
        """Extract text from TXT bytes."""
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                logger.error(f"Error decoding TXT bytes: {str(e)}")
                raise
    
    def _extract_from_pdf_bytes(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")
        
        text_content = []
        
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
        except Exception as e:
            logger.error(f"Error reading PDF from bytes: {str(e)}")
            raise
        
        return '\n'.join(text_content)
    
    def _extract_from_word(self, file_path: Path) -> str:
        """Extract text from Word document."""
        # Try docx2txt first (simpler and more reliable)
        if docx2txt is not None:
            try:
                return docx2txt.process(str(file_path))
            except Exception as e:
                logger.warning(f"docx2txt failed for {file_path}: {str(e)}, trying python-docx")
        
        # Fallback to python-docx
        if Document is not None:
            try:
                doc = Document(file_path)
                text_content = []
                
                for paragraph in doc.paragraphs:
                    text_content.append(paragraph.text)
                
                # Extract text from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = []
                        for cell in row.cells:
                            row_text.append(cell.text)
                        text_content.append(' | '.join(row_text))
                
                return '\n'.join(text_content)
            except Exception as e:
                logger.error(f"python-docx failed for {file_path}: {str(e)}")
                raise
        
        raise ImportError("Neither docx2txt nor python-docx is available for Word document extraction")
    
    def _extract_from_word_bytes(self, file_bytes: bytes) -> str:
        """Extract text from Word document bytes."""
        if Document is None:
            raise ImportError("python-docx is required for Word extraction from bytes. Install with: pip install python-docx")
        
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_content = []
            
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text)
                    text_content.append(' | '.join(row_text))
            
            return '\n'.join(text_content)
        except Exception as e:
            logger.error(f"Error reading Word document from bytes: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix common OCR/extraction issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Fix broken words across lines
        
        return text
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing metadata
        """
        file_path = Path(file_path)
        metadata = {
            'filename': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
        }
        
        try:
            if file_path.suffix.lower() == '.pdf' and pdfplumber is not None:
                with pdfplumber.open(file_path) as pdf:
                    metadata.update({
                        'page_count': len(pdf.pages),
                        'pdf_metadata': pdf.metadata or {}
                    })
            elif file_path.suffix.lower() in ['.docx', '.doc'] and Document is not None:
                doc = Document(file_path)
                metadata.update({
                    'paragraph_count': len(doc.paragraphs),
                    'table_count': len(doc.tables)
                })
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {str(e)}")
        
        return metadata


# Convenience function
def extract_resume_text(file_path: str) -> str:
    """
    Convenience function to extract and clean text from a resume file.
    
    Args:
        file_path: Path to the resume file
        
    Returns:
        Cleaned extracted text
    """
    extractor = DocumentExtractor()
    raw_text = extractor.extract_text(file_path)
    return extractor.clean_text(raw_text)