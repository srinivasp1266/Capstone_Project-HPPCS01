"""
Basic tests for CV Builder application.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all main modules can be imported."""
    try:
        from src.document_extractor import DocumentExtractor
        from src.llm_client import OllamaClient, LLMService
        from src.resume_parser import ResumeParser, ResumeData
        from src.job_parser import JobDescriptionParser
        from src.cv_generator import CVTailoringEngine
        from src.document_generator import CVDocumentGenerator
        from src.config import settings
        
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_document_extractor():
    """Test document extractor initialization."""
    from src.document_extractor import DocumentExtractor
    
    extractor = DocumentExtractor()
    assert extractor is not None
    assert hasattr(extractor, 'extract_text')
    assert hasattr(extractor, 'clean_text')

def test_resume_data_structure():
    """Test resume data structure creation."""
    from src.resume_parser import ResumeData, PersonalInfo, Experience
    
    resume_data = ResumeData()
    assert resume_data is not None
    assert hasattr(resume_data, 'personal_info')
    assert hasattr(resume_data, 'experience')
    assert hasattr(resume_data, 'skills')
    
    # Test default initialization
    assert isinstance(resume_data.experience, list)
    assert isinstance(resume_data.skills, dict)

def test_job_requirements_structure():
    """Test job requirements structure."""
    from src.job_parser import JobRequirements
    
    job_req = JobRequirements()
    assert job_req is not None
    assert hasattr(job_req, 'title')
    assert hasattr(job_req, 'required_skills')
    assert hasattr(job_req, 'keywords')
    
    # Test default initialization
    assert isinstance(job_req.required_skills, list)
    assert isinstance(job_req.keywords, list)

def test_config_settings():
    """Test configuration settings."""
    from src.config import settings
    
    assert settings is not None
    assert hasattr(settings, 'ollama_host')
    assert hasattr(settings, 'ollama_model')
    assert settings.ollama_host is not None
    assert settings.ollama_model is not None

def test_cv_tailoring_engine():
    """Test CV tailoring engine initialization."""
    from src.cv_generator import CVTailoringEngine
    
    # Test that it can be initialized (without LLM service for now)
    engine = CVTailoringEngine(None)
    assert engine is not None
    assert hasattr(engine, 'tailor_cv')

def test_document_generator():
    """Test document generator initialization."""
    from src.document_generator import CVDocumentGenerator
    
    generator = CVDocumentGenerator()
    assert generator is not None
    assert hasattr(generator, 'generate_html')
    assert hasattr(generator, 'generate_markdown')

def test_text_cleaning():
    """Test text cleaning functionality."""
    from src.document_extractor import DocumentExtractor
    
    extractor = DocumentExtractor()
    
    # Test cleaning excessive whitespace
    dirty_text = "This   has     excessive    whitespace"
    clean_text = extractor.clean_text(dirty_text)
    assert "   " not in clean_text
    assert clean_text == "This has excessive whitespace"
    
    # Test cleaning excessive newlines
    dirty_text = "Line 1\n\n\n\nLine 2"
    clean_text = extractor.clean_text(dirty_text)
    assert "\n\n\n" not in clean_text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])