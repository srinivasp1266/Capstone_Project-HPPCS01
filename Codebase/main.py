#!/usr/bin/env python3
"""
HAAI++ Capstone Project - CV Creation using Multi-LLM Architecture
Project ID: HPPCS[01]
Author: Srinivas P
Date: October 2025

This is the main entry point for the CV Builder application that uses
multiple Large Language Models for enhanced CV creation and tailoring.

MULTI-LLM ARCHITECTURE:
- Gemma 2B: Content Generation Specialist (fast, creative)
- Llama2 7B: Data Analysis Specialist (powerful, analytical)

Usage:
    python main.py [--host HOST] [--timeout TIMEOUT]
    
Arguments:
    --host: Ollama server host (default: http://localhost:11434)
    --timeout: Request timeout in seconds (default: 120)
"""

import streamlit as st
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import tempfile
import io

# All files are now in the same directory per HAAI++ requirements
try:
    from document_extractor import DocumentExtractor
    from llm_client import OllamaClient, LLMService
    from multi_llm_service import MultiLLMService  # New multi-LLM service
    from resume_parser import ResumeParser, ResumeData
    from job_parser import JobDescriptionParser
    from cv_generator import CVTailoringEngine
    from document_generator import CVDocumentGenerator
    from config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all dependencies are installed and all files are in the same directory.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="HAAI++ Multi-LLM CV Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3498db, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    
    .info-box {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d5f4e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fef9e7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = None
    if 'tailored_cv' not in st.session_state:
        st.session_state.tailored_cv = None
    if 'generated_document' not in st.session_state:
        st.session_state.generated_document = None
    if 'ollama_host' not in st.session_state:
        st.session_state.ollama_host = settings.ollama_host

# Initialize session state immediately when module is loaded
init_session_state()


def initialize_services():
    """Initialize Multi-LLM and other services."""
    if st.session_state.llm_client is None:
        try:
            with st.spinner("Initializing AI services with dual-model architecture..."):
                # Initialize Multi-LLM service for HAAI++ capstone requirements
                st.session_state.llm_client = MultiLLMService(
                    host=settings.ollama_host,
                    timeout=settings.ollama_timeout
                )
            st.success("✅ Multi-LLM AI services initialized successfully!")
            st.info("🎓 **HAAI++ Capstone**: Using Gemma 2B for content generation and Llama2 7B for data analysis")
        except Exception as e:
            st.error(f"❌ Failed to initialize AI services: {str(e)}")
            st.error("Please make sure Ollama is running and both models (gemma:2b, llama2:7b) are available.")
            return False
    return True


def handle_resume_upload():
    """Handle resume file upload and processing."""
    st.markdown('<div class="section-header">📁 Upload Your Resume</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx', 'txt'],
        help="Upload your existing resume in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from document
                extractor = DocumentExtractor()
                resume_text = extractor.extract_text(tmp_file_path)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                if not resume_text.strip():
                    st.error("❌ Could not extract text from the uploaded file. Please try a different format.")
                    return
                
                # Parse resume using Multi-LLM service
                parser = ResumeParser(st.session_state.llm_client)
                
                # Debug: Show raw text length
                st.info(f"📄 Extracted {len(resume_text)} characters from resume")
                
                # Parse the resume with enhanced error handling
                try:
                    with st.spinner("🤖 Processing resume with AI models..."):
                        st.session_state.resume_data = parser.parse_resume(resume_text)
                    
                    # Check what was parsed
                    if st.session_state.resume_data:
                        logger.info(f"Resume parsing completed successfully")
                        st.success("✅ Resume processed successfully!")
                        
                        # Show parsing method used
                        if hasattr(st.session_state.resume_data, '_parsing_method'):
                            method = getattr(st.session_state.resume_data, '_parsing_method', 'unknown')
                            if method == 'basic':
                                st.info("ℹ️ Used basic text extraction (LLM unavailable)")
                            elif method == 'fallback':
                                st.warning("⚠️ Used fallback parsing due to LLM timeout")
                            else:
                                st.success("🎯 Used advanced AI parsing")
                    else:
                        st.warning("⚠️ Resume processed but limited data extracted")
                        # Create a minimal resume data structure
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo()
                        
                except ConnectionError as conn_error:
                    st.error("🔌 **Connection Error**: Cannot reach AI services")
                    st.info("💡 **Tip**: Make sure Ollama is running with: `ollama serve`")
                    st.info("📋 **Fallback**: Using basic text extraction instead")
                    
                    # Use basic extraction as fallback
                    from llm_client import LLMService
                    basic_service = LLMService(None)  # No client = basic extraction only
                    st.session_state.resume_data = basic_service._extract_resume_data_basic(resume_text)
                    
                except TimeoutError as timeout_error:
                    st.error("⏰ **Timeout Error**: AI processing took too long")
                    st.info("💡 **Tip**: Try with a shorter resume or restart Ollama")
                    st.info("📋 **Fallback**: Using basic text extraction instead")
                    
                    # Use basic extraction as fallback
                    from llm_client import LLMService
                    basic_service = LLMService(None)
                    st.session_state.resume_data = basic_service._extract_resume_data_basic(resume_text)
                    
                except Exception as parse_error:
                    st.error(f"❌ **Parsing Error**: {str(parse_error)}")
                    logger.error(f"Resume parsing error: {str(parse_error)}")
                    
                    # Try basic extraction as last resort
                    try:
                        st.info("📋 **Attempting basic extraction...**")
                        from llm_client import LLMService
                        basic_service = LLMService(None)
                        basic_data = basic_service._extract_resume_data_basic(resume_text)
                        
                        # Convert to proper format
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo(
                            name=basic_data.get('name', ''),
                            email=basic_data.get('email', ''),
                            phone=basic_data.get('phone', ''),
                            address=basic_data.get('address', ''),
                            linkedin=basic_data.get('linkedin', ''),
                            website=basic_data.get('website', ''),
                        )
                        st.success("✅ Basic extraction completed!")
                        
                    except Exception as fallback_error:
                        st.error(f"❌ All extraction methods failed: {str(fallback_error)}")
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo()
                
                # Display extracted information
                with st.expander("📋 Extracted Resume Information", expanded=True):
                    if st.session_state.resume_data and st.session_state.resume_data.personal_info:
                        st.subheader("👤 Personal Information")
                        info = st.session_state.resume_data.personal_info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Name:** {info.name or 'Not found'}")
                            st.write(f"**Email:** {info.email or 'Not found'}")
                            st.write(f"**Phone:** {info.phone or 'Not found'}")
                        with col2:
                            st.write(f"**Location:** {info.address or 'Not found'}")
                            st.write(f"**LinkedIn:** {info.linkedin or 'Not found'}")
                            st.write(f"**Website:** {info.website or 'Not found'}")
                    
                    if st.session_state.resume_data and st.session_state.resume_data.experience:
                        st.subheader("💼 Work Experience")
                        for i, exp in enumerate(st.session_state.resume_data.experience):
                            st.write(f"**{exp.position or 'Position'}** at {exp.company or 'Company'} ({exp.start_date or 'Start'} - {exp.end_date or 'End'})")
                    else:
                        st.subheader("💼 Work Experience")
                        st.write("No work experience found in the resume")
                    
                    if st.session_state.resume_data and st.session_state.resume_data.skills:
                        st.subheader("🛠️ Skills")
                        skills = st.session_state.resume_data.skills
                        if isinstance(skills, dict):
                            for category, skill_list in skills.items():
                                if skill_list:
                                    st.write(f"**{category.title()}:** {', '.join(skill_list)}")
                        elif isinstance(skills, list):
                            st.write(f"**Skills:** {', '.join(skills)}")
                        else:
                            st.write(f"**Skills:** {skills}")
                    else:
                        st.subheader("🛠️ Skills")
                        st.write("No skills found in the resume")
                
                # Add manual continue button
                st.markdown("---")
                if st.button("➡️ Continue to Job Description", type="primary", key="continue_to_job"):
                    st.rerun()
                            
            except Exception as e:
                st.error(f"❌ Error processing resume: {str(e)}")
                logger.error(f"Resume processing error: {str(e)}")


def handle_job_description():
    """Handle job description input and processing."""
    st.markdown('<div class="section-header">📋 Job Description</div>', unsafe_allow_html=True)
    
    job_text = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Copy and paste the job description, requirements, and qualifications here..."
    )
    
    if st.button("🔍 Analyze Job Requirements", disabled=not job_text.strip()):
        with st.spinner("Analyzing job requirements..."):
            try:
                # Parse job description using Multi-LLM service
                parser = JobDescriptionParser(st.session_state.llm_client)
                st.session_state.job_requirements = parser.parse_job_description(job_text)
                
                st.success("✅ Job requirements analyzed successfully!")
                
                # Display extracted job information
                with st.expander("📊 Extracted Job Requirements", expanded=True):
                    reqs = st.session_state.job_requirements
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📌 Basic Information")
                        st.write(f"**Title:** {reqs.title}")
                        st.write(f"**Company:** {reqs.company}")
                        st.write(f"**Experience Level:** {reqs.experience_level}")
                        
                        if reqs.required_skills:
                            st.subheader("🎯 Required Skills")
                            for skill in reqs.required_skills[:10]:  # Show top 10
                                st.write(f"• {skill}")
                    
                    with col2:
                        if reqs.responsibilities:
                            st.subheader("📝 Key Responsibilities")
                            for resp in reqs.responsibilities[:5]:  # Show top 5
                                st.write(f"• {resp}")
                        
                        if reqs.keywords:
                            st.subheader("🔑 Important Keywords")
                            keywords_text = ", ".join(reqs.keywords[:15])  # Show top 15
                            st.write(keywords_text)
                
                # Add manual continue button after successful analysis
                st.markdown("---")
                if st.button("➡️ Continue to AI Tailoring", type="primary", key="continue_to_ai"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error analyzing job description: {str(e)}")
                logger.error(f"Job analysis error: {str(e)}")
                
                # Even on error, allow manual continuation with basic job data
                st.warning("⚠️ Job analysis failed, but you can still proceed with basic information")
                if st.button("➡️ Continue Anyway", type="secondary", key="continue_anyway"):
                    # Create minimal job requirements
                    from job_parser import JobRequirements
                    st.session_state.job_requirements = JobRequirements(
                        title="Software Developer",
                        company="Unknown Company",
                        description=job_text,
                        required_skills=[],
                        responsibilities=[],
                        keywords=[]
                    )
                    st.rerun()


def handle_cv_tailoring():
    """Handle CV tailoring process."""
    st.markdown('<div class="section-header">🤖 AI-Powered CV Tailoring</div>', unsafe_allow_html=True)
    
    if not st.session_state.resume_data:
        st.warning("⚠️ Please upload and process your resume first.")
        return
    
    if not st.session_state.job_requirements:
        st.warning("⚠️ Please analyze the job description first.")
        return
    
    # Tailoring options
    st.subheader("⚙️ Tailoring Options")
    col1, col2 = st.columns(2)
    with col1:
        optimize_summary = st.checkbox("📝 Optimize professional summary", value=True)
        highlight_skills = st.checkbox("⭐ Highlight relevant skills", value=True)
    with col2:
        optimize_projects = st.checkbox("🚀 Optimize project descriptions", value=True)
        add_achievements = st.checkbox("🏆 Enhance achievement statements", value=True)
    
    if st.button("🤖 Start AI Tailoring", type="primary", disabled=st.session_state.tailored_cv is not None):
        with st.spinner("AI is tailoring your CV... This may take a few minutes."):
            try:
                # Create tailoring engine with Multi-LLM service
                engine = CVTailoringEngine(st.session_state.llm_client)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Analyzing compatibility...")
                progress_bar.progress(20)
                
                # Tailor the CV
                tailored_cv = engine.tailor_cv(
                    st.session_state.resume_data,
                    st.session_state.job_requirements
                )
                
                progress_bar.progress(60)
                status_text.text("✨ Optimizing content...")
                
                # Store result
                st.session_state.tailored_cv = tailored_cv
                
                progress_bar.progress(100)
                status_text.text("✅ CV tailoring completed!")
                
                st.success("🎉 Your CV has been successfully tailored!")
                
                # Show tailoring score
                score = tailored_cv.tailoring_score
                st.metric(
                    "🎯 Tailoring Score", 
                    f"{score:.1f}%",
                    delta=f"{score-75:.1f}%" if score > 75 else None
                )
                
                # Add continue button after successful tailoring
                st.markdown("---")
                if st.button("➡️ Continue to Document Generation", type="primary", key="continue_to_doc"):
                    st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error during CV tailoring: {error_msg}")
                
                # Provide specific troubleshooting tips
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    st.info("💡 **Tip**: Connection issue with the AI service. Please check that Ollama is running (`ollama serve`) and try again.")
                elif "JSON parse failed" in error_msg or "Extra data" in error_msg:
                    st.info("💡 **Tip**: The AI model's response couldn't be parsed properly. This often happens with complex resumes. Try simplifying your resume or job description.")
                elif "Connection" in error_msg or "timeout" in error_msg.lower():
                    st.info("💡 **Tip**: Connection issue with the AI service. Please check that Ollama is running (`ollama serve`) and try again.")
                else:
                    st.info("💡 **Tip**: An unexpected error occurred. Please check the logs for more details and try again.")
                
                # Provide debugging information if available
                with st.expander("🔧 Debug Information (for developers)"):
                    st.code(f"""
Error Type: {type(e).__name__}
Error Message: {error_msg}
                    
Please check the application logs for more detailed information.
                    """, language="text")
    
    # Display current results if available
    if st.session_state.tailored_cv:
        st.markdown("### 📈 Tailoring Results")
        display_tailoring_results(st.session_state.tailored_cv)


def display_tailoring_results(tailored_cv):
    """Display tailoring results."""
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Match Score", f"{tailored_cv.tailoring_score:.1f}%")
    
    with col2:
        skills_count = sum(len(skills) for skills in tailored_cv.skills.values()) if tailored_cv.skills else 0
        st.metric("🛠️ Skills", skills_count)
    
    with col3:
        exp_count = len(tailored_cv.experience) if tailored_cv.experience else 0
        st.metric("💼 Experiences", exp_count)
    
    with col4:
        proj_count = len(tailored_cv.projects) if tailored_cv.projects else 0
        st.metric("🚀 Projects", proj_count)
    
    # Show optimized content
    with st.expander("📝 Optimized Professional Summary", expanded=True):
        st.write(tailored_cv.summary)
    
    if tailored_cv.optimization_notes:
        with st.expander("💡 Optimization Notes"):
            for note in tailored_cv.optimization_notes:
                st.info(note)
    
    # Add continue button for already completed tailoring
    st.markdown("---")
    if st.button("➡️ Continue to Document Generation", type="primary", key="continue_from_results"):
        st.rerun()


def handle_document_generation():
    """Handle final document generation."""
    st.markdown('<div class="section-header">📄 Generate Final CV</div>', unsafe_allow_html=True)
    
    if not st.session_state.tailored_cv:
        st.warning("⚠️ Please complete the CV tailoring step first.")
        return
    
    # Check PDF availability
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        pdf_available = True
    except ImportError:
        pdf_available = False
    
    # Document format selection
    st.subheader("📋 Document Options")
    
    # Format selection: HTML always available, PDF conditionally
    format_options = ["HTML"]
    if pdf_available:
        format_options.append("PDF")
    else:
        st.warning("⚠️ PDF generation is not available. Please install reportlab: `pip install reportlab`")
    
    format_type = st.selectbox(
        "📄 Choose Output Format:",
        format_options,
        index=0,
        help="Select the output format for your CV"
    )
    
    if format_type == "HTML":
        st.info("📄 **HTML Format**: Interactive web-based resume with modern styling")
    else:
        st.info("📄 **PDF Format**: Print-ready document with preserved styling")
    
    # Template selection
    template_style = st.selectbox(
        "🎨 Choose Template Style:",
        ["modern", "professional", "clean"],
        index=0,
        help="Select the visual style for your CV"
    )
    
    if st.button("📄 Generate CV Document", type="primary"):
        with st.spinner(f"Generating your professional CV in {format_type} format..."):
            try:
                # Generate document
                generator = CVDocumentGenerator()
                
                # Convert TailoredCV object to dictionary for Jinja2 template
                cv_dict = asdict(st.session_state.tailored_cv)
                
                if format_type == "HTML":
                    document_content = generator.generate_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    html_filename = f"tailored_cv_{template_style}_{timestamp}.html"
                    
                    # Save to data/output directory
                    output_path = Path("data/output") / html_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(document_content)
                        
                        # Verify the file was created
                        if output_path.exists():
                            file_size = output_path.stat().st_size
                            st.success(f"✅ HTML CV document generated and saved successfully!")
                            st.info(f"📁 File saved to: {output_path} ({file_size:,} bytes)")
                        else:
                            st.error(f"❌ Failed to save HTML file to {output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving HTML file: {str(e)}")
                        logger.error(f"Error saving HTML file: {str(e)}")
                    
                    # Store the generated document
                    st.session_state.generated_document = {
                        'html_content': document_content,
                        'html_filename': html_filename,
                        'output_path': str(output_path),
                        'format': 'HTML'
                    }
                    
                    # Preview
                    with st.expander("👁️ Document Preview", expanded=True):
                        st.components.v1.html(document_content, height=600, scrolling=True)
                    
                    # Download button for HTML
                    st.download_button(
                        label="⬇️ Download HTML CV",
                        data=document_content,
                        file_name=st.session_state.generated_document['html_filename'],
                        mime='text/html',
                        help="Click to download your tailored CV as HTML"
                    )
                
                elif format_type == "PDF":
                    # Generate both HTML (for preview) and PDF
                    html_content = generator.generate_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    pdf_bytes = generator.generate_pdf_from_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    # Generate filenames with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    html_filename = f"tailored_cv_{template_style}_{timestamp}.html"
                    pdf_filename = f"tailored_cv_{template_style}_{timestamp}.pdf"
                    
                    # Save both files to data/output directory
                    output_dir = Path("data/output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    html_output_path = output_dir / html_filename
                    pdf_output_path = output_dir / pdf_filename
                    
                    # Save HTML file
                    try:
                        with open(html_output_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Verify the file was created
                        if html_output_path.exists():
                            file_size = html_output_path.stat().st_size
                            st.success(f"✅ HTML file saved successfully: {html_output_path} ({file_size:,} bytes)")
                        else:
                            st.error(f"❌ Failed to save HTML file to {html_output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving HTML file: {str(e)}")
                        logger.error(f"Error saving HTML file: {str(e)}")
                    
                    # Save PDF file
                    try:
                        with open(pdf_output_path, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        # Verify the file was created
                        if pdf_output_path.exists():
                            file_size = pdf_output_path.stat().st_size
                            st.success(f"✅ PDF file saved successfully: {pdf_output_path} ({file_size:,} bytes)")
                        else:
                            st.error(f"❌ Failed to save PDF file to {pdf_output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving PDF file: {str(e)}")
                        logger.error(f"Error saving PDF file: {str(e)}")
                    
                    # Store the generated documents
                    st.session_state.generated_document = {
                        'html_content': html_content,
                        'pdf_content': pdf_bytes,
                        'html_filename': html_filename,
                        'pdf_filename': pdf_filename,
                        'html_output_path': str(html_output_path),
                        'pdf_output_path': str(pdf_output_path),
                        'format': 'PDF'
                    }
                    
                    st.success("✅ PDF CV document with HTML preview generated successfully!")
                    st.info(f"📁 Both files saved to data/output directory")
                    
                    # Show files in output directory
                    try:
                        output_files = list(Path("data/output").glob("*.html")) + list(Path("data/output").glob("*.pdf"))
                        if output_files:
                            st.markdown("**Recent files in output directory:**")
                            for file_path in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                                file_size = file_path.stat().st_size
                                st.markdown(f"- {file_path.name} ({file_size:,} bytes)")
                    except Exception as e:
                        logger.error(f"Error listing output files: {str(e)}")
                    
                    # Preview (show HTML version)
                    with st.expander("👁️ Document Preview (HTML version)", expanded=True):
                        st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    # Download buttons for both formats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="⬇️ Download PDF CV",
                            data=pdf_bytes,
                            file_name=st.session_state.generated_document['pdf_filename'],
                            mime='application/pdf',
                            help="Click to download your tailored CV as PDF"
                        )
                    with col2:
                        st.download_button(
                            label="⬇️ Download HTML CV",
                            data=html_content,
                            file_name=st.session_state.generated_document['html_filename'],
                            mime='text/html',
                            help="Click to download your tailored CV as HTML"
                        )
                
            except Exception as e:
                st.error(f"❌ Error generating document: {str(e)}")
                logger.error(f"Document generation error: {str(e)}")


def main():
    """Main application function."""
    # Handle command line arguments if run directly
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="HAAI++ Capstone Project - Multi-LLM CV Builder"
        )
        parser.add_argument(
            "--host", 
            default="http://localhost:11434",
            help="Ollama server host (default: http://localhost:11434)"
        )
        parser.add_argument(
            "--timeout", 
            type=int, 
            default=120,
            help="Request timeout in seconds (default: 120)"
        )
        
        args = parser.parse_args()
        
        # Set environment variables for the application
        os.environ["OLLAMA_HOST"] = args.host
        os.environ["OLLAMA_TIMEOUT"] = str(args.timeout)
        
        print("=" * 60)
        print("🎓 HAAI++ Capstone Project - CV Builder")
        print("📋 Project ID: HPPCS[01]")
        print("🤖 Multi-LLM Architecture: Gemma 2B + Llama2 7B")
        print("=" * 60)
        print(f"🔧 Ollama Host: {args.host}")
        print(f"⏱️  Timeout: {args.timeout}s")
        print("🚀 Starting CV Builder application...")
        print("=" * 60)
    
    # Header
    st.markdown('<div class="main-header">🤖 Multi-LLM AI-Powered CV Builder</div>', unsafe_allow_html=True)
    
    # Ensure session state is initialized
    init_session_state()
    
    st.markdown("""
    <div class="info-box">
    <h4>Welcome to the Multi-LLM AI-Powered CV Builder! 🎓</h4>
    <p>This application uses a specialized dual-model architecture for optimal CV tailoring:</p>
    <ul>
    <li><strong>Gemma 2B</strong>: Fast, efficient content generation and professional writing</li>
    <li><strong>Llama2 7B</strong>: Advanced data analysis and requirement matching</li>
    </ul>
    <p>Simply upload your existing resume and provide a job description to get started.</p>
    <p><em>Built for HAAI++ Capstone Project with justified multi-LLM architecture.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Ollama configuration
        st.subheader("🤖 Multi-LLM Configuration")
        ollama_host = st.text_input(
            "Ollama Host", 
            value=st.session_state.get("ollama_host", "http://localhost:11434"),
            key="ollama_host_input"
        )
        
        # Show current multi-model setup
        st.markdown("""
        **Active Models:**
        - 🎨 **Gemma 2B**: Content Generation
        - 🔍 **Llama2 7B**: Data Analysis
        """)
        
        if st.button("🔄 Initialize/Reconnect AI"):
            try:
                st.session_state.llm_client = None
                st.session_state.ollama_host = ollama_host
                settings.ollama_host = ollama_host
                initialize_services()
                st.success("✅ AI services reconnected successfully!")
            except Exception as e:
                st.error(f"❌ Failed to connect to AI services: {str(e)}")
                st.info("💡 Make sure Ollama is running with: `ollama serve`")
        
        st.divider()
        
        # Process steps indicator
        st.subheader("📋 Process Steps")
        steps = [
            ("1. Upload Resume", st.session_state.resume_data),
            ("2. Add Job Description", st.session_state.job_requirements), 
            ("3. AI Tailoring", st.session_state.tailored_cv),
            ("4. Generate CV", st.session_state.generated_document)
        ]
        
        for step_name, is_completed in steps:
            if is_completed:
                st.success(f"✅ {step_name}")
            else:
                st.write(f"⏳ {step_name}")
        
        # Current step indicator
        if not st.session_state.resume_data:
            st.info("👆 **Current Step**: Upload Resume")
        elif not st.session_state.job_requirements:
            st.info("👆 **Current Step**: Add Job Description")
        elif not st.session_state.tailored_cv:
            st.info("👆 **Current Step**: AI Tailoring")
        elif not st.session_state.generated_document:
            st.info("👆 **Current Step**: Generate CV")
        else:
            st.success("🎉 **All Steps Complete!**")
    
    # Initialize services
    if not initialize_services():
        st.stop()
    
    # Main content - conditional rendering based on progress
    if not st.session_state.resume_data:
        # Step 1: Resume Upload
        handle_resume_upload()
    elif not st.session_state.job_requirements:
        # Step 2: Job Description
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Resume uploaded successfully!")
        with col2:
            if st.button("� Upload Different Resume"):
                st.session_state.resume_data = None
                st.rerun()
        
        handle_job_description()
    elif not st.session_state.tailored_cv:
        # Step 3: AI Tailoring
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Resume and job requirements processed!")
        with col2:
            if st.button("🔄 Restart Process"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.rerun()
        
        handle_cv_tailoring()
    elif not st.session_state.generated_document:
        # Step 4: Document Generation
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ CV tailoring completed!")
        with col2:
            if st.button("🔄 Restart Process"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.session_state.tailored_cv = None
                st.rerun()
        
        handle_document_generation()
    else:
        # Step 5: Final Results
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ All steps completed! Your CV is ready!")
        with col2:
            if st.button("🔄 Start New CV"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.session_state.tailored_cv = None
                st.session_state.generated_document = None
                st.rerun()
        
        # Show final document
        handle_document_generation()


if __name__ == "__main__":
    main()