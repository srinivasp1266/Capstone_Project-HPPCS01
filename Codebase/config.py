"""
Configuration module for CV Builder application.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Ollama Settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="gemma:2b", description="Ollama model name")
    ollama_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Application Settings
    max_file_size_mb: int = Field(default=10, description="Maximum upload file size in MB")
    supported_formats: list = Field(default=["pdf", "docx", "doc"], description="Supported file formats")
    
    # Paths
    upload_dir: str = Field(default=".", description="Upload directory")
    output_dir: str = Field(default=".", description="Output directory")
    templates_dir: str = Field(default=".", description="Templates directory")
    
    # LLM Settings
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM response")
    temperature: float = Field(default=0.3, description="LLM temperature")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create required directories if they don't exist."""
    directories = [
        settings.upload_dir,
        settings.output_dir,
        settings.templates_dir
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Default prompts configuration
PROMPTS = {
    "extract_resume": """
    Please extract the following information from the resume text below and return it in JSON format:
    
    Required fields:
    - name: Full name
    - email: Email address
    - phone: Phone number
    - address: Physical address
    - linkedin: LinkedIn profile URL
    - summary: Professional summary/objective
    - education: List of education entries with degree, institution, year, GPA (if available)
    - experience: List of work experience with company, position, dates, responsibilities
    - skills: List of technical and soft skills
    - projects: List of projects with name, description, technologies
    - certifications: List of certifications with name, issuer, date
    - achievements: Notable achievements or awards
    
    Resume text:
    {resume_text}
    
    Return only valid JSON without any additional text or formatting.
    """,
    
    "extract_job": """
    Please analyze the job description below and extract key information in JSON format:
    
    Required fields:
    - title: Job title
    - company: Company name
    - location: Job location
    - employment_type: Full-time, Part-time, Contract, etc.
    - experience_level: Junior, Mid-level, Senior, etc.
    - required_skills: List of required technical skills
    - preferred_skills: List of preferred skills
    - responsibilities: List of key responsibilities
    - qualifications: Required qualifications/education
    - keywords: Important keywords for ATS optimization
    - salary_range: Salary range if mentioned
    
    Job description:
    {job_description}
    
    Return only valid JSON without any additional text or formatting.
    """,
    
    "tailor_resume": """
    Based on the candidate's profile and the target job requirements, create a tailored resume section.
    
    Candidate Profile:
    {candidate_profile}
    
    Target Job Requirements:
    {job_requirements}
    
    Section to tailor: {section}
    Original content: {original_content}
    
    Instructions:
    1. Incorporate relevant keywords from the job requirements
    2. Highlight experiences and skills that match the job
    3. Use action verbs and quantifiable achievements
    4. Ensure ATS-friendly formatting
    5. Keep the content truthful but optimized
    
    Return the tailored content for the {section} section.
    """,
    
    "generate_summary": """
    Create a professional summary for the candidate based on their profile and the target job.
    
    Candidate Profile:
    {candidate_profile}
    
    Target Job:
    {target_job}
    
    Create a compelling 3-4 sentence professional summary that:
    1. Highlights relevant experience and skills
    2. Incorporates key job requirements
    3. Shows value proposition
    4. Uses industry-specific keywords
    
    Return only the summary text without additional formatting.
    """
}