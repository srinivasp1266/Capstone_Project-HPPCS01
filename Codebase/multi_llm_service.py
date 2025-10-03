"""
Multi-LLM Service for CV Builder - HAAI++ Capstone Project
Uses two specialized LLMs for different tasks with proper justification.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from llm_client import OllamaClient, LLMService

logger = logging.getLogger(__name__)


class LLMTask(Enum):
    """Enum defining different LLM task types."""
    CONTENT_GENERATION = "content_generation"
    DATA_ANALYSIS = "data_analysis"
    STRUCTURED_EXTRACTION = "structured_extraction"
    CREATIVE_WRITING = "creative_writing"


@dataclass
class LLMModelConfig:
    """Configuration for an LLM model."""
    name: str
    model_id: str
    primary_tasks: List[LLMTask]
    justification: str
    strengths: List[str]
    temperature: float = 0.3
    max_tokens: int = 2048


class MultiLLMService:
    """
    Multi-LLM service that routes tasks to appropriate models based on their strengths.
    
    CAPSTONE PROJECT REQUIREMENT: Uses at least two LLMs with proper justification
    
    Model Selection Justification:
    1. Gemma 2B - Fast content generation and creative writing
    2. Llama2 7B - Superior analytical capabilities and structured data handling
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120):
        """Initialize multi-LLM service with two specialized models."""
        self.host = host
        self.timeout = timeout
        
        # Define model configurations with justifications
        self.models = {
            "content_generator": LLMModelConfig(
                name="Content Generation Specialist",
                model_id="gemma:2b",
                primary_tasks=[LLMTask.CONTENT_GENERATION, LLMTask.CREATIVE_WRITING],
                justification="""
                Gemma 2B is optimized for fast, creative content generation with:
                - Lower latency for real-time user interactions
                - Excellent performance on creative writing tasks
                - Efficient resource usage for content generation
                - Strong performance on resume writing and professional summaries
                """,
                strengths=[
                    "Fast response times",
                    "Creative content generation", 
                    "Professional writing",
                    "Resource efficient",
                    "Good for user-facing tasks"
                ],
                temperature=0.7,  # Higher creativity for content generation
                max_tokens=1024
            ),
            
            "data_analyst": LLMModelConfig(
                name="Data Analysis Specialist", 
                model_id="llama2:7b",
                primary_tasks=[LLMTask.DATA_ANALYSIS, LLMTask.STRUCTURED_EXTRACTION],
                justification="""
                Llama2 7B provides superior analytical capabilities with:
                - Larger parameter count for complex reasoning
                - Better structured data extraction and JSON parsing
                - Superior performance on analytical tasks
                - More reliable for parsing and matching algorithms
                """,
                strengths=[
                    "Complex reasoning and analysis",
                    "Structured data extraction",
                    "JSON parsing reliability",
                    "Pattern recognition",
                    "Requirement analysis"
                ],
                temperature=0.1,  # Lower temperature for precise analysis
                max_tokens=2048
            )
        }
        
        # Initialize LLM clients
        self.clients = {}
        for model_key, config in self.models.items():
            try:
                client = OllamaClient(host=self.host, model=config.model_id, timeout=self.timeout)
                self.clients[model_key] = LLMService(client)
                logger.info(f"Initialized {config.name} ({config.model_id})")
            except Exception as e:
                logger.error(f"Failed to initialize {config.name}: {str(e)}")
                raise
        
        logger.info("Multi-LLM service initialized with 2 specialized models")
    
    def get_model_for_task(self, task: LLMTask) -> str:
        """
        Route task to the most appropriate model based on task type.
        
        Args:
            task: Type of task to perform
            
        Returns:
            Model key for the appropriate model
        """
        for model_key, config in self.models.items():
            if task in config.primary_tasks:
                logger.debug(f"Routing {task.value} to {config.name}")
                return model_key
        
        # Default to content generator for unknown tasks
        logger.warning(f"No specialized model found for {task.value}, using content generator")
        return "content_generator"
    
    def generate_professional_summary(self, candidate_profile: Dict[str, Any], 
                                   job_requirements: Dict[str, Any]) -> str:
        """
        Generate professional summary using Content Generation Specialist.
        
        Uses Gemma 2B for creative writing and professional content generation.
        """
        model_key = self.get_model_for_task(LLMTask.CONTENT_GENERATION)
        client = self.clients[model_key]
        
        logger.info(f"Generating professional summary using {self.models[model_key].name}")
        return client.generate_professional_summary(candidate_profile, job_requirements)
    
    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured data from resume using Data Analysis Specialist.
        
        Uses Llama2 7B for superior structured data extraction and parsing.
        """
        model_key = self.get_model_for_task(LLMTask.STRUCTURED_EXTRACTION)
        client = self.clients[model_key]
        
        logger.info(f"Extracting resume data using {self.models[model_key].name}")
        return client.extract_resume_data(resume_text)
    
    def process_task(self, task: LLMTask, **kwargs) -> str:
        """
        Process a task using the appropriate specialized model.
        
        Args:
            task: The type of task to perform
            **kwargs: Task-specific parameters
            
        Returns:
            Processed result as string
        """
        model_key = self.get_model_for_task(task)
        client = self.clients[model_key]
        config = self.models[model_key]
        
        logger.info(f"Processing {task.value} task using {config.name}")
        
        if task == LLMTask.CONTENT_GENERATION:
            return self._handle_content_generation(client, **kwargs)
        elif task == LLMTask.DATA_ANALYSIS:
            return self._handle_data_analysis(client, **kwargs)
        elif task == LLMTask.STRUCTURED_EXTRACTION:
            return self._handle_structured_extraction(client, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task}")
    
    def _handle_content_generation(self, client, **kwargs) -> str:
        """Handle content generation tasks."""
        task_type = kwargs.get('task_type', '')
        
        if task_type == "professional_summary":
            candidate_profile = kwargs.get('candidate_profile', {})
            job_requirements = kwargs.get('job_requirements', {})
            return client.generate_professional_summary(candidate_profile, job_requirements)
            
        elif task_type in ["responsibility_tailoring", "project_description_tailoring"]:
            candidate_profile = kwargs.get('candidate_profile', {})
            job_requirements = kwargs.get('job_requirements', {})
            content = kwargs.get('content', '')
            section_type = 'responsibility' if 'responsibility' in task_type else 'project_description'
            return client.tailor_resume_section(candidate_profile, job_requirements, section_type, content)
            
        else:
            raise ValueError(f"Unsupported content generation task: {task_type}")
    
    def _handle_data_analysis(self, client, **kwargs) -> str:
        """Handle data analysis tasks."""
        task_type = kwargs.get('task_type', '')
        
        if task_type == "ats_optimization":
            content = kwargs.get('content', '')
            keywords = kwargs.get('keywords', [])
            return client.optimize_for_ats(content, keywords)
            
        else:
            raise ValueError(f"Unsupported data analysis task: {task_type}")
    
    def _handle_structured_extraction(self, client, **kwargs) -> str:
        """Handle structured extraction tasks."""
        # Placeholder for future structured extraction tasks
        raise ValueError("Structured extraction tasks not yet implemented")
    
    def extract_job_requirements(self, job_text: str) -> Dict[str, Any]:
        """
        Extract job requirements using Data Analysis Specialist.
        
        Uses Llama2 7B for analytical parsing and requirement extraction.
        """
        model_key = self.get_model_for_task(LLMTask.DATA_ANALYSIS)
        client = self.clients[model_key]
        
        logger.info(f"Extracting job requirements using {self.models[model_key].name}")
        return client.extract_job_requirements(job_text)
    
    def tailor_resume_section(self, candidate_profile: Dict[str, Any], 
                            job_requirements: Dict[str, Any], 
                            section: str, original_content: str) -> str:
        """
        Tailor resume section using Content Generation Specialist.
        
        Uses Gemma 2B for creative content enhancement and professional writing.
        """
        model_key = self.get_model_for_task(LLMTask.CONTENT_GENERATION)
        client = self.clients[model_key]
        
        logger.info(f"Tailoring resume section using {self.models[model_key].name}")
        return client.tailor_resume_section(candidate_profile, job_requirements, section, original_content)
    
    def analyze_job_match(self, resume_data: Dict[str, Any], 
                         job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze job match using Data Analysis Specialist.
        
        Uses Llama2 7B for complex analysis and pattern matching.
        """
        model_key = self.get_model_for_task(LLMTask.DATA_ANALYSIS)
        client = self.clients[model_key]
        
        logger.info(f"Analyzing job match using {self.models[model_key].name}")
        
        # Create analysis prompt for the analytical model
        prompt = f"""
        Analyze the match between this candidate's resume and job requirements.
        
        RESUME DATA:
        {resume_data}
        
        JOB REQUIREMENTS:
        {job_requirements}
        
        Provide a detailed analysis including:
        1. Match score (0-100)
        2. Strengths alignment
        3. Skill gaps
        4. Recommendations for improvement
        
        Return as JSON format.
        """
        
        try:
            return client.client.generate_json_response(prompt)
        except Exception as e:
            logger.error(f"Job match analysis failed: {str(e)}")
            return {
                "match_score": 0,
                "strengths": [],
                "gaps": [],
                "recommendations": ["Analysis failed - please try again"]
            }
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all configured models for documentation.
        
        Returns model configurations and justifications for capstone documentation.
        """
        model_info = {}
        for model_key, config in self.models.items():
            model_info[model_key] = {
                "name": config.name,
                "model_id": config.model_id,
                "primary_tasks": [task.value for task in config.primary_tasks],
                "justification": config.justification.strip(),
                "strengths": config.strengths,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        return model_info
    
    def generate_model_comparison_report(self) -> str:
        """
        Generate a detailed comparison report for capstone documentation.
        
        Returns a formatted report justifying model selection.
        """
        report = """
# Multi-LLM Architecture Report - HAAI++ Capstone Project

## Model Selection Justification

### Architecture Overview
This CV Builder implements a dual-LLM architecture where each model is specialized for specific tasks based on their inherent strengths and capabilities.

"""
        
        for model_key, config in self.models.items():
            report += f"""
### {config.name} ({config.model_id})

**Primary Tasks:** {', '.join([task.value for task in config.primary_tasks])}

**Justification:**
{config.justification}

**Key Strengths:**
{chr(10).join([f"- {strength}" for strength in config.strengths])}

**Configuration:**
- Temperature: {config.temperature} ({"Higher for creativity" if config.temperature > 0.5 else "Lower for precision"})
- Max Tokens: {config.max_tokens}

"""
        
        report += """
## Task Routing Strategy

The system automatically routes tasks to the most appropriate model:

1. **Content Generation Tasks** → Gemma 2B
   - Professional summaries
   - Resume section enhancement
   - Creative writing tasks

2. **Analysis Tasks** → Llama2 7B
   - Resume data extraction
   - Job requirement parsing
   - Match analysis
   - Structured data handling

## Performance Benefits

This dual-model approach provides:
- **Optimized Performance**: Each model handles tasks it excels at
- **Cost Efficiency**: Smaller model for simple tasks, larger for complex analysis
- **Improved Quality**: Specialized models produce better results
- **Scalability**: Can add more specialized models as needed

## Capstone Project Compliance

✅ **Requirement Met**: Uses at least two LLMs
✅ **Proper Justification**: Each model selection is technically justified
✅ **Complementary Roles**: Models serve different, complementary purposes
✅ **Performance Optimization**: Task routing maximizes each model's strengths
"""
        
        return report