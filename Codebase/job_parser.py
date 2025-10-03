"""
Job description parsing and analysis module.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class JobRequirements:
    """Job requirements structure."""
    title: str = ""
    company: str = ""
    location: str = ""
    employment_type: str = ""
    experience_level: str = ""
    salary_range: str = ""
    required_skills: List[str] = None
    preferred_skills: List[str] = None
    responsibilities: List[str] = None
    qualifications: List[str] = None
    keywords: List[str] = None
    benefits: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.required_skills is None:
            self.required_skills = []
        if self.preferred_skills is None:
            self.preferred_skills = []
        if self.responsibilities is None:
            self.responsibilities = []
        if self.qualifications is None:
            self.qualifications = []
        if self.keywords is None:
            self.keywords = []
        if self.benefits is None:
            self.benefits = []


class JobDescriptionParser:
    """Parser for extracting structured data from job descriptions."""
    
    def __init__(self, llm_service=None):
        """
        Initialize job description parser.
        
        Args:
            llm_service: LLM service for intelligent extraction
        """
        self.llm_service = llm_service
        
        # Regex patterns for basic extraction
        self.salary_pattern = re.compile(r'\$[\d,]+(?:\s*-\s*\$?[\d,]+)?(?:\s*(?:per|/)\s*(?:year|hour|month))?', re.IGNORECASE)
        self.experience_pattern = re.compile(r'(\d+)[\+]?\s*(?:to\s*(\d+))?\s*years?\s*(?:of\s*)?experience', re.IGNORECASE)
        
        # Common section keywords
        self.responsibility_keywords = [
            'responsibilities', 'duties', 'what you\'ll do', 'role', 'you will'
        ]
        
        self.qualification_keywords = [
            'qualifications', 'requirements', 'must have', 'required',
            'minimum qualifications', 'basic qualifications'
        ]
        
        self.preferred_keywords = [
            'preferred', 'nice to have', 'bonus', 'plus', 'preferred qualifications'
        ]
        
        self.benefit_keywords = [
            'benefits', 'perks', 'we offer', 'compensation', 'package'
        ]
        
        # Skill categories
        self.technical_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'scala',
            'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'asp.net',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'git', 'jenkins', 'ci/cd', 'devops',
            'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch',
            'hadoop', 'spark', 'kafka', 'elasticsearch'
        ]
        
        self.soft_skills = [
            'communication', 'leadership', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'organized', 'detail-oriented',
            'self-motivated', 'collaborative', 'innovative', 'strategic thinking'
        ]
    
    def parse_job_description(self, job_text: str) -> JobRequirements:
        """
        Parse job description text into structured data.
        
        Args:
            job_text: Raw job description text
            
        Returns:
            JobRequirements object with extracted information
        """
        if self.llm_service:
            return self._parse_with_llm(job_text)
        else:
            return self._parse_with_regex(job_text)
    
    def _parse_with_llm(self, job_text: str) -> JobRequirements:
        """Parse job description using LLM service."""
        try:
            # Get structured data from LLM
            llm_data = self.llm_service.extract_job_requirements(job_text)
            
            # Convert to JobRequirements structure
            job_req = JobRequirements(
                title=llm_data.get('title', '') or '',
                company=llm_data.get('company', '') or '',
                location=llm_data.get('location', '') or '',
                employment_type=llm_data.get('employment_type', '') or '',
                experience_level=llm_data.get('experience_level', '') or '',
                salary_range=llm_data.get('salary_range', '') or '',
                required_skills=llm_data.get('required_skills', []) or [],
                preferred_skills=llm_data.get('preferred_skills', []) or [],
                responsibilities=llm_data.get('responsibilities', []) or [],
                qualifications=llm_data.get('qualifications', []) or [],
                keywords=llm_data.get('keywords', []) or [],
                benefits=llm_data.get('benefits', []) or [],
                description=job_text or ''
            )
            
            # Generate additional keywords if not provided
            if not job_req.keywords:
                job_req.keywords = self._extract_keywords(job_text)
            
            return job_req
            
        except Exception as e:
            logger.error(f"Error parsing job description with LLM: {str(e)}")
            # Fallback to regex parsing
            return self._parse_with_regex(job_text)
    
    def _parse_with_regex(self, job_text: str) -> JobRequirements:
        """Parse job description using regex patterns."""
        job_req = JobRequirements(description=job_text)
        
        # Extract basic information
        job_req.title = self._extract_job_title(job_text)
        job_req.company = self._extract_company_name(job_text)
        job_req.location = self._extract_location(job_text)
        job_req.employment_type = self._extract_employment_type(job_text)
        job_req.salary_range = self._extract_salary(job_text)
        job_req.experience_level = self._extract_experience_level(job_text)
        
        # Split into sections and extract detailed information
        sections = self._split_into_sections(job_text)
        
        for section_name, section_content in sections.items():
            if any(keyword in section_name.lower() for keyword in self.responsibility_keywords):
                job_req.responsibilities.extend(self._extract_list_items(section_content))
            elif any(keyword in section_name.lower() for keyword in self.qualification_keywords):
                job_req.qualifications.extend(self._extract_list_items(section_content))
            elif any(keyword in section_name.lower() for keyword in self.preferred_keywords):
                job_req.preferred_skills.extend(self._extract_list_items(section_content))
            elif any(keyword in section_name.lower() for keyword in self.benefit_keywords):
                job_req.benefits.extend(self._extract_list_items(section_content))
        
        # Extract skills from entire text
        all_skills = self._extract_skills(job_text)
        job_req.required_skills = all_skills['technical'] + all_skills['tools']
        
        # Extract keywords
        job_req.keywords = self._extract_keywords(job_text)
        
        return job_req
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from text."""
        lines = text.split('\n')
        
        # Common job title patterns
        title_patterns = [
            r'(?:job title|position|role):\s*(.+)',
            r'hiring\s+(?:for\s+)?(.+?)(?:\s*[-|]|$)',
            r'^(.+?)(?:\s*[-|]\s*.+)?$'  # First line heuristic
        ]
        
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line:
                continue
                
            for pattern in title_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if len(title.split()) <= 6 and title:  # Reasonable title length
                        return title
        
        return ""
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from text."""
        patterns = [
            r'(?:company|employer|organization):\s*(.+)',
            r'at\s+(.+?)(?:\s*[-|]|$)',
            r'join\s+(.+?)(?:\s+team|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                if company and len(company.split()) <= 5:
                    return company
        
        return ""
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text."""
        patterns = [
            r'(?:location|based in|office in):\s*(.+)',
            r'in\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?)',
            r'([A-Z][a-z]+,\s*[A-Z]{2})',
            r'remote|hybrid|on-site'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_employment_type(self, text: str) -> str:
        """Extract employment type from text."""
        types = ['full-time', 'part-time', 'contract', 'temporary', 'internship', 'freelance']
        
        text_lower = text.lower()
        for emp_type in types:
            if emp_type in text_lower:
                return emp_type.title()
        
        return ""
    
    def _extract_salary(self, text: str) -> str:
        """Extract salary information from text."""
        matches = self.salary_pattern.findall(text)
        if matches:
            return matches[0]
        return ""
    
    def _extract_experience_level(self, text: str) -> str:
        """Extract experience level from text."""
        experience_matches = self.experience_pattern.findall(text)
        
        if experience_matches:
            min_exp = int(experience_matches[0][0])
            if min_exp <= 2:
                return "Entry Level"
            elif min_exp <= 5:
                return "Mid Level"
            else:
                return "Senior Level"
        
        # Look for explicit level mentions
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['senior', 'lead', 'principal']):
            return "Senior Level"
        elif any(keyword in text_lower for keyword in ['junior', 'entry', 'graduate']):
            return "Entry Level"
        else:
            return "Mid Level"
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split job description into sections."""
        sections = {}
        current_section = "general"
        current_content = []
        
        section_headers = [
            'responsibilities', 'duties', 'requirements', 'qualifications',
            'skills', 'experience', 'benefits', 'perks', 'about',
            'what you\'ll do', 'what we offer', 'nice to have'
        ]
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            is_section_header = False
            for header in section_headers:
                if (header in line_lower and 
                    len(line.split()) <= 5 and
                    (line_lower.startswith(header) or line_lower.endswith(header))):
                    
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = header
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text."""
        items = []
        
        # Split by common list indicators
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove bullet points and clean up
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                item = line[1:].strip()
                if item:
                    items.append(item)
            elif re.match(r'^\d+\.', line):
                item = re.sub(r'^\d+\.', '', line).strip()
                if item:
                    items.append(item)
            else:
                # Split by sentences if no clear list structure
                sentences = re.split(r'[.;]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:
                        items.append(sentence)
        
        return items
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from job description."""
        skills = {"technical": [], "soft": [], "tools": []}
        
        text_lower = text.lower()
        
        # Extract technical skills
        for skill in self.technical_skills:
            if skill.lower() in text_lower:
                skills["technical"].append(skill)
        
        # Extract soft skills
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                skills["soft"].append(skill)
        
        # Extract tool mentions
        tool_patterns = [
            r'\b(?:experience with|knowledge of|proficiency in|familiar with)\s+([^.]+)',
            r'\b(?:using|working with)\s+([A-Za-z0-9\s,]+)'
        ]
        
        for pattern in tool_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                tools = [tool.strip() for tool in match.split(',')]
                skills["tools"].extend([tool for tool in tools if tool and len(tool) > 1])
        
        # Remove duplicates
        for category in skills:
            skills[category] = list(set(skills[category]))
        
        return skills
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords for ATS optimization."""
        keywords = []
        
        # Technical terms
        tech_pattern = re.compile(r'\b(?:' + '|'.join(self.technical_skills) + r')\b', re.IGNORECASE)
        keywords.extend([match.group() for match in tech_pattern.finditer(text)])
        
        # Job-related terms
        job_terms = [
            'develop', 'design', 'implement', 'manage', 'lead', 'analyze',
            'optimize', 'collaborate', 'coordinate', 'execute', 'deliver',
            'maintain', 'support', 'troubleshoot', 'document', 'test'
        ]
        
        for term in job_terms:
            if term.lower() in text.lower():
                keywords.append(term)
        
        # Industry-specific terms
        industry_pattern = re.compile(
            r'\b(?:agile|scrum|devops|microservices|api|database|frontend|backend|'
            r'fullstack|cloud|mobile|web|software|application|system|platform|'
            r'architecture|security|performance|scalability)\b',
            re.IGNORECASE
        )
        keywords.extend([match.group() for match in industry_pattern.finditer(text)])
        
        # Remove duplicates and sort
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def analyze_job_match(self, job_requirements: JobRequirements, candidate_skills: List[str]) -> Dict[str, Any]:
        """
        Analyze how well a candidate matches the job requirements.
        
        Args:
            job_requirements: Job requirements object
            candidate_skills: List of candidate skills
            
        Returns:
            Match analysis dictionary
        """
        required_skills = [skill.lower() for skill in job_requirements.required_skills]
        preferred_skills = [skill.lower() for skill in job_requirements.preferred_skills]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Calculate matches
        required_matches = [skill for skill in required_skills if skill in candidate_skills_lower]
        preferred_matches = [skill for skill in preferred_skills if skill in candidate_skills_lower]
        
        # Calculate percentages
        required_match_percentage = (len(required_matches) / len(required_skills) * 100) if required_skills else 100
        preferred_match_percentage = (len(preferred_matches) / len(preferred_skills) * 100) if preferred_skills else 0
        
        # Overall match score
        overall_score = (required_match_percentage * 0.7 + preferred_match_percentage * 0.3)
        
        return {
            'overall_score': round(overall_score, 1),
            'required_match_percentage': round(required_match_percentage, 1),
            'preferred_match_percentage': round(preferred_match_percentage, 1),
            'required_matches': required_matches,
            'preferred_matches': preferred_matches,
            'missing_required_skills': [skill for skill in required_skills if skill not in candidate_skills_lower],
            'missing_preferred_skills': [skill for skill in preferred_skills if skill not in candidate_skills_lower],
            'match_level': self._get_match_level(overall_score)
        }
    
    def _get_match_level(self, score: float) -> str:
        """Get match level based on score."""
        if score >= 80:
            return "Excellent Match"
        elif score >= 60:
            return "Good Match"
        elif score >= 40:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def to_dict(self, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Convert JobRequirements to dictionary."""
        return asdict(job_requirements)
    
    def from_dict(self, data: Dict[str, Any]) -> JobRequirements:
        """Create JobRequirements from dictionary."""
        return JobRequirements(**data)