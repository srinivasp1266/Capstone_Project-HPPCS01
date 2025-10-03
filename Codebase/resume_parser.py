"""
Resume data extraction and processing module.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonalInfo:
    """Personal information structure."""
    name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    linkedin: str = ""
    website: str = ""
    github: str = ""


@dataclass
class Education:
    """Education entry structure."""
    degree: str = ""
    institution: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    gpa: str = ""
    relevant_coursework: List[str] = None
    honors: List[str] = None
    
    def __post_init__(self):
        if self.relevant_coursework is None:
            self.relevant_coursework = []
        if self.honors is None:
            self.honors = []


@dataclass
class Experience:
    """Work experience entry structure."""
    company: str = ""
    position: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    responsibilities: List[str] = None
    achievements: List[str] = None
    technologies: List[str] = None
    
    def __post_init__(self):
        if self.responsibilities is None:
            self.responsibilities = []
        if self.achievements is None:
            self.achievements = []
        if self.technologies is None:
            self.technologies = []


@dataclass
class Project:
    """Project entry structure."""
    name: str = ""
    description: str = ""
    technologies: List[str] = None
    start_date: str = ""
    end_date: str = ""
    url: str = ""
    github_url: str = ""
    key_features: List[str] = None
    
    def __post_init__(self):
        if self.technologies is None:
            self.technologies = []
        if self.key_features is None:
            self.key_features = []


@dataclass
class Certification:
    """Certification entry structure."""
    name: str = ""
    issuer: str = ""
    date: str = ""
    expiry_date: str = ""
    credential_id: str = ""
    url: str = ""


@dataclass
class ResumeData:
    """Complete resume data structure."""
    personal_info: PersonalInfo = None
    summary: str = ""
    education: List[Education] = None
    experience: List[Experience] = None
    skills: Dict[str, List[str]] = None
    projects: List[Project] = None
    certifications: List[Certification] = None
    achievements: List[str] = None
    languages: List[str] = None
    
    def __post_init__(self):
        if self.personal_info is None:
            self.personal_info = PersonalInfo()
        if self.education is None:
            self.education = []
        if self.experience is None:
            self.experience = []
        if self.skills is None:
            self.skills = {"technical": [], "soft": [], "tools": [], "languages": []}
        if self.projects is None:
            self.projects = []
        if self.certifications is None:
            self.certifications = []
        if self.achievements is None:
            self.achievements = []
        if self.languages is None:
            self.languages = []


class ResumeParser:
    """Parser for extracting structured data from resume text."""
    
    def __init__(self, llm_service=None):
        """
        Initialize resume parser.
        
        Args:
            llm_service: LLM service for intelligent extraction
        """
        self.llm_service = llm_service
        
        # Regex patterns for basic extraction
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[\d\s\-\(\)\.]{10,15}')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w\-]+', re.IGNORECASE)
        self.github_pattern = re.compile(r'github\.com/[\w\-]+', re.IGNORECASE)
        self.website_pattern = re.compile(r'https?://[\w\-\.]+\.[\w]{2,4}/?[\w\-\._~:/?#[\]@!\$&\'\(\)\*\+,;=]*', re.IGNORECASE)
    
    def parse_resume(self, resume_text: str) -> ResumeData:
        """
        Parse resume text into structured data.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            ResumeData object with extracted information
        """
        if self.llm_service:
            return self._parse_with_llm(resume_text)
        else:
            return self._parse_with_regex(resume_text)
    
    def _parse_with_llm(self, resume_text: str) -> ResumeData:
        """Parse resume using LLM service."""
        try:
            # Get structured data from LLM
            llm_data = self.llm_service.extract_resume_data(resume_text)
            
            # Convert to ResumeData structure
            resume_data = ResumeData()
            
            # Personal information
            if 'name' in llm_data or 'email' in llm_data:
                resume_data.personal_info = PersonalInfo(
                    name=llm_data.get('name', '') or '',
                    email=llm_data.get('email', '') or '',
                    phone=llm_data.get('phone', '') or '',
                    address=llm_data.get('address', '') or '',
                    linkedin=llm_data.get('linkedin', '') or '',
                    website=llm_data.get('website', '') or '',
                    github=llm_data.get('github', '') or ''
                )
            
            # Summary
            resume_data.summary = llm_data.get('summary', '') or ''
            
            # Education
            if 'education' in llm_data and isinstance(llm_data['education'], list):
                resume_data.education = []
                for edu in llm_data['education']:
                    if isinstance(edu, dict):
                        resume_data.education.append(Education(
                            degree=edu.get('degree', '') or '',
                            institution=edu.get('institution', '') or '',
                            location=edu.get('location', '') or '',
                            start_date=edu.get('start_date', '') or '',
                            end_date=edu.get('end_date', '') or '',
                            gpa=edu.get('gpa', '') or '',
                            relevant_coursework=edu.get('relevant_coursework', []) or [],
                            honors=edu.get('honors', []) or []
                        ))
            
            # Experience
            if 'experience' in llm_data and isinstance(llm_data['experience'], list):
                resume_data.experience = []
                for exp in llm_data['experience']:
                    if isinstance(exp, dict):
                        resume_data.experience.append(Experience(
                            company=exp.get('company', '') or '',
                            position=exp.get('position', '') or '',
                            location=exp.get('location', '') or '',
                            start_date=exp.get('start_date', '') or '',
                            end_date=exp.get('end_date', '') or '',
                            responsibilities=exp.get('responsibilities', []) or [],
                            achievements=exp.get('achievements', []) or [],
                            technologies=exp.get('technologies', []) or []
                        ))
            
            # Skills
            if 'skills' in llm_data:
                if isinstance(llm_data['skills'], dict):
                    resume_data.skills = llm_data['skills']
                elif isinstance(llm_data['skills'], list):
                    # Categorize skills automatically
                    resume_data.skills = self._categorize_skills(llm_data['skills'])
            
            # Projects
            if 'projects' in llm_data and isinstance(llm_data['projects'], list):
                resume_data.projects = []
                for proj in llm_data['projects']:
                    if isinstance(proj, dict):
                        resume_data.projects.append(Project(
                            name=proj.get('name', '') or '',
                            description=proj.get('description', '') or '',
                            technologies=proj.get('technologies', []) or [],
                            start_date=proj.get('start_date', '') or '',
                            end_date=proj.get('end_date', '') or '',
                            url=proj.get('url', '') or '',
                            github_url=proj.get('github_url', '') or '',
                            key_features=proj.get('key_features', []) or []
                        ))
            
            # Certifications
            if 'certifications' in llm_data and isinstance(llm_data['certifications'], list):
                resume_data.certifications = []
                for cert in llm_data['certifications']:
                    if isinstance(cert, dict):
                        resume_data.certifications.append(Certification(
                            name=cert.get('name', ''),
                            issuer=cert.get('issuer', ''),
                            date=cert.get('date', ''),
                            expiry_date=cert.get('expiry_date', ''),
                            credential_id=cert.get('credential_id', ''),
                            url=cert.get('url', '')
                        ))
            
            # Achievements and Languages with type checking
            achievements_data = llm_data.get('achievements', [])
            if isinstance(achievements_data, str):
                # If it's a string, try to split it by newlines or bullet points
                resume_data.achievements = [ach.strip() for ach in achievements_data.replace('•', '\n').split('\n') if ach.strip()]
            elif isinstance(achievements_data, list):
                resume_data.achievements = achievements_data
            else:
                resume_data.achievements = []
            
            # Ensure languages is always a list
            languages_data = llm_data.get('languages', [])
            if isinstance(languages_data, str):
                # If it's a string, try to split it
                resume_data.languages = [lang.strip() for lang in languages_data.split(',') if lang.strip()]
            elif isinstance(languages_data, list):
                resume_data.languages = languages_data
            else:
                resume_data.languages = []
            
            return resume_data
            
        except Exception as e:
            logger.error(f"Error parsing resume with LLM: {str(e)}")
            # Fallback to regex parsing
            return self._parse_with_regex(resume_text)
    
    def _parse_with_regex(self, resume_text: str) -> ResumeData:
        """Parse resume using regex patterns."""
        resume_data = ResumeData()
        
        # Extract personal information
        resume_data.personal_info = self._extract_personal_info(resume_text)
        
        # Extract sections using simple heuristics
        sections = self._split_into_sections(resume_text)
        
        for section_name, section_content in sections.items():
            if 'education' in section_name.lower():
                resume_data.education.extend(self._extract_education(section_content))
            elif 'experience' in section_name.lower() or 'work' in section_name.lower():
                resume_data.experience.extend(self._extract_experience(section_content))
            elif 'skill' in section_name.lower():
                skills = self._extract_skills(section_content)
                for category, skill_list in skills.items():
                    resume_data.skills[category].extend(skill_list)
            elif 'project' in section_name.lower():
                resume_data.projects.extend(self._extract_projects(section_content))
            elif 'certification' in section_name.lower():
                resume_data.certifications.extend(self._extract_certifications(section_content))
        
        return resume_data
    
    def _extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information using regex."""
        personal_info = PersonalInfo()
        
        # Email
        email_matches = self.email_pattern.findall(text)
        if email_matches:
            personal_info.email = email_matches[0]
        
        # Phone
        phone_matches = self.phone_pattern.findall(text)
        if phone_matches:
            personal_info.phone = phone_matches[0]
        
        # LinkedIn
        linkedin_matches = self.linkedin_pattern.findall(text)
        if linkedin_matches:
            personal_info.linkedin = f"https://{linkedin_matches[0]}"
        
        # GitHub
        github_matches = self.github_pattern.findall(text)
        if github_matches:
            personal_info.github = f"https://{github_matches[0]}"
        
        # Website
        website_matches = self.website_pattern.findall(text)
        for match in website_matches:
            if 'linkedin' not in match.lower() and 'github' not in match.lower():
                personal_info.website = match
                break
        
        # Name (heuristic: first line that looks like a name)
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if (len(line.split()) >= 2 and 
                not any(char.isdigit() for char in line) and
                not '@' in line and
                not 'http' in line.lower()):
                personal_info.name = line
                break
        
        return personal_info
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split resume text into sections."""
        sections = {}
        current_section = "header"
        current_content = []
        
        section_headers = [
            'education', 'experience', 'work experience', 'employment',
            'skills', 'technical skills', 'projects', 'certifications',
            'achievements', 'awards', 'summary', 'objective'
        ]
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            is_section_header = False
            for header in section_headers:
                if header in line_lower and len(line.split()) <= 3:
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
    
    def _extract_education(self, text: str) -> List[Education]:
        """Extract education entries from text."""
        # Simplified extraction - would need more sophisticated parsing
        educations = []
        
        # This is a basic implementation - in practice, you'd want more robust parsing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_edu = Education()
        
        for line in lines:
            if any(degree in line.lower() for degree in ['bachelor', 'master', 'phd', 'b.s.', 'm.s.', 'b.a.', 'm.a.']):
                if current_edu.degree or current_edu.institution:
                    educations.append(current_edu)
                current_edu = Education()
                current_edu.degree = line
            elif any(word in line.lower() for word in ['university', 'college', 'institute', 'school']):
                current_edu.institution = line
        
        if current_edu.degree or current_edu.institution:
            educations.append(current_edu)
        
        return educations
    
    def _extract_experience(self, text: str) -> List[Experience]:
        """Extract work experience entries from text."""
        experiences = []
        
        # Basic implementation - would need more sophisticated parsing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_exp = Experience()
        
        for line in lines:
            # Look for job titles or company names
            if len(line.split()) <= 5 and not line.startswith('-') and not line.startswith('•'):
                if current_exp.company or current_exp.position:
                    experiences.append(current_exp)
                current_exp = Experience()
                
                if any(word in line.lower() for word in ['engineer', 'developer', 'manager', 'analyst', 'specialist']):
                    current_exp.position = line
                else:
                    current_exp.company = line
            elif line.startswith('-') or line.startswith('•'):
                responsibility = line.lstrip('-•').strip()
                current_exp.responsibilities.append(responsibility)
        
        if current_exp.company or current_exp.position:
            experiences.append(current_exp)
        
        return experiences
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text."""
        skills = {"technical": [], "soft": [], "tools": [], "languages": []}
        
        # Remove common section indicators
        text = text.replace('•', '').replace('-', '')
        
        # Split by common delimiters
        skill_items = []
        for delimiter in [',', '|', '\n', ';']:
            if delimiter in text:
                skill_items = [item.strip() for item in text.split(delimiter) if item.strip()]
                break
        
        if not skill_items:
            skill_items = text.split()
        
        # Categorize skills (basic categorization)
        technical_keywords = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'node']
        tool_keywords = ['git', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'jira']
        language_keywords = ['english', 'spanish', 'french', 'german', 'chinese', 'japanese']
        
        for item in skill_items:
            item_lower = item.lower()
            
            if any(keyword in item_lower for keyword in technical_keywords):
                skills["technical"].append(item)
            elif any(keyword in item_lower for keyword in tool_keywords):
                skills["tools"].append(item)
            elif any(keyword in item_lower for keyword in language_keywords):
                skills["languages"].append(item)
            else:
                skills["technical"].append(item)  # Default to technical
        
        return skills
    
    def _extract_projects(self, text: str) -> List[Project]:
        """Extract project entries from text."""
        projects = []
        
        # Basic implementation
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_project = Project()
        
        for line in lines:
            if not line.startswith('-') and not line.startswith('•') and len(line.split()) <= 8:
                if current_project.name:
                    projects.append(current_project)
                current_project = Project()
                current_project.name = line
            elif line.startswith('-') or line.startswith('•'):
                description = line.lstrip('-•').strip()
                current_project.description += f" {description}"
        
        if current_project.name:
            projects.append(current_project)
        
        return projects
    
    def _extract_certifications(self, text: str) -> List[Certification]:
        """Extract certification entries from text."""
        certifications = []
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            if line and not line.startswith('-') and not line.startswith('•'):
                cert = Certification()
                parts = line.split(',')
                if parts:
                    cert.name = parts[0].strip()
                    if len(parts) > 1:
                        cert.issuer = parts[1].strip()
                certifications.append(cert)
        
        return certifications
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize a list of skills."""
        categorized = {"technical": [], "soft": [], "tools": [], "languages": []}
        
        technical_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
            'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch'
        ]
        
        tool_keywords = [
            'git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins',
            'jira', 'slack', 'microsoft office', 'excel', 'powerpoint', 'word'
        ]
        
        soft_keywords = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'organized'
        ]
        
        language_keywords = [
            'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
            'hindi', 'arabic', 'portuguese', 'russian'
        ]
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if any(keyword in skill_lower for keyword in technical_keywords):
                categorized["technical"].append(skill)
            elif any(keyword in skill_lower for keyword in tool_keywords):
                categorized["tools"].append(skill)
            elif any(keyword in skill_lower for keyword in soft_keywords):
                categorized["soft"].append(skill)
            elif any(keyword in skill_lower for keyword in language_keywords):
                categorized["languages"].append(skill)
            else:
                categorized["technical"].append(skill)  # Default to technical
        
        return categorized
    
    def to_dict(self, resume_data: ResumeData) -> Dict[str, Any]:
        """Convert ResumeData to dictionary."""
        return {
            'personal_info': asdict(resume_data.personal_info) if resume_data.personal_info else {},
            'summary': resume_data.summary,
            'education': [asdict(edu) for edu in resume_data.education],
            'experience': [asdict(exp) for exp in resume_data.experience],
            'skills': resume_data.skills,
            'projects': [asdict(proj) for proj in resume_data.projects],
            'certifications': [asdict(cert) for cert in resume_data.certifications],
            'achievements': resume_data.achievements,
            'languages': resume_data.languages
        }
    
    def from_dict(self, data: Dict[str, Any]) -> ResumeData:
        """Create ResumeData from dictionary."""
        resume_data = ResumeData()
        
        if 'personal_info' in data:
            resume_data.personal_info = PersonalInfo(**data['personal_info'])
        
        resume_data.summary = data.get('summary', '')
        
        if 'education' in data:
            resume_data.education = [Education(**edu) for edu in data['education']]
        
        if 'experience' in data:
            resume_data.experience = [Experience(**exp) for exp in data['experience']]
        
        resume_data.skills = data.get('skills', {"technical": [], "soft": [], "tools": [], "languages": []})
        
        if 'projects' in data:
            resume_data.projects = [Project(**proj) for proj in data['projects']]
        
        if 'certifications' in data:
            resume_data.certifications = [Certification(**cert) for cert in data['certifications']]
        
        resume_data.achievements = data.get('achievements', [])
        resume_data.languages = data.get('languages', [])
        
        return resume_data