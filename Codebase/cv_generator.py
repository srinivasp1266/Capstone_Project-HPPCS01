"""
CV tailoring and generation engine.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from resume_parser import ResumeData, Experience, Project, Education
from job_parser import JobRequirements
from multi_llm_service import MultiLLMService, LLMTask

logger = logging.getLogger(__name__)


@dataclass
class TailoredCV:
    """Tailored CV structure."""
    personal_info: Dict[str, Any] = None
    summary: str = ""
    education: List[Dict[str, Any]] = None
    experience: List[Dict[str, Any]] = None
    skills: Dict[str, List[str]] = None
    projects: List[Dict[str, Any]] = None
    certifications: List[Dict[str, Any]] = None
    achievements: List[str] = None
    languages: List[str] = None
    tailoring_score: float = 0.0
    optimization_notes: List[str] = None
    
    def __post_init__(self):
        if self.personal_info is None:
            self.personal_info = {}
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
        if self.optimization_notes is None:
            self.optimization_notes = []


class CVTailoringEngine:
    """Engine for tailoring CVs to specific job requirements."""
    
    def __init__(self, multi_llm_service: MultiLLMService):
        """
        Initialize CV tailoring engine.
        
        Args:
            multi_llm_service: Multi-LLM service for intelligent tailoring using specialized models
        """
        self.multi_llm_service = multi_llm_service
        
        # ATS-friendly keywords and action verbs
        self.action_verbs = [
            'achieved', 'implemented', 'developed', 'designed', 'managed',
            'led', 'created', 'optimized', 'improved', 'reduced', 'increased',
            'delivered', 'executed', 'coordinated', 'collaborated', 'analyzed',
            'resolved', 'streamlined', 'enhanced', 'built', 'established'
        ]
        
        # Industry keywords
        self.industry_keywords = {
            'tech': ['software', 'development', 'programming', 'coding', 'algorithm',
                    'architecture', 'framework', 'api', 'database', 'cloud',
                    'microservices', 'scalability', 'performance', 'security'],
            'data': ['analytics', 'machine learning', 'ai', 'statistics',
                    'modeling', 'visualization', 'big data', 'python', 'sql',
                    'tableau', 'tensorflow', 'pytorch'],
            'management': ['leadership', 'strategy', 'planning', 'budget',
                          'team', 'project management', 'stakeholder', 'roi',
                          'process improvement', 'agile', 'scrum']
        }
    
    def tailor_cv(self, resume_data: ResumeData, job_requirements: JobRequirements) -> TailoredCV:
        """
        Tailor a CV to match job requirements.
        
        Args:
            resume_data: Original resume data
            job_requirements: Target job requirements
            
        Returns:
            TailoredCV object with optimized content
        """
        tailored_cv = TailoredCV()
        
        # Copy personal information (unchanged)
        if resume_data.personal_info:
            tailored_cv.personal_info = asdict(resume_data.personal_info)
        
        # Generate tailored professional summary
        tailored_cv.summary = self._tailor_summary(resume_data, job_requirements)
        
        # Tailor experience section
        tailored_cv.experience = self._tailor_experience(resume_data.experience, job_requirements)
        
        # Tailor skills section
        tailored_cv.skills = self._tailor_skills(resume_data.skills, job_requirements)
        
        # Tailor education (with relevant coursework emphasis)
        tailored_cv.education = self._tailor_education(resume_data.education, job_requirements)
        
        # Tailor projects section
        tailored_cv.projects = self._tailor_projects(resume_data.projects, job_requirements)
        
        # Copy certifications and achievements (with relevance ranking)
        tailored_cv.certifications = self._rank_certifications(resume_data.certifications, job_requirements)
        
        # Handle achievements with type checking
        if isinstance(resume_data.achievements, str):
            # If achievements is a string, split it
            achievements_list = [ach.strip() for ach in resume_data.achievements.split('\n') if ach.strip()]
            tailored_cv.achievements = self._rank_achievements(achievements_list, job_requirements)
        elif isinstance(resume_data.achievements, list):
            tailored_cv.achievements = self._rank_achievements(resume_data.achievements, job_requirements)
        else:
            tailored_cv.achievements = []
        
        # Languages (unchanged but potentially reordered)
        if resume_data.languages:
            # Handle case where languages might be a string instead of a list
            if isinstance(resume_data.languages, str):
                # If it's a string, try to split it or convert to list
                tailored_cv.languages = [lang.strip() for lang in resume_data.languages.split(',') if lang.strip()]
            elif isinstance(resume_data.languages, list):
                tailored_cv.languages = resume_data.languages.copy()
            else:
                tailored_cv.languages = []
        else:
            tailored_cv.languages = []
        
        # Calculate tailoring score
        tailored_cv.tailoring_score = self._calculate_tailoring_score(tailored_cv, job_requirements)
        
        # Generate optimization notes
        tailored_cv.optimization_notes = self._generate_optimization_notes(tailored_cv, job_requirements)
        
        return tailored_cv
    
    def _tailor_summary(self, resume_data: ResumeData, job_requirements: JobRequirements) -> str:
        """Generate a tailored professional summary."""
        try:
            # Prepare candidate profile
            candidate_profile = {
                'current_summary': resume_data.summary,
                'experience': [asdict(exp) for exp in resume_data.experience[:3]],  # Top 3 experiences
                'skills': resume_data.skills,
                'education': [asdict(edu) for edu in resume_data.education]
            }
            
            # Use Multi-LLM service with content generation task (Gemma 2B)
            tailored_summary = self.multi_llm_service.process_task(
                LLMTask.CONTENT_GENERATION,
                candidate_profile=candidate_profile,
                job_requirements=asdict(job_requirements),
                task_type="professional_summary"
            )
            
            return tailored_summary.strip()
            
        except Exception as e:
            logger.error(f"Error tailoring summary: {str(e)}")
            # Fallback to original summary
            return resume_data.summary or ""
    
    def _tailor_experience(self, experiences: List[Experience], job_requirements: JobRequirements) -> List[Dict[str, Any]]:
        """Tailor work experience entries."""
        tailored_experiences = []
        
        job_keywords = job_requirements.keywords + job_requirements.required_skills
        job_keywords_lower = [keyword.lower() for keyword in job_keywords]
        
        for exp in experiences:
            tailored_exp = asdict(exp)
            
            # Calculate relevance score for this experience
            relevance_score = self._calculate_experience_relevance(exp, job_requirements)
            tailored_exp['relevance_score'] = relevance_score
            
            # Tailor responsibilities using LLM
            if exp.responsibilities:
                try:
                    tailored_responsibilities = []
                    for responsibility in exp.responsibilities:
                        tailored_resp = self.multi_llm_service.process_task(
                            LLMTask.CONTENT_GENERATION,
                            candidate_profile={'experience': asdict(exp)},
                            job_requirements=asdict(job_requirements),
                            task_type="responsibility_tailoring",
                            content=responsibility
                        )
                        tailored_responsibilities.append(tailored_resp.strip())
                    
                    tailored_exp['responsibilities'] = tailored_responsibilities
                except Exception as e:
                    logger.error(f"Error tailoring responsibilities: {str(e)}")
            
            # Optimize achievements for ATS
            if exp.achievements:
                tailored_exp['achievements'] = self._optimize_achievements(
                    exp.achievements, job_keywords
                )
            
            # Highlight relevant technologies
            if exp.technologies:
                relevant_techs = []
                for tech in exp.technologies:
                    # Handle both string and dict technologies
                    if isinstance(tech, dict):
                        tech_text = tech.get('name', '') or tech.get('technology', '') or str(tech)
                    else:
                        tech_text = str(tech)
                    
                    if tech_text.lower() in job_keywords_lower:
                        relevant_techs.insert(0, tech)  # Put relevant tech first
                    else:
                        relevant_techs.append(tech)
                tailored_exp['technologies'] = relevant_techs[:10]  # Limit to top 10
            
            tailored_experiences.append(tailored_exp)
        
        # Sort experiences by relevance score
        tailored_experiences.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return tailored_experiences
    
    def _tailor_skills(self, skills: Dict[str, List[str]], job_requirements: JobRequirements) -> Dict[str, List[str]]:
        """Tailor skills section to emphasize job-relevant skills."""
        tailored_skills = {"technical": [], "soft": [], "tools": [], "languages": []}
        
        # Handle case where skills might be None or not a dict
        if not skills or not isinstance(skills, dict):
            return tailored_skills
        
        job_skills = (job_requirements.required_skills + 
                     job_requirements.preferred_skills + 
                     job_requirements.keywords)
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        # Process each skill category
        for category, skill_list in skills.items():
            if category not in tailored_skills:
                tailored_skills[category] = []
            
            # Ensure skill_list is a list
            if not isinstance(skill_list, list):
                continue
            
            # Separate relevant and other skills
            relevant_skills = []
            other_skills = []
            
            for skill in skill_list:
                if skill.lower() in job_skills_lower:
                    relevant_skills.append(skill)
                else:
                    other_skills.append(skill)
            
            # Put relevant skills first
            tailored_skills[category] = relevant_skills + other_skills[:15]  # Limit total skills
        
        # Add missing critical skills if candidate has related experience
        self._suggest_missing_skills(tailored_skills, job_requirements)
        
        return tailored_skills
    
    def _tailor_education(self, educations: List[Education], job_requirements: JobRequirements) -> List[Dict[str, Any]]:
        """Tailor education section."""
        tailored_educations = []
        
        for edu in educations:
            tailored_edu = asdict(edu)
            
            # Calculate relevance based on job requirements
            relevance_score = self._calculate_education_relevance(edu, job_requirements)
            tailored_edu['relevance_score'] = relevance_score
            
            # Emphasize relevant coursework
            if edu.relevant_coursework:
                job_keywords = job_requirements.keywords + job_requirements.required_skills
                job_keywords_lower = [keyword.lower() for keyword in job_keywords]
                
                relevant_courses = []
                other_courses = []
                
                for course in edu.relevant_coursework:
                    if any(keyword in course.lower() for keyword in job_keywords_lower):
                        relevant_courses.append(course)
                    else:
                        other_courses.append(course)
                
                tailored_edu['relevant_coursework'] = relevant_courses + other_courses[:8]
            
            tailored_educations.append(tailored_edu)
        
        # Sort by relevance
        tailored_educations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return tailored_educations
    
    def _tailor_projects(self, projects: List[Project], job_requirements: JobRequirements) -> List[Dict[str, Any]]:
        """Tailor projects section."""
        tailored_projects = []
        
        job_keywords = job_requirements.keywords + job_requirements.required_skills
        job_keywords_lower = [keyword.lower() for keyword in job_keywords]
        
        for project in projects:
            tailored_project = asdict(project)
            
            # Calculate project relevance
            relevance_score = self._calculate_project_relevance(project, job_requirements)
            tailored_project['relevance_score'] = relevance_score
            
            # Tailor project description using Multi-LLM service
            if project.description:
                try:
                    tailored_description = self.multi_llm_service.process_task(
                        LLMTask.CONTENT_GENERATION,
                        candidate_profile={'project': asdict(project)},
                        job_requirements=asdict(job_requirements),
                        task_type="project_description_tailoring",
                        content=project.description
                    )
                    tailored_project['description'] = tailored_description.strip()
                except Exception as e:
                    logger.error(f"Error tailoring project description: {str(e)}")
            
            # Emphasize relevant technologies
            if project.technologies:
                relevant_techs = []
                other_techs = []
                
                for tech in project.technologies:
                    # Handle both string and dict technologies
                    if isinstance(tech, dict):
                        tech_text = tech.get('name', '') or tech.get('technology', '') or str(tech)
                    else:
                        tech_text = str(tech)
                    
                    if tech_text.lower() in job_keywords_lower:
                        relevant_techs.append(tech)
                    else:
                        other_techs.append(tech)
                
                tailored_project['technologies'] = relevant_techs + other_techs
            
            tailored_projects.append(tailored_project)
        
        # Sort by relevance and limit to top 5 most relevant
        tailored_projects.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return tailored_projects[:5]
    
    def _rank_certifications(self, certifications: List[Any], job_requirements: JobRequirements) -> List[Dict[str, Any]]:
        """Rank certifications by relevance."""
        if not certifications:
            return []
        
        ranked_certs = []
        job_keywords_lower = [keyword.lower() for keyword in 
                             job_requirements.keywords + job_requirements.required_skills]
        
        for cert in certifications:
            cert_dict = asdict(cert) if hasattr(cert, '__dict__') else cert
            
            # Calculate relevance score
            relevance_score = 0
            cert_text = f"{cert_dict.get('name', '')} {cert_dict.get('issuer', '')}".lower()
            
            for keyword in job_keywords_lower:
                if keyword in cert_text:
                    relevance_score += 1
            
            cert_dict['relevance_score'] = relevance_score
            ranked_certs.append(cert_dict)
        
        # Sort by relevance
        ranked_certs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return ranked_certs
    
    def _rank_achievements(self, achievements: List[str], job_requirements: JobRequirements) -> List[str]:
        """Rank achievements by relevance."""
        if not achievements:
            return []
        
        job_keywords_lower = [keyword.lower() for keyword in 
                             job_requirements.keywords + job_requirements.required_skills]
        
        # Score achievements
        scored_achievements = []
        for achievement in achievements:
            score = 0
            # Handle both string and dict achievements
            if isinstance(achievement, dict):
                achievement_text = achievement.get('text', '') or achievement.get('description', '') or str(achievement)
            else:
                achievement_text = str(achievement)
            
            achievement_lower = achievement_text.lower()
            
            for keyword in job_keywords_lower:
                if keyword in achievement_lower:
                    score += 1
            
            # Bonus for quantified achievements
            if any(char.isdigit() for char in achievement):
                score += 1
            
            scored_achievements.append((score, achievement))
        
        # Sort by score and return achievement text
        scored_achievements.sort(key=lambda x: x[0], reverse=True)
        
        return [achievement for score, achievement in scored_achievements]
    
    def _calculate_experience_relevance(self, experience: Experience, job_requirements: JobRequirements) -> float:
        """Calculate relevance score for an experience entry."""
        score = 0
        
        job_keywords = (job_requirements.keywords + 
                       job_requirements.required_skills + 
                       job_requirements.responsibilities)
        job_keywords_lower = [keyword.lower() for keyword in job_keywords]
        
        # Check position title
        if experience.position:
            position_lower = experience.position.lower()
            for keyword in job_keywords_lower:
                if keyword in position_lower:
                    score += 2
        
        # Check responsibilities
        for responsibility in experience.responsibilities:
            # Handle both string and dict responsibilities
            if isinstance(responsibility, dict):
                resp_text = responsibility.get('text', '') or responsibility.get('description', '') or str(responsibility)
            else:
                resp_text = str(responsibility)
            
            resp_lower = resp_text.lower()
            for keyword in job_keywords_lower:
                if keyword in resp_lower:
                    score += 1
        
        # Check technologies
        for tech in experience.technologies:
            # Handle both string and dict technologies
            if isinstance(tech, dict):
                tech_text = tech.get('name', '') or tech.get('technology', '') or str(tech)
            else:
                tech_text = str(tech)
            
            if tech_text.lower() in job_keywords_lower:
                score += 1.5
        
        return score
    
    def _calculate_education_relevance(self, education: Education, job_requirements: JobRequirements) -> float:
        """Calculate relevance score for an education entry."""
        score = 0
        
        job_keywords_lower = [keyword.lower() for keyword in 
                             job_requirements.keywords + job_requirements.required_skills]
        
        # Check degree
        if education.degree:
            degree_lower = education.degree.lower()
            for keyword in job_keywords_lower:
                if keyword in degree_lower:
                    score += 1
        
        # Check relevant coursework
        for course in education.relevant_coursework:
            # Handle both string and dict courses
            if isinstance(course, dict):
                course_text = course.get('name', '') or course.get('title', '') or str(course)
            else:
                course_text = str(course)
            
            course_lower = course_text.lower()
            for keyword in job_keywords_lower:
                if keyword in course_lower:
                    score += 0.5
        
        return score
    
    def _calculate_project_relevance(self, project: Project, job_requirements: JobRequirements) -> float:
        """Calculate relevance score for a project."""
        score = 0
        
        job_keywords_lower = [keyword.lower() for keyword in 
                             job_requirements.keywords + job_requirements.required_skills]
        
        # Check project name and description
        project_name = project.name or ""
        project_desc = project.description or ""
        project_text = f"{project_name} {project_desc}".lower()
        for keyword in job_keywords_lower:
            if keyword in project_text:
                score += 1
        
        # Check technologies
        for tech in project.technologies:
            # Handle both string and dict technologies
            if isinstance(tech, dict):
                tech_text = tech.get('name', '') or tech.get('technology', '') or str(tech)
            else:
                tech_text = str(tech)
            
            if tech_text.lower() in job_keywords_lower:
                score += 1.5
        
        return score
    
    def _optimize_achievements(self, achievements: List[str], job_keywords: List[str]) -> List[str]:
        """Optimize achievements for ATS by incorporating keywords."""
        optimized = []
        
        for achievement in achievements:
            # Use Multi-LLM service with analysis task (Llama2 7B) for ATS optimization
            try:
                optimized_achievement = self.multi_llm_service.process_task(
                    LLMTask.DATA_ANALYSIS,
                    content=achievement,
                    keywords=job_keywords,
                    task_type="ats_optimization"
                )
                optimized.append(optimized_achievement.strip())
            except Exception as e:
                logger.error(f"Error optimizing achievement: {str(e)}")
                optimized.append(achievement)
        
        return optimized
    
    def _suggest_missing_skills(self, tailored_skills: Dict[str, List[str]], job_requirements: JobRequirements):
        """Suggest missing skills that might be implied by experience."""
        # This is a placeholder for more sophisticated skill inference
        # In practice, you could use the LLM to infer skills from experience descriptions
        pass
    
    def _calculate_tailoring_score(self, tailored_cv: TailoredCV, job_requirements: JobRequirements) -> float:
        """Calculate overall tailoring score."""
        total_score = 0
        max_score = 0
        
        # Score based on keyword matches
        job_keywords = set(keyword.lower() for keyword in 
                          job_requirements.keywords + job_requirements.required_skills)
        
        cv_text = f"{tailored_cv.summary} ".lower()
        
        # Add experience text
        for exp in tailored_cv.experience:
            cv_text += f" {' '.join(exp.get('responsibilities', []))}"
        
        # Add skills
        all_skills = []
        for skill_list in tailored_cv.skills.values():
            all_skills.extend(skill_list)
        cv_text += f" {' '.join(all_skills)}".lower()
        
        # Calculate matches
        matches = 0
        for keyword in job_keywords:
            if keyword in cv_text:
                matches += 1
        
        # Calculate score as percentage
        if job_keywords:
            score = (matches / len(job_keywords)) * 100
        else:
            score = 100
        
        return min(score, 100)  # Cap at 100%
    
    def _generate_optimization_notes(self, tailored_cv: TailoredCV, job_requirements: JobRequirements) -> List[str]:
        """Generate optimization notes and suggestions."""
        notes = []
        
        # Check for missing required skills
        required_skills_lower = [skill.lower() for skill in job_requirements.required_skills]
        cv_skills_lower = []
        
        for skill_list in tailored_cv.skills.values():
            cv_skills_lower.extend([skill.lower() for skill in skill_list])
        
        missing_skills = [skill for skill in required_skills_lower 
                         if skill not in cv_skills_lower]
        
        if missing_skills:
            notes.append(f"Consider adding these required skills if you have them: {', '.join(missing_skills[:5])}")
        
        # Check summary length
        if len(tailored_cv.summary.split()) < 30:
            notes.append("Professional summary could be more detailed (aim for 3-4 sentences)")
        
        # Check for quantified achievements
        has_numbers = False
        for exp in tailored_cv.experience:
            exp_text = ' '.join(exp.get('responsibilities', []) + exp.get('achievements', []))
            if any(char.isdigit() for char in exp_text):
                has_numbers = True
                break
        
        if not has_numbers:
            notes.append("Consider adding quantifiable achievements (numbers, percentages, dollar amounts)")
        
        # Check for action verbs
        summary_has_action_verbs = any(verb in tailored_cv.summary.lower() 
                                      for verb in self.action_verbs[:10])
        if not summary_has_action_verbs:
            notes.append("Use more action verbs in your professional summary")
        
        return notes
    
    def get_ats_optimization_score(self, tailored_cv: TailoredCV, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Get detailed ATS optimization analysis."""
        analysis = {
            'overall_score': tailored_cv.tailoring_score,
            'keyword_density': self._calculate_keyword_density(tailored_cv, job_requirements),
            'format_score': self._calculate_format_score(tailored_cv),
            'content_score': self._calculate_content_score(tailored_cv),
            'recommendations': tailored_cv.optimization_notes
        }
        
        return analysis
    
    def _calculate_keyword_density(self, tailored_cv: TailoredCV, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Calculate keyword density for ATS optimization."""
        job_keywords = job_requirements.keywords + job_requirements.required_skills
        
        # Get all text from CV
        cv_text = tailored_cv.summary
        for exp in tailored_cv.experience:
            cv_text += ' ' + ' '.join(exp.get('responsibilities', []))
        
        total_words = len(cv_text.split())
        keyword_count = 0
        
        for keyword in job_keywords:
            keyword_count += cv_text.lower().count(keyword.lower())
        
        density = (keyword_count / total_words) * 100 if total_words > 0 else 0
        
        return {
            'density_percentage': round(density, 2),
            'total_keywords': len(job_keywords),
            'keywords_used': keyword_count,
            'optimal_range': '2-5%',
            'status': 'optimal' if 2 <= density <= 5 else 'needs_improvement'
        }
    
    def _calculate_format_score(self, tailored_cv: TailoredCV) -> float:
        """Calculate ATS format compatibility score."""
        score = 100  # Start with perfect score
        
        # Deduct for potential formatting issues
        # This is a simplified version - in practice, you'd analyze actual document formatting
        
        if not tailored_cv.personal_info.get('email'):
            score -= 10
        
        if not tailored_cv.summary:
            score -= 15
        
        if len(tailored_cv.experience) == 0:
            score -= 20
        
        return max(score, 0)
    
    def _calculate_content_score(self, tailored_cv: TailoredCV) -> float:
        """Calculate content quality score."""
        score = 0
        
        # Professional summary quality
        if tailored_cv.summary:
            summary_words = len(tailored_cv.summary.split())
            if 20 <= summary_words <= 60:
                score += 15
            else:
                score += 5
        
        # Experience section quality
        if tailored_cv.experience:
            score += 20
            
            # Check for quantified achievements
            has_numbers = any(
                any(char.isdigit() for char in ' '.join(exp.get('responsibilities', [])))
                for exp in tailored_cv.experience
            )
            if has_numbers:
                score += 15
        
        # Skills section
        total_skills = sum(len(skills) for skills in tailored_cv.skills.values())
        if 10 <= total_skills <= 25:
            score += 20
        elif total_skills > 0:
            score += 10
        
        # Projects section
        if tailored_cv.projects:
            score += 15
        
        # Education section
        if tailored_cv.education:
            score += 15
        
        return min(score, 100)  # Cap at 100