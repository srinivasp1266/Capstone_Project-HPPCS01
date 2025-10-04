"""
LLM Client module for CV Builder.
Handles communication with Ollama and other LLM providers.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
import requests
import time

try:
    from langchain_community.llms import Ollama
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    Ollama = None
    BaseMessage = None
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM server."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma:2b", timeout: int = 120):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds (reduced from 300 to 120)
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        logger.info(f"Initializing OllamaClient with model {model} and timeout {timeout}s")
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """Validate connection to Ollama server."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if the model is available
            models = response.json().get("models", [])
            available_models = [model.get("name", "") for model in models]
            
            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                # Try to pull the model
                self._pull_model()
            
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.host}: {str(e)}")
            return False
    
    def _pull_model(self) -> bool:
        """Pull the specified model if not available."""
        try:
            logger.info(f"Pulling model {self.model}...")
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes for model download
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to pull model {self.model}: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """
        Generate response from the LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated response text
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 2048),
                }
            }
            
            if system_message:
                payload["system"] = system_message
            
            # Make the request
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.RequestException as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response JSON: {str(e)}")
            raise
    
    def generate_json_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM with improved error handling.
        
        Args:
            prompt: Prompt for JSON generation
            
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON formatting instructions to the prompt
        enhanced_prompt = f"""
        {prompt}
        
        CRITICAL JSON FORMATTING REQUIREMENTS:
        1. Return ONLY valid JSON - no additional text, explanations, or comments
        2. Use double quotes for all keys and string values
        3. For nested objects, use proper JSON syntax with colons and braces
        4. For arrays, use square brackets with comma-separated values
        5. Ensure all brackets and braces are properly closed
        6. NO trailing commas after the last item in objects or arrays
        7. Do not include any text before {{ or after }}
        8. Start your response immediately with {{ and end with }}
        
        Example of correct format:
        {{
            "key": "value",
            "array": ["item1", "item2"],
            "object": {{
                "nested_key": "nested_value"
            }}
        }}
        
        REMEMBER: NO TRAILING COMMAS - this is critical for JSON validity!
        """
        
        try:
            response = self.generate_response(enhanced_prompt)
            logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}...")
            
            parsed_response = self._parse_json_response(response)
            logger.debug(f"Successfully parsed JSON with keys: {list(parsed_response.keys()) if parsed_response else 'None'}")
            return parsed_response
        except Exception as e:
            logger.error(f"Error generating JSON response: {str(e)}")
            logger.debug(f"Prompt that caused error: {enhanced_prompt[:200]}...")
            return self._get_fallback_response(prompt)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with enhanced error handling."""
        if not response:
            logger.warning("Empty response received")
            return {}
            
        # Clean up the response first
        json_str = response.strip()
        
        # Remove common prefixes that might interfere with JSON
        prefixes_to_remove = [
            "Here is the JSON response:",
            "Here's the JSON:",
            "The JSON response is:",
            "JSON:",
            "Response:",
            "Here is the extracted information:",
            "Based on the resume, here is the JSON:"
        ]
        
        for prefix in prefixes_to_remove:
            if json_str.startswith(prefix):
                json_str = json_str[len(prefix):].strip()
        
        # Find the first '{' or '[' and the last matching '}' or ']'
        start_brace = json_str.find('{')
        start_bracket = json_str.find('[')
        
        # Determine which comes first
        if start_brace == -1 and start_bracket == -1:
            logger.warning("No JSON structure found in response")
            logger.debug(f"Response content: {response[:200]}...")
            return self._extract_json_manually(response)
            
        if start_brace == -1:
            start_idx = start_bracket
            start_char = '['
            end_char = ']'
        elif start_bracket == -1:
            start_idx = start_brace
            start_char = '{'
            end_char = '}'
        else:
            if start_brace < start_bracket:
                start_idx = start_brace
                start_char = '{'
                end_char = '}'
            else:
                start_idx = start_bracket
                start_char = '['
                end_char = ']'
        
        # Extract JSON portion
        json_str = json_str[start_idx:]
        
        # Find the matching closing character with proper depth tracking
        depth = 0
        end_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
        
        if end_idx == -1:
            logger.warning("No matching closing brace/bracket found")
            # Try to find the last occurrence as fallback
            end_idx = max(json_str.rfind('}'), json_str.rfind(']'))
            if end_idx > 0:
                json_str = json_str[:end_idx + 1]
            else:
                logger.error("Could not determine JSON boundaries")
                return self._extract_json_manually(response)
        else:
            json_str = json_str[:end_idx + 1]
        
        # Try parsing the JSON
        try:
            parsed = json.loads(json_str)
            logger.debug("Successfully parsed JSON on first attempt")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Initial JSON parse failed: {str(e)}")
            logger.debug(f"Failed JSON string (first 500 chars): {json_str[:500]}...")
            logger.debug(f"Error at position {e.pos}: '{json_str[max(0, e.pos-20):e.pos+20]}'")
            
            # Try to fix common JSON issues
            fixed_json = self._fix_common_json_issues(json_str)
            try:
                parsed = json.loads(fixed_json)
                logger.debug("Successfully parsed JSON after fixing issues")
                return parsed
            except json.JSONDecodeError as e2:
                logger.error(f"Fixed JSON parse failed: {str(e2)}")
                logger.debug(f"Fixed JSON string (first 500 chars): {fixed_json[:500]}...")
                logger.debug(f"Error at position {e2.pos}: '{fixed_json[max(0, e2.pos-20):e2.pos+20]}'")
                
                # Try one more time with aggressive cleaning
                try:
                    aggressive_clean = self._aggressive_json_clean(json_str)
                    parsed = json.loads(aggressive_clean)
                    logger.debug("Successfully parsed JSON after aggressive cleaning")
                    return parsed
                except json.JSONDecodeError as e3:
                    logger.error(f"Aggressive clean JSON parse failed: {str(e3)}")
                    logger.debug(f"Aggressive clean JSON (first 500 chars): {aggressive_clean[:500]}...")
                    logger.debug(f"Error at position {e3.pos}: '{aggressive_clean[max(0, e3.pos-20):e3.pos+20]}'")
                    
                    # Final fallback - try to extract key information manually
                    return self._extract_json_manually(response)
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        import re
        
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        
        # Remove markdown code block markers if present
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'^```\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        
        # Remove any text before the first '{' or '['
        first_brace = json_str.find('{')
        first_bracket = json_str.find('[')
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            json_str = json_str[first_brace:]
        elif first_bracket != -1:
            json_str = json_str[first_bracket:]
        
        # Remove any text after the last '}' or ']'
        last_brace = json_str.rfind('}')
        last_bracket = json_str.rfind(']')
        if last_brace != -1 and last_brace > last_bracket:
            json_str = json_str[:last_brace + 1]
        elif last_bracket != -1:
            json_str = json_str[:last_bracket + 1]
        
        # Fix unquoted keys (word: -> "word":)
        json_str = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_str)
        
        # Fix trailing commas before closing braces/brackets (most important for your error)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix multiple consecutive commas
        json_str = re.sub(r',\s*,+', ',', json_str)
        
        # Fix missing commas between objects/arrays/values
        # Between } and {
        json_str = re.sub(r'}\s*(?=\s*{)', r'},', json_str)
        # Between ] and [
        json_str = re.sub(r']\s*(?=\s*\[)', r'],', json_str)
        # Between } and [
        json_str = re.sub(r'}\s*(?=\s*\[)', r'},', json_str)
        # Between ] and {
        json_str = re.sub(r']\s*(?=\s*{)', r'],', json_str)
        # Between quoted strings
        json_str = re.sub(r'"\s*(?=\s*"[^:]*":)', r'",', json_str)
        # Between value and next key
        json_str = re.sub(r'(["\d}\]])\s*(?=\s*"[^:]*":)', r'\1,', json_str)
        
        # Fix single quotes to double quotes (but be careful with apostrophes in content)
        json_str = re.sub(r"'([^']*)'(\s*[,:\]}])", r'"\1"\2', json_str)
        
        # Fix line breaks within strings
        lines = json_str.split('\n')
        fixed_lines = []
        in_string = False
        current_line = ""
        
        for line in lines:
            if not in_string:
                # Count unescaped quotes
                quote_count = len(re.findall(r'(?<!\\)"', line))
                if quote_count % 2 == 1:
                    in_string = True
                    current_line = line
                else:
                    fixed_lines.append(line)
            else:
                # We're in a multiline string
                current_line += " " + line.strip()
                quote_count = len(re.findall(r'(?<!\\)"', line))
                if quote_count % 2 == 1:
                    in_string = False
                    fixed_lines.append(current_line)
                    current_line = ""
        
        # If we ended in a string, add the remaining line
        if current_line:
            fixed_lines.append(current_line)
        
        json_str = '\n'.join(fixed_lines)
        
        # Fix specific comma delimiter issues
        # Remove comma before colon
        json_str = re.sub(r',\s*:', ':', json_str)
        # Remove comma at start of object/array
        json_str = re.sub(r'([{\[])\s*,', r'\1', json_str)
        # Ensure comma after values (except before } or ])
        json_str = re.sub(r'(["\d}\]])\s*(?=\s*"[^:}]*":)', r'\1,', json_str)
        
        return json_str
    
    def _reconstruct_json_from_parts(self, json_str: str) -> str:
        """Reconstruct valid JSON by parsing individual components."""
        import re
        
        # Find all key-value pairs
        result_dict = {}
        
        # Extract simple key-value pairs
        kv_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
        matches = re.findall(kv_pattern, json_str)
        for key, value in matches:
            result_dict[key] = value
        
        # Extract array patterns
        array_pattern = r'"([^"]+)"\s*:\s*\[(.*?)\]'
        array_matches = re.findall(array_pattern, json_str, re.DOTALL)
        for key, array_content in array_matches:
            if array_content.strip():
                # Split by comma and clean up
                items = [item.strip().strip('"').strip("'") for item in array_content.split(',')]
                result_dict[key] = [item for item in items if item]
            else:
                result_dict[key] = []
        
        # Extract object patterns
        obj_pattern = r'"([^"]+)"\s*:\s*\{(.*?)\}'
        obj_matches = re.findall(obj_pattern, json_str, re.DOTALL)
        for key, obj_content in obj_matches:
            nested_dict = {}
            nested_kv_matches = re.findall(kv_pattern, obj_content)
            for nested_key, nested_value in nested_kv_matches:
                nested_dict[nested_key] = nested_value
            if nested_dict:
                result_dict[key] = nested_dict
        
        # Convert back to JSON
        return json.dumps(result_dict, ensure_ascii=False)
    
    def _aggressive_json_clean(self, json_str: str) -> str:
        """Aggressive JSON cleaning as last resort before manual extraction."""
        import re
        
        # Start with basic fixes
        json_str = self._fix_common_json_issues(json_str)
        
        # Try reconstruction approach first
        try:
            reconstructed = self._reconstruct_json_from_parts(json_str)
            # Test if it's valid
            json.loads(reconstructed)
            return reconstructed
        except:
            pass  # Continue with aggressive cleaning
        
        # More aggressive fixes
        # Remove any remaining non-JSON content at start and end
        json_str = re.sub(r'^[^{\[]*', '', json_str)
        json_str = re.sub(r'[^}\]]*$', '', json_str)
        
        # Fix common malformed patterns
        # Fix arrays that aren't properly formed
        json_str = re.sub(r':\s*([^",\[\{][^,\]\}]*(?:,\s*[^",\[\{][^,\]\}]*)*)\s*([,}])', r': ["\1"]\2', json_str)
        
        # Fix missing quotes around string values
        json_str = re.sub(r':\s*([^",\[\{][^,\]\}]*)\s*([,}])', r': "\1"\2', json_str)
        
        # Fix boolean and null values that got quoted
        json_str = json_str.replace('"true"', 'true')
        json_str = json_str.replace('"false"', 'false')
        json_str = json_str.replace('"null"', 'null')
        
        # Fix numbers that got quoted but shouldn't be
        json_str = re.sub(r'"\s*(\d+(?:\.\d+)?)\s*"', r'\1', json_str)
        
        # Remove trailing commas more aggressively
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        json_str = re.sub(r',\s*,+', ',', json_str)
        
        # Fix spacing issues
        json_str = re.sub(r'\s*:\s*', ':', json_str)
        json_str = re.sub(r'\s*,\s*', ',', json_str)
        
        return json_str
    
    def _extract_json_manually(self, response: str) -> Dict[str, Any]:
        """Extract information manually when JSON parsing completely fails."""
        logger.warning("Falling back to manual JSON extraction")
        
        result = {
            "personal_info": {"name": "Unknown", "email": "", "phone": "", "address": "", "linkedin": ""},
            "summary": "",
            "experience": [],
            "education": [],
            "skills": {"technical": [], "soft": [], "tools": [], "languages": []},
            "projects": [],
            "certifications": [],
            "achievements": [],
            "languages": []
        }
        
        # Try to extract key information using regex patterns
        patterns = {
            "name": [
                r'"?name"?\s*:\s*"([^"]+)"',
                r'"?full_name"?\s*:\s*"([^"]+)"',
                r'Name\s*:\s*([^\n,]+)',
                r'name\s*=\s*"([^"]+)"'
            ],
            "email": [
                r'"?email"?\s*:\s*"([^"]+)"',
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'Email\s*:\s*([^\n,]+)'
            ],
            "phone": [
                r'"?phone"?\s*:\s*"([^"]+)"',
                r'Phone\s*:\s*([^\n,]+)',
                r'(\+?[\d\s\-\(\)]+)'
            ],
            "summary": [
                r'"?summary"?\s*:\s*"([^"]+)"',
                r'"?professional_summary"?\s*:\s*"([^"]+)"',
                r'Summary\s*:\s*([^\n]+)'
            ]
        }
        
        # Extract using patterns
        for key, regex_list in patterns.items():
            for pattern in regex_list:
                matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
                if matches:
                    value = matches[0].strip()
                    if key == "name" or key == "email" or key == "phone" or key == "summary":
                        if key in ["name", "email", "phone"]:
                            if "personal_info" not in result:
                                result["personal_info"] = {}
                            result["personal_info"][key] = value
                        else:
                            result[key] = value
                    break
        
        # Extract experience section
        exp_patterns = [
            r'"?experience"?\s*:\s*\[(.*?)\]',
            r'"?work_experience"?\s*:\s*\[(.*?)\]'
        ]
        
        for pattern in exp_patterns:
            exp_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if exp_match:
                exp_text = exp_match.group(1)
                # Try to extract individual experience entries
                experiences = []
                exp_entries = re.split(r'},{?', exp_text)
                for entry in exp_entries:
                    exp_obj = {}
                    company_match = re.search(r'"?company"?\s*:\s*"([^"]+)"', entry, re.IGNORECASE)
                    position_match = re.search(r'"?(?:position|title|job_title)"?\s*:\s*"([^"]+)"', entry, re.IGNORECASE)
                    
                    if company_match:
                        exp_obj["company"] = company_match.group(1)
                    if position_match:
                        exp_obj["position"] = position_match.group(1)
                    
                    if exp_obj:
                        exp_obj.setdefault("responsibilities", [])
                        exp_obj.setdefault("start_date", "")
                        exp_obj.setdefault("end_date", "")
                        exp_obj.setdefault("location", "")
                        experiences.append(exp_obj)
                
                if experiences:
                    result["experience"] = experiences
                break
        
        # Extract skills
        skills_patterns = [
            r'"?(?:technical_)?skills"?\s*:\s*\[(.*?)\]',
            r'"?skills"?\s*:\s*{(.*?)}',
        ]
        
        for pattern in skills_patterns:
            skills_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if skills_match:
                skills_text = skills_match.group(1)
                if '{' in skills_text:  # Object format
                    tech_match = re.search(r'"?technical"?\s*:\s*\[(.*?)\]', skills_text, re.IGNORECASE)
                    if tech_match:
                        tech_skills = [s.strip().strip('"') for s in tech_match.group(1).split(',')]
                        result["skills"]["technical"] = [s for s in tech_skills if s]
                else:  # Array format
                    skills_list = [s.strip().strip('"') for s in skills_text.split(',')]
                    result["skills"]["technical"] = [s for s in skills_list if s]
                break
        
        # Clean up empty nested structures
        if result.get("personal_info"):
            result["personal_info"] = {k: v for k, v in result["personal_info"].items() if v}
        
        if result.get("skills"):
            result["skills"] = {k: v for k, v in result["skills"].items() if v}
            if not result["skills"]:
                result["skills"] = {"technical": [], "soft": [], "tools": [], "languages": []}
        
        logger.debug(f"Manually extracted: {result}")
        return result
    
    def _get_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Provide structured fallback when all parsing fails."""
        logger.error("All JSON parsing methods failed, using fallback")
        
        # Determine response type from prompt
        if "resume" in prompt.lower():
            return {
                "name": "Unable to parse name",
                "email": "",
                "phone": "",
                "address": "",
                "linkedin": "",
                "summary": "Unable to parse professional summary",
                "education": [],
                "experience": [{
                    "company": "Unable to parse company",
                    "position": "Unable to parse position", 
                    "location": "",
                    "start_date": "",
                    "end_date": "",
                    "responsibilities": ["Unable to parse responsibilities"],
                    "achievements": [],
                    "technologies": []
                }],
                "skills": {"technical": [], "soft": [], "tools": [], "languages": []},
                "projects": [],
                "certifications": [],
                "achievements": [],
                "languages": [],
                "error": "JSON parsing failed - please try again or check resume format"
            }
        elif "job" in prompt.lower():
            return {
                "title": "Unable to parse job title",
                "company": "",
                "location": "",
                "employment_type": "",
                "experience_level": "",
                "required_skills": [],
                "preferred_skills": [],
                "soft_skills": [],
                "responsibilities": ["Unable to parse responsibilities"],
                "qualifications": [],
                "keywords": [],
                "benefits": [],
                "company_description": "",
                "error": "JSON parsing failed - please try again or check job description format"
            }
        else:
            return {"error": "JSON parsing failed", "raw_response": prompt[:500]}
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat completion response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            system_message = ""
            
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    system_message = content
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = "\n".join(prompt_parts)
            
            return self.generate_response(prompt, system_message, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise


class LLMService:
    """High-level service for LLM operations."""
    
    def __init__(self, client: OllamaClient):
        """
        Initialize LLM service.
        
        Args:
            client: OllamaClient instance
        """
        self.client = client
    
    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured data from resume text using smart multi-stage approach.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Comprehensive structured resume data
        """
        try:
            # First, try comprehensive extraction with reasonable text length
            if len(resume_text) > 12000:
                # For very long resumes, extract in sections but maintain quality
                return self._extract_resume_data_sectioned(resume_text)
            else:
                # For normal length resumes, use full comprehensive extraction
                return self._extract_resume_data_comprehensive(resume_text)
        except Exception as e:
            logger.error(f"Comprehensive extraction failed: {e}")
            # Fallback to progressive extraction
            return self._extract_resume_data_progressive(resume_text)
    
    def _extract_resume_data_comprehensive(self, resume_text: str) -> Dict[str, Any]:
        """Comprehensive resume extraction with full LLM power."""
        prompt = f"""
        You are an expert resume parser. Extract comprehensive information from the resume and return it in valid JSON format.

        CRITICAL: Return ONLY valid JSON with proper syntax. No additional text, explanations, or comments.

        Required JSON structure:
        {{
            "name": "Full name of the candidate",
            "email": "Email address", 
            "phone": "Phone number",
            "address": "Complete address",
            "linkedin": "LinkedIn profile URL",
            "website": "Personal website URL",
            "github": "GitHub profile URL",
            "summary": "Professional summary or objective statement",
            
            "experience": [
                {{
                    "company": "Company name",
                    "position": "Job title/position", 
                    "location": "Job location",
                    "start_date": "Start date",
                    "end_date": "End date or Present",
                    "responsibilities": ["Detailed responsibility 1", "Detailed responsibility 2"],
                    "achievements": ["Quantified achievement 1", "Achievement 2"],
                    "technologies": ["Tech1", "Tech2"]
                }}
            ],
            
            "education": [
                {{
                    "degree": "Full degree name",
                    "institution": "University/college name", 
                    "location": "Institution location",
                    "start_date": "Start date",
                    "end_date": "Graduation date/year",
                    "gpa": "GPA if mentioned",
                    "relevant_coursework": ["Course 1", "Course 2"],
                    "honors": ["Honors/awards"]
                }}
            ],
            
            "skills": {{
                "technical": ["Programming languages", "Frameworks", "Databases"],
                "soft": ["Leadership", "Communication"], 
                "tools": ["Software tools", "Development tools"],
                "languages": ["English", "Other languages"]
            }},
            
            "projects": [
                {{
                    "name": "Project name",
                    "description": "Detailed project description",
                    "technologies": ["Tech used"],
                    "start_date": "Project start date",
                    "end_date": "Project end date",
                    "url": "Live project URL",
                    "github_url": "GitHub repository URL",
                    "key_features": ["Feature 1", "Feature 2"]
                }}
            ],
            
            "certifications": [
                {{
                    "name": "Certification name",
                    "issuer": "Issuing organization", 
                    "date": "Date obtained",
                    "credential_id": "ID if available",
                    "url": "Verification URL"
                }}
            ],
            
            "achievements": ["Professional achievement 1", "Award 2"],
            "languages": ["English (Fluent)", "Other languages"]
        }}

        EXTRACTION GUIDELINES:
        1. Extract ALL information present - preserve completeness
        2. For missing information, use empty strings or empty arrays
        3. Preserve exact dates and quantified achievements
        4. Include all technical skills and technologies mentioned
        5. Maintain professional language and original meaning

        Resume text:
        {resume_text}

        Return ONLY the JSON object:
        """
        
        return self.client.generate_json_response(prompt)
    
    def _extract_resume_data_sectioned(self, resume_text: str) -> Dict[str, Any]:
        """Extract from very long resumes by processing key sections."""
        # Split resume into logical sections
        sections = self._split_resume_into_sections(resume_text)
        
        result = {
            "name": "", "email": "", "phone": "", "address": "", "linkedin": "", 
            "website": "", "github": "", "summary": "", "experience": [], 
            "education": [], "skills": {}, "projects": [], "certifications": [], 
            "achievements": [], "languages": []
        }
        
        # Extract basic info from header section
        if sections.get('header'):
            basic_info = self._extract_basic_info(sections['header'])
            result.update(basic_info)
        
        # Extract experience section
        if sections.get('experience'):
            exp_data = self._extract_experience_section(sections['experience'])
            result['experience'] = exp_data
        
        # Extract education section  
        if sections.get('education'):
            edu_data = self._extract_education_section(sections['education'])
            result['education'] = edu_data
            
        # Extract skills section
        if sections.get('skills'):
            skills_data = self._extract_skills_section(sections['skills'])
            result['skills'] = skills_data
            
        return result
    
    def _extract_resume_data_progressive(self, resume_text: str) -> Dict[str, Any]:
        """Progressive extraction when full extraction fails."""
        logger.info("Using progressive extraction as fallback")
        
        # Start with basic extraction
        result = self._extract_resume_data_basic(resume_text)
        
        try:
            # Try to enhance with LLM for specific sections
            if len(resume_text) < 6000:  # Only for shorter text
                enhanced_skills = self._extract_skills_enhanced(resume_text)
                if enhanced_skills:
                    result['skills'] = enhanced_skills
                    
                enhanced_experience = self._extract_experience_enhanced(resume_text)
                if enhanced_experience:
                    result['experience'] = enhanced_experience
        except Exception as e:
            logger.warning(f"Enhanced extraction failed, using basic results: {e}")
        
        return result
    
    def _split_resume_into_sections(self, resume_text: str) -> Dict[str, str]:
        """Split resume into logical sections for processing."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'header': r'^(.*?)(?=\n.*(?:experience|education|skills|summary))',
            'experience': r'(work\s+experience|professional\s+experience|experience|employment)(.*?)(?=\n.*(?:education|skills|projects|certifications))',
            'education': r'(education|academic)(.*?)(?=\n.*(?:experience|skills|projects|certifications))',
            'skills': r'(skills|technical\s+skills|technologies)(.*?)(?=\n.*(?:experience|education|projects|certifications))',
            'projects': r'(projects|personal\s+projects)(.*?)(?=\n.*(?:experience|education|skills|certifications))'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(0)[:3000]  # Limit section size
        
        # If no clear sections found, use first part as header
        if not sections:
            sections['header'] = resume_text[:2000]
            sections['experience'] = resume_text[2000:8000]
            
        return sections
    
    def _extract_basic_info(self, header_text: str) -> Dict[str, str]:
        """Extract basic contact information."""
        prompt = f"""
        Extract contact information from this resume header text and return valid JSON:

        {{
            "name": "candidate's full name",
            "email": "email address",
            "phone": "phone number", 
            "address": "address",
            "linkedin": "LinkedIn URL",
            "website": "website URL",
            "github": "GitHub URL",
            "summary": "professional summary if present"
        }}

        Header text:
        {header_text}

        Return ONLY the JSON object:
        """
        
        try:
            return self.client.generate_json_response(prompt)
        except:
            return self._extract_resume_data_basic(header_text)
    
    def _extract_experience_section(self, experience_text: str) -> List[Dict]:
        """Extract work experience section."""
        prompt = f"""
        Extract work experience from this text and return valid JSON array:

        [
            {{
                "company": "Company name",
                "position": "Job title",
                "location": "Location",
                "start_date": "Start date", 
                "end_date": "End date",
                "responsibilities": ["responsibility 1", "responsibility 2"],
                "achievements": ["achievement 1", "achievement 2"],
                "technologies": ["tech 1", "tech 2"]
            }}
        ]

        Experience text:
        {experience_text}

        Return ONLY the JSON array:
        """
        
        try:
            result = self.client.generate_json_response(prompt)
            return result if isinstance(result, list) else []
        except:
            return []
    
    def _extract_skills_enhanced(self, resume_text: str) -> Dict[str, List[str]]:
        """Extract skills with LLM enhancement."""
        # Extract skills section text
        skills_pattern = r'(?:skills|technologies|technical\s+skills)[:\-\s]+(.*?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(skills_pattern, resume_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            skills_text = match.group(1)
            prompt = f"""
            Categorize these skills into JSON format:

            {{
                "technical": ["programming languages", "frameworks"],
                "tools": ["software tools", "platforms"],
                "soft": ["soft skills"],
                "languages": ["spoken languages"]
            }}

            Skills text: {skills_text}

            Return ONLY the JSON object:
            """
            
            try:
                return self.client.generate_json_response(prompt)
            except:
                pass
        
        # Fallback to basic extraction
        return {"technical": [], "tools": [], "soft": [], "languages": []}
    
    def _extract_experience_enhanced(self, resume_text: str) -> List[Dict]:
        """Extract experience with LLM enhancement."""
        # Try to find experience section
        exp_pattern = r'(?:experience|employment)[:\-\s]+(.*?)(?:\n(?:education|skills|projects)|\Z)'
        match = re.search(exp_pattern, resume_text, re.IGNORECASE | re.DOTALL)
        
        if match and len(match.group(1)) < 4000:  # Only if reasonably sized
            return self._extract_experience_section(match.group(1))
        
        return []
    
    def _extract_resume_data_basic(self, resume_text: str) -> Dict[str, Any]:
        """Enhanced basic fallback extraction using regex patterns."""
        logger.info("Using enhanced basic regex extraction as fallback")
        
        result = {
            "name": "",
            "email": "",
            "phone": "",
            "address": "",
            "linkedin": "",
            "website": "",
            "github": "",
            "summary": "",
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "languages": []
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        if emails:
            result["email"] = emails[0]
        
        # Extract phone (multiple patterns)
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, resume_text)
            if phones:
                result["phone"] = phones[0]
                break
        
        # Extract LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', resume_text, re.IGNORECASE)
        if linkedin_match:
            result["linkedin"] = "https://" + linkedin_match.group()
        
        # Extract GitHub
        github_match = re.search(r'github\.com/[\w-]+', resume_text, re.IGNORECASE)
        if github_match:
            result["github"] = "https://" + github_match.group()
        
        # Extract name (first line that looks like a name)
        lines = resume_text.strip().split('\n')
        for line in lines[:8]:  # Check first 8 lines
            line = line.strip()
            if (line and 
                len(line) > 3 and len(line) < 60 and 
                ' ' in line and
                not any(char in line for char in '@+()') and
                not any(keyword in line.lower() for keyword in ['resume', 'cv', 'curriculum', 'email', 'phone', 'address']) and
                re.match(r'^[A-Za-z\s\.,-]+$', line)):
                words = line.split()
                if 2 <= len(words) <= 4:
                    result["name"] = line
                    break
        
        # Enhanced summary extraction
        summary_patterns = [
            r'(?:PROFESSIONAL SUMMARY|SUMMARY|OBJECTIVE|PROFILE|ABOUT ME?)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|\n\s*\n|$)',
            r'(?:CAREER OBJECTIVE|PERSONAL STATEMENT)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|\n\s*\n|$)'
        ]
        for pattern in summary_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
                # Clean up the summary
                summary_lines = [line.strip() for line in summary.split('\n') if line.strip() and len(line.strip()) > 10]
                if summary_lines:
                    result["summary"] = ' '.join(summary_lines[:2])  # First 2 meaningful lines
                break
        
        # If no summary found, create a basic one
        if not result["summary"]:
            result["summary"] = f"Experienced professional seeking new opportunities in my field."
        
        # Enhanced skills extraction
        skills_patterns = [
            r'(?:TECHNICAL SKILLS|SKILLS|TECHNOLOGIES|COMPETENCIES|PROGRAMMING LANGUAGES?)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|\n\s*\n|$)',
            r'(?:TOOLS?|SOFTWARE|FRAMEWORKS?)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|\n\s*\n|$)'
        ]
        for pattern in skills_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                skills_text = match.group(1).strip()
                # Extract individual skills
                skills = []
                for line in skills_text.split('\n'):
                    line = line.strip().replace('•', '').replace('-', '').strip()
                    if line and len(line) < 100:
                        # Split by common delimiters
                        line_skills = re.split(r'[,;|•·]', line)
                        for skill in line_skills:
                            skill = skill.strip()
                            if skill and len(skill) > 1 and len(skill) < 30 and not skill.isdigit():
                                skills.append(skill)
                
                result["skills"] = list(set(skills))[:15]  # Remove duplicates and limit
                break
        
        # If no skills found in dedicated section, look for common tech terms
        if not result["skills"]:
            tech_terms = [
                'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'SQL', 'HTML', 'CSS',
                'Git', 'Docker', 'AWS', 'Azure', 'MongoDB', 'PostgreSQL', 'Machine Learning',
                'Data Analysis', 'Project Management', 'Agile', 'Scrum'
            ]
            found_skills = []
            for term in tech_terms:
                if re.search(rf'\b{re.escape(term)}\b', resume_text, re.IGNORECASE):
                    found_skills.append(term)
            result["skills"] = found_skills[:10]
        
        # Basic experience extraction
        exp_patterns = [
            r'(?:WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT|PROFESSIONAL EXPERIENCE)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|$)',
            r'(?:CAREER HISTORY|WORK HISTORY)[\s\n:]*([^A-Z]*?)(?:\n\s*[A-Z]{2,}|$)'
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                exp_text = match.group(1).strip()
                # Try to extract job titles and companies
                exp_lines = [line.strip() for line in exp_text.split('\n') if line.strip()]
                experience = []
                for line in exp_lines[:6]:  # Limit to 6 entries
                    if len(line) > 10 and len(line) < 100:
                        experience.append({
                            "position": line,
                            "company": "",
                            "duration": "",
                            "description": ""
                        })
                result["experience"] = experience
                break
        
        logger.info(f"Enhanced basic extraction completed: name='{result['name']}', email='{result['email']}', skills_count={len(result['skills'])}")
        return result
        
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract key information from job description.
        
        Args:
            job_description: Raw job description text
            
        Returns:
            Structured job requirements
        """
        prompt = f"""
        You are an expert job description analyzer. Extract comprehensive information from the job posting below and return it in valid JSON format.

        IMPORTANT: Return ONLY valid JSON with proper syntax. No additional text, explanations, or comments.

        Required JSON structure:
        {{
            "title": "Exact job title as posted",
            "company": "Company name",
            "location": "Full location (city, state, country, remote options)",
            "employment_type": "Full-time/Part-time/Contract/Freelance/Internship",
            "experience_level": "Entry/Junior/Mid-level/Senior/Lead/Principal/Executive",
            "department": "Department/team name if mentioned",
            "salary_range": "Salary range if mentioned (preserve exact format)",
            
            "required_skills": [
                "All mandatory technical skills mentioned",
                "Programming languages required",
                "Frameworks and libraries required"
            ],
            
            "preferred_skills": [
                "Nice-to-have technical skills",
                "Preferred programming languages",
                "Bonus technologies and tools"
            ],
            
            "soft_skills": [
                "Communication skills",
                "Leadership abilities", 
                "Teamwork and collaboration"
            ],
            
            "responsibilities": [
                "Primary responsibility 1 - preserve exact wording",
                "Key duty 2 with all details",
                "Development task 3"
            ],
            
            "qualifications": [
                "Education requirements (degree, field of study)",
                "Years of experience required",
                "Specific experience requirements"
            ],
            
            "keywords": [
                "All technical keywords for ATS optimization",
                "Industry-specific terms",
                "Important technologies mentioned"
            ],
            
            "benefits": [
                "Health insurance",
                "Retirement plans", 
                "Vacation/PTO"
            ],
            
            "company_description": "Description of the company and its mission/culture if provided"
        }}

        Job description to analyze:
        {job_description}
        """
        
        return self.client.generate_json_response(prompt)
    
    def tailor_resume_section(self, candidate_profile: Dict[str, Any], job_requirements: Dict[str, Any], 
                             section: str, original_content: str) -> str:
        """
        Tailor a specific resume section for a job.
        
        Args:
            candidate_profile: Candidate's profile data
            job_requirements: Job requirements data
            section: Section name to tailor
            original_content: Original section content
            
        Returns:
            Tailored section content
        """
        prompt = f"""
        You are an elite resume writer and ATS optimization expert. Transform resume content into compelling, results-driven statements that showcase exceptional value and perfectly align with job requirements.

        CANDIDATE PROFILE:
        {json.dumps(candidate_profile, indent=2)}
        
        TARGET JOB REQUIREMENTS:
        {json.dumps(job_requirements, indent=2)}
        
        SECTION TO ENHANCE: {section}
        ORIGINAL CONTENT: {original_content}
        
        TRANSFORMATION REQUIREMENTS:
        
        For RESPONSIBILITY statements:
        • Transform into achievement-focused bullet points with quantified results
        • Start each bullet with powerful action verbs: Led, Architected, Delivered, Optimized, Implemented, Designed, Scaled, etc.
        • Include specific technologies, frameworks, and methodologies from job posting
        • Add concrete metrics: percentages, dollar amounts, user numbers, performance improvements, time savings
        • Show business impact and value delivered, not just tasks performed
        • Each bullet must tell a complete success story in 1-2 lines
        
        For ACHIEVEMENT statements:
        • Highlight measurable accomplishments that demonstrate exceptional performance
        • Use the STAR method framework: Situation + Action + Result
        • Include recognition, awards, or standout contributions
        • Show competitive advantages and unique value propositions
        • Quantify everything possible with specific numbers and percentages
        
        CONTENT EXCELLENCE STANDARDS:
        1. Each statement must be unique - NO repetitive or similar content
        2. Use EXACT keywords and technologies from job requirements
        3. Focus on outcomes, not activities - what was achieved, not what was done
        4. Include scope and scale: team sizes, project budgets, user volumes, timeline impact
        5. Demonstrate progression and increasing responsibility
        6. Show both technical depth and business acumen
        7. Use industry-specific terminology and current technology trends
        8. Make every word count - eliminate filler and weak language
        
        FORMATTING RULES:
        • Use • symbol for bullet points (NOT -, *, or other symbols)
        • Each bullet: 1-2 lines maximum, 15-25 words ideal
        • NO markdown syntax (**, ##, etc.)
        • NO HTML tags or special formatting
        • Professional, confident tone
        • Consistent verb tenses (past tense for previous roles, present for current)
        
        QUALITY ASSURANCE:
        - Every bullet must start differently and cover distinct accomplishments
        - No vague terms like "responsible for," "worked on," or "helped with"
        - Include at least 2 relevant technical terms per bullet point
        - Ensure each statement would impress a hiring manager in 5 seconds
        - Content must be ATS-friendly while remaining human-readable
        
        OUTPUT FORMAT:
        Return ONLY the enhanced bullet points or content for the {section} section.
        NO explanatory text, headers, or additional commentary.
        Each bullet point should be immediately ready for resume inclusion.
        """
        
        response = self.client.generate_response(prompt)
        return self._clean_content(response)
    
    def generate_professional_summary(self, candidate_profile: Dict[str, Any], job_requirements: Dict[str, Any]) -> str:
        """
        Generate a professional summary tailored to the job.
        
        Args:
            candidate_profile: Candidate's profile data
            job_requirements: Job requirements data
            
        Returns:
            Tailored professional summary
        """
        prompt = f"""
        You are an elite executive resume writer with expertise in ATS optimization and modern hiring practices. Create a compelling professional summary that immediately captures attention and demonstrates exceptional value.

        CANDIDATE PROFILE:
        {json.dumps(candidate_profile, indent=2)}
        
        TARGET JOB REQUIREMENTS:
        {json.dumps(job_requirements, indent=2)}
        
        PROFESSIONAL SUMMARY REQUIREMENTS:
        
        Create exactly 4 sentences following this strategic structure:
        
        Sentence 1: "[X]+ years experienced [Job-Relevant Title] specializing in [3 EXACT technologies from job posting]"
        Sentence 2: "Proven expertise in [2-3 core competencies from job requirements] with track record of [specific quantified achievement with numbers]"  
        Sentence 3: "Led [leadership achievement with team size/scope] while delivering [business impact with specific metrics/percentages]"
        Sentence 4: "Strong [2 relevant soft skills from job posting] combined with [domain expertise] and passion for [industry-relevant focus area]"
        
        CONTENT EXCELLENCE STANDARDS:
        1. Use EXACT technical terms and keywords from the job posting - no substitutions
        2. Include specific, impressive quantified achievements (percentages, dollar amounts, user numbers, team sizes)
        3. Match the seniority level and experience mentioned in job requirements
        4. Incorporate industry-specific terminology and current trends
        5. Show both deep technical expertise and strong business acumen
        6. Each sentence must be information-dense yet readable
        7. ELIMINATE all generic phrases like "highly motivated," "team player," or "detail-oriented"
        8. Focus on unique value proposition and competitive advantages
        
        LANGUAGE REQUIREMENTS:
        - Executive-level, confident tone without being arrogant
        - Active voice throughout
        - Industry-appropriate terminology
        - Precise, impactful word choices
        - Professional but human-readable
        
        FORMATTING RULES:
        - Single, well-structured paragraph
        - NO bullet points, markdown, or special formatting
        - NO headers, explanations, or extra text
        - Ready for direct insertion into resume
        - Optimal length: 80-120 words total
        
        OUTPUT: Return ONLY the polished professional summary paragraph - no additional commentary or formatting.
        """
        
        response = self.client.generate_response(prompt)
        return self._clean_content(response)
    
    def _clean_content(self, content: str) -> str:
        """Clean up generated content by removing formatting artifacts and ensuring professional structure."""
        if not content:
            return ""
        
        # Remove markdown formatting and code block markers
        content = content.replace('**', '').replace('##', '').replace('*', '')
        content = content.replace('```', '').replace('`', '')
        
        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            'Here is ', 'Here\'s ', 'Based on ', 'According to ',
            'Summary:', 'Professional Summary:', 'Experience:', 'Responsibilities:'
        ]
        for prefix in prefixes_to_remove:
            if content.strip().startswith(prefix):
                content = content.strip()[len(prefix):].strip()
        
        # Clean and normalize line breaks
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Process lines for better structure
        cleaned_lines = []
        for line in lines:
            # Skip common filler lines
            if line.lower() in ['summary', 'experience', 'responsibilities', 'achievements']:
                continue
                
            # Standardize bullet points
            if line.lstrip().startswith(('•', '-', '*', '◦', '▸', '→')):
                bullet_content = line.lstrip()[1:].strip()
                if bullet_content and len(bullet_content) > 10:  # Ensure substantial content
                    # Capitalize first letter if not already
                    bullet_content = bullet_content[0].upper() + bullet_content[1:] if bullet_content else ""
                    # Ensure period at end for professional appearance
                    if bullet_content and not bullet_content.endswith(('.', '!', '?', ')')):
                        bullet_content += '.'
                    cleaned_lines.append(f"• {bullet_content}")
            elif line.strip() and len(line.strip()) > 15:  # Regular content with minimum length
                # For paragraph content, ensure proper capitalization
                line = line.strip()
                if line and not line[0].isupper():
                    line = line[0].upper() + line[1:]
                cleaned_lines.append(line)
        
        # Remove duplicate or very similar lines
        final_lines = []
        seen_content = []
        
        for line in cleaned_lines:
            line_content = line.replace('• ', '').lower().strip()
            
            # Check for substantial similarity (70% word overlap)
            is_duplicate = False
            for seen in seen_content:
                seen_words = set(seen.split())
                current_words = set(line_content.split())
                if len(seen_words) > 0 and len(current_words) > 0:
                    overlap = len(seen_words & current_words) / max(len(seen_words), len(current_words))
                    if overlap > 0.7:  # 70% similarity threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_lines.append(line)
                seen_content.append(line_content)
        
        # Join and ensure proper spacing
        result = '\n'.join(final_lines)
        
        # Final cleanup: remove extra spaces and ensure single spaces
        result = ' '.join(result.split())
        
        # For bullet points, add line breaks
        if '•' in result:
            result = result.replace('• ', '\n• ').strip()
        
        return result
    
    def optimize_for_ats(self, content: str, keywords: List[str]) -> str:
        """
        Optimize content for ATS systems.
        
        Args:
            content: Original content
            keywords: Keywords to incorporate
            
        Returns:
            ATS-optimized content
        """
        keywords_str = ", ".join(keywords)
        
        prompt = f"""
        Optimize the following content for ATS (Applicant Tracking Systems) by naturally incorporating these keywords: {keywords_str}
        
        Original content:
        {content}
        
        Guidelines:
        1. Incorporate keywords naturally without keyword stuffing
        2. Maintain readability and professional tone
        3. Use industry-standard terminology
        4. Keep the core meaning and achievements intact
        5. Use proper formatting and structure
        
        Return only the optimized content without explanations.
        """
        
        return self.client.generate_response(prompt)