"""
LLM Client module for handling Mistral and Gemini API integrations.
Supports both models with standardized interface and error handling.
"""

import asyncio
import httpx
import time
import uuid
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Intelligent rate limiter for API calls to prevent overwhelming servers.
    Adapts to API response patterns and implements smart delays.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.consecutive_rate_limits = 0
        self.success_count = 0
        
    async def wait_if_needed(self):
        """Smart waiting based on recent API behavior."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Calculate required delay based on recent performance
        if self.consecutive_rate_limits > 0:
            required_delay = min(self.max_delay, self.base_delay * (2 ** self.consecutive_rate_limits))
        else:
            required_delay = self.base_delay * (1 + (self.consecutive_rate_limits * 0.5))
            
        if time_since_last < required_delay:
            wait_time = required_delay - time_since_last
            logger.info(f"Rate limiter: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            
        self.last_request_time = time.time()
        
    def record_success(self):
        """Record successful API call."""
        self.success_count += 1
        if self.success_count > 5:  # Reset after several successes
            self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - 1)
            self.success_count = 0
            
    def record_rate_limit(self):
        """Record rate limit hit."""
        self.consecutive_rate_limits += 1
        self.success_count = 0


class LLMClient:
    """
    Unified client for multiple LLM backends (Mistral, Gemini).
    Handles API calls, retries, and response processing.
    """
    
    def __init__(self, api_key: str, model_type: str):
        """
        Initialize LLM client with API key and model type.
        
        Args:
            api_key (str): API key for the selected LLM service
            model_type (str): Either 'mistral' or 'gemini'
        """
        self.api_key = api_key or ""  # Handle None/empty API keys
        self.model_type = model_type.lower()
        
        # Mistral API configuration - using OpenRouter for Mistral 7B
        self.mistral_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Default Mistral model for OpenRouter
        self.mistral_model = 'mistralai/mistral-7b-instruct'
        
        # OpenRouter model mapping for various types
        self.openrouter_models = {
            'mistral': 'mistralai/mistral-7b-instruct',
            'openchat': 'openchat/openchat-7b',
            'deepseek': 'deepseek/deepseek-r1',
            'gptoss': 'gpt-oss-20b'
        }
        
        # Gemini configuration with enhanced settings for accuracy
        if self.model_type == 'gemini':
            genai.configure(api_key=api_key)
            
            # Configure generation settings for optimal speed and accuracy balance
            generation_config = genai.types.GenerationConfig(
                temperature=0.02,       # Slightly reduced for better accuracy
                top_p=0.85,            # Optimized for speed while maintaining quality
                top_k=40,              # Balanced for speed and vocabulary coverage
                max_output_tokens=900, # Increased to ensure complete long descriptions without truncation
                candidate_count=1       # Single best candidate
            )
            
            # Enhanced system instruction for pharmaceutical accuracy
            system_instruction = (
             "Always be factually accurate and industry-compliant."
             "Focus on therapeutic benefits, proper usage, dosage forms, and key features."
             "Use a clear, formal, and informative tone."
             "Do not write like an advertisement."
             "Do not include disclaimers, warnings, or unnecessary filler."
             "Write only the pure description, starting directly with the product information."
            )
            
            self.gemini_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=generation_config,
                system_instruction=system_instruction
            )
    
    def _clean_response(self, text: str) -> str:
        """
        Aggressively clean the LLM response and convert asterisks to circle bullets.
        
        Args:
            text (str): Raw response text from LLM
            
        Returns:
            str: Cleaned text with circle bullets (•) instead of asterisks
        """
        try:
            if not text or not isinstance(text, str):
                logger.debug("Empty or invalid text input to _clean_response")
                return ""
            
            import re
            
            # Remove model control tokens that shouldn't appear in output
            # These are special tokens used by various AI models
            control_tokens = [
                '[Bot]', '[/Bot]', '[INST]', '[/INST]', '[SYS]', '[/SYS]',
                '<|im_start|>', '<|im_end|>', '<s>', '</s>',
                '[OUT]', '[/OUT]', '<bot>', '</bot>', '<assistant>', '</assistant>',
                '<<SYS>>', '<</SYS>>', '[BINST]', '[/BINST]',
                '[EOS]', '</EOS>', '<EOS>', '<eos>', '[/EOS]',
                '[BOS]', '[BOT]', '[BOT\\]', '[/BOT]'
            ]
            for token in control_tokens:
                text = text.replace(token, '')
            
            # SUPER AGGRESSIVE ASTERISK REMOVAL - Remove ALL asterisks first (including unicode variants)
            text = str(text).replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            
            # Now process line by line for bullet conversion
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                try:
                    line = str(line).strip().replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
                    # Convert dash bullets to circle bullets
                    if line.startswith('- '):
                        line = '• ' + line[2:]
                    elif line.startswith('-'):
                        line = '• ' + line[1:]
                    # If line has content but no bullet, add one for short descriptions
                    elif line and not line.startswith('•'):
                        # Only add bullets if it looks like a bullet point (not paragraph text)
                        if len(line) < 100 and line and (line[0].isupper() or line[0].isdigit()):
                            line = '• ' + line
                    cleaned_lines.append(line)
                except Exception as e:
                    logger.warning(f"Error processing line in _clean_response: {str(e)}")
                    # Add the original line if processing fails
                    cleaned_lines.append(str(line).strip().replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', ''))
            text = '\n'.join(cleaned_lines)
            
            # Apply regex cleaning with error handling
            try:
                # Remove markdown bold formatting (**text** and __text__)
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                text = re.sub(r'__(.*?)__', r'\1', text)
                # Remove markdown italic formatting (*text* and _text_)
                text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)
                text = re.sub(r'_([^_\n]+?)_', r'\1', text)
                # Remove any remaining asterisks (should be none left)
                text = re.sub(r'[\*＊﹡∗]+', '', text)
                # Remove any remaining underscores used for formatting
                text = re.sub(r'[_]+', '', text)
                # Remove markdown headers (# ## ###)
                text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
                # Remove markdown code blocks (```text```)
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
                
                # Remove inline code (`text`)
                text = re.sub(r'`(.*?)`', r'\1', text)
                
                # Remove HTML tags if any
                text = re.sub(r'<[^>]+>', '', text)
                
                # Remove any other special formatting characters
                text = re.sub(r'[~`]', '', text)
                
                # Clean up extra whitespace
                text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
                text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single
                
            except Exception as e:
                logger.warning(f"Error in regex cleaning: {str(e)}")
                # Fallback to basic cleaning if regex fails
                text = text.replace('*', '').replace('_', '').replace('`', '')
            
            # FINAL ASTERISK REMOVAL - Make absolutely sure no asterisks remain
            text = str(text).replace('*', '')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Critical error in _clean_response: {str(e)}")
            # Return safe fallback
            return str(text).replace('*', '').strip() if text else ""
    
    async def generate_description(
        self,
        product_name: str,
        description_type: str,
        category: str = None,
        retry_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate high-accuracy product description with validation.
        
        Args:
            product_name (str): Name of the pharmaceutical product
            description_type (str): Either 'short' or 'long'
            retry_context (dict, optional): Context from previous attempts
            
        Returns:
            str: Generated description or empty string if failed
        """
        try:
            # Input validation
            if not product_name or not isinstance(product_name, str):
                logger.error(f"Invalid product name: {product_name}")
                return ""
            
            if description_type not in ['short', 'long']:
                logger.error(f"Invalid description type: {description_type}")
                return ""
            
            product_name = str(product_name).strip()
            if not product_name:
                logger.error("Empty product name after cleaning")
                return ""
            
            if retry_context:
                try:
                    logger.debug(
                        "Retrying %s description for %s (attempt %s) with context keys=%s",
                        description_type,
                        product_name,
                        retry_context.get('attempt'),
                        list(retry_context.keys())
                    )
                except Exception:
                    logger.debug(
                        "Retrying %s description for %s with retry context", description_type, product_name
                    )

            prompt = self._get_prompt(
                product_name,
                description_type,
                category,
                retry_context=retry_context
            )
            if not prompt:
                logger.error(f"Failed to generate prompt for {product_name}")
                return ""
            
            # Generate description with selected model
            raw_description = ""
            try:
                if self.model_type in self.openrouter_models:
                    model_id = self.openrouter_models[self.model_type]
                    raw_description = await self._call_mistral(prompt, model_override=model_id)
                elif self.model_type == 'gemini':
                    raw_description = await self._call_gemini(prompt)
                else:
                    logger.error(f"Unsupported model type: {self.model_type}")
                    return ""
            except Exception as e:
                logger.error(f"API call failed for {product_name}: {str(e)}")
                return ""
            
            if not raw_description:
                logger.warning(f"Empty response from API for {product_name}")
                return ""
            
            # FIRST: Clean the raw response to remove all asterisks
            try:
                cleaned_description = self._clean_response(raw_description)
            except Exception as e:
                logger.error(f"Cleaning failed for {product_name}: {str(e)}")
                # Fallback cleaning
                cleaned_description = str(raw_description).replace('*', '').strip()
            
            # THEN: Validate and enhance the cleaned description
            try:
                validated_description = self._validate_and_enhance_description(
                    cleaned_description, product_name, description_type
                )
            except Exception as e:
                logger.error(f"Validation failed for {product_name}: {str(e)}")
                # Return cleaned description as fallback
                validated_description = cleaned_description
            
            return validated_description or ""
                
        except Exception as e:
            logger.error(f"Critical error generating description for {product_name}: {str(e)}")
            return ""
    
    def _validate_and_enhance_description(self, description: str, product_name: str, description_type: str) -> str:
        """
        Validate and enhance the generated description for maximum accuracy and completeness.
        
        Args:
            description (str): Raw generated description
            product_name (str): Product name for context
            description_type (str): Type of description (short/long)
            
        Returns:
            str: Validated and enhanced description with improved accuracy
        """
        try:
            # Input validation
            if not description or not isinstance(description, str):
                logger.warning(f"Empty or invalid description for {product_name}")
                return ""
            
            if not product_name or not isinstance(product_name, str):
                logger.error(f"Invalid product name in validation: {product_name}")
                return str(description).strip()
            
            if description_type not in ['short', 'long']:
                logger.error(f"Invalid description type in validation: {description_type}")
                return str(description).strip()
            
            description = str(description).strip()
            product_name = str(product_name).strip()
            
            if len(description) < 15:
                logger.warning(f"Generated description too short for {product_name}: {len(description)} chars")
                return ""
            
            # Clean and validate description with error handling
            try:
                # SUPER AGGRESSIVE: Remove ALL asterisks - no exceptions
                description = description.replace('*', '')
                
                # Remove other formatting characters
                description = description.replace('_', '').replace('`', '')
                
                # Convert any remaining dash bullets to circle bullets
                lines = description.split('\n')
                cleaned_lines = []
                for line in lines:
                    try:
                        line = str(line).strip()
                        if line.startswith('- '):
                            line = '• ' + line[2:]
                        elif line.startswith('-'):
                            line = '• ' + line[1:]
                        cleaned_lines.append(line)
                    except Exception as e:
                        logger.warning(f"Error processing line in validation: {str(e)}")
                        cleaned_lines.append(str(line).strip().replace('*', ''))
                
                description = '\n'.join(cleaned_lines)
                
                # Remove any markdown formatting that might have been missed
                import re
                try:
                    description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)
                    description = re.sub(r'\*(.*?)\*', r'\1', description)
                    description = re.sub(r'__(.*?)__', r'\1', description)
                    description = re.sub(r'_(.*?)_', r'\1', description)
                except Exception as e:
                    logger.warning(f"Regex error in validation: {str(e)}")
                
                # FINAL ASTERISK CHECK - Remove any that might have been added back
                description = description.replace('*', '')
                
            except Exception as e:
                logger.error(f"Error in description cleaning: {str(e)}")
                # Fallback to basic cleaning
                description = str(description).replace('*', '').replace('_', '').strip()
            
            # Enhanced accuracy validation with error handling
            try:
                product_lower = product_name.lower()
                description_lower = description.lower()
                
                # Check if description is relevant to the product
                product_keywords = product_lower.split()
                relevance_score = sum(1 for keyword in product_keywords if keyword and keyword in description_lower)
                
                if relevance_score == 0:
                    # Add product context if missing
                    try:
                        if description_type == 'short':
                            description = f"• {product_name} - " + description.lstrip('•').strip()
                        else:
                            description = f"{product_name} " + description.lower()
                            if description:
                                description = description[0].upper() + description[1:]
                    except Exception as e:
                        logger.warning(f"Error adding product context: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Error in relevance validation: {str(e)}")
                
            return self._format_description_safely(description, description_type, product_name)
            
        except Exception as e:
            logger.error(f"Critical error in validation for {product_name}: {str(e)}")
            # Return safe fallback
            return str(description).replace('*', '').strip() if description else ""
    
    def _format_description_safely(self, description: str, description_type: str, product_name: str) -> str:
        """
        Safely format description based on type with comprehensive error handling.
        
        Args:
            description (str): Description to format
            description_type (str): 'short' or 'long'
            product_name (str): Product name for context
            
        Returns:
            str: Safely formatted description
        """
        try:
            if not description or not isinstance(description, str):
                logger.warning(f"Empty description in formatting for {product_name}")
                return ""
                
            description = str(description).strip()
            
            if description_type == 'short':
                return self._format_short_description_safely(description, product_name)
            elif description_type == 'long':
                return self._format_long_description_safely(description, product_name)
            else:
                logger.error(f"Invalid description type in formatting: {description_type}")
                return description.replace('*', '')
                
        except Exception as e:
            logger.error(f"Error in format_description_safely for {product_name}: {str(e)}")
            return str(description).replace('*', '').strip() if description else ""
    
    def _format_short_description_safely(self, description: str, product_name: str) -> str:
        """
        Safely format short description with bullet points.
        """
        try:
            # Enhanced bullet point validation - ensure circle bullets
            lines = description.split('\n')
            bullet_lines = []
            
            for line in lines:
                try:
                    line = str(line).strip()
                    if line:
                        # REMOVE ANY ASTERISKS FIRST - before any processing
                        line = line.replace('*', '')
                        
                        # Convert dash bullets to circle bullets
                        if line.startswith('-'):
                            line = '•' + line[1:]
                        elif not line.startswith('•'):
                            # Add circle bullet if missing
                            line = '• ' + line
                        
                        # Clean and format the line - NO ASTERISKS ALLOWED
                        line = line.replace('_', '').strip()
                        if line.endswith('.') or line.endswith(','):
                            line = line[:-1]  # Remove punctuation
                        
                        # FINAL CHECK: Remove any asterisks that might have been added
                        line = line.replace('*', '')
                        
                        if line.strip():  # Only add non-empty lines
                            bullet_lines.append(line)
                            
                except Exception as e:
                    logger.warning(f"Error processing bullet line: {str(e)}")
                    # Add safe version of the line
                    safe_line = str(line).replace('*', '').strip()
                    if safe_line and not safe_line.startswith('•'):
                        safe_line = '• ' + safe_line
                    if safe_line:
                        bullet_lines.append(safe_line)
            
            # Ensure we have reasonable number of bullet points
            try:
                if len(bullet_lines) < 2:
                    logger.warning(f"Insufficient bullet points for {product_name}: {len(bullet_lines)}")
                    # Pad with safe generic bullets if needed
                    while len(bullet_lines) < 4:
                        if len(bullet_lines) == 0:
                            bullet_lines.append(f"• Effective {product_name.lower()} solution")
                        elif len(bullet_lines) == 1:
                            bullet_lines.append("• High-quality healthcare product")
                        elif len(bullet_lines) == 2:
                            bullet_lines.append("• Trusted and reliable formula")
                        else:
                            bullet_lines.append("• Professional grade quality")
                
                # Take only first 4 bullet points for consistency
                formatted_bullets = bullet_lines[:4]
                
                return '\n'.join(formatted_bullets)
                
            except Exception as e:
                logger.error(f"Error in bullet point padding: {str(e)}")
                return '\n'.join(bullet_lines) if bullet_lines else ""
                
        except Exception as e:
            logger.error(f"Critical error in short description formatting: {str(e)}")
            # Return safe fallback
            return str(description).replace('*', '').strip()
    
    def _format_long_description_safely(self, description: str, product_name: str) -> str:
        """
        Safely format long description.
        """
        try:
            # Remove all asterisks and basic formatting
            description = description.replace('*', '').replace('_', '').strip()
            
            # Enhanced long description validation for SHORT, simple content
            sentences = description.split('.')
            if len(sentences) < 2:
                logger.warning(f"Long description too short for {product_name}")
                return description  # Return as-is if too short to process
            
            # Enhanced product name integration
            try:
                if product_name.lower() not in description.lower():
                    # Add product name with better integration
                    first_sentence = sentences[0].strip()
                    if first_sentence:
                        description = f"{product_name} " + first_sentence.lower() + '. ' + '. '.join(sentences[1:])
                        if description:
                            description = description[0].upper() + description[1:]
            except Exception as e:
                logger.warning(f"Error integrating product name: {str(e)}")
            
            # FINAL ASTERISK REMOVAL - Make absolutely sure no asterisks remain (including unicode)
            description = str(description).replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            if '*' in description or '＊' in description or '﹡' in description or '∗' in description:
                logger.warning('Asterisk detected in cleaned text! Forcing removal.')
                description = description.replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error in long description formatting: {str(e)}")
            return str(description).replace('*', '').strip()
    
    def _get_prompt(
        self,
        product_name: str,
        description_type: str,
        category: str = None,
        retry_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate standardized product descriptions for e-commerce pharmaceutical platform.
        
        Args:
            product_name (str): Product name to generate description for
            description_type (str): 'short' or 'long'
            
        Returns:
            str: Formatted prompt for standardized descriptions
        """
        try:
            if not product_name or not isinstance(product_name, str):
                logger.error(f"Invalid product name in _get_prompt: {product_name}")
                return ""
            
            product_name = str(product_name).strip()
            if not product_name:
                return ""
            
            # Compose strict context for both descriptions
            base_context = (
                f"Important: The short description and long description must ALWAYS describe the SAME medicine.\n"
                f"Do NOT replace or confuse the active ingredient with another drug.\n"
                f"If the short description says the medicine is for a specific condition (e.g., ADHD), the long description must also be for that same condition (not for any other condition or drug, e.g., not acid reflux, asthma, or pain relief).\n"
                f"Use ONLY the active ingredient(s), dosage form, strength, and therapeutic use given in the input.\n"
                f"Do NOT introduce any different or conflicting indications, ingredients, or uses between the short and long descriptions.\n"
                f"Both the short and long descriptions must highlight the same primary uses/indications of the medicine. Do not emphasize one condition or use in the long description if the short description gives equal weight to multiple uses (e.g., allergies and other symptoms). Maintain consistency in the main therapeutic focus between both descriptions.\n"
                f"If uncertain, keep the description general but consistent across both short and long versions.\n"
            )

            if retry_context:
                try:
                    attempt = retry_context.get('attempt')
                    missing = retry_context.get('missing')
                    existing_short = (retry_context.get('existing_short') or '').strip()
                    existing_long = (retry_context.get('existing_long') or '').strip()
                    previous_errors = retry_context.get('previous_errors')

                    def _trim(value: str, limit: int = 400) -> str:
                        value = value.strip()
                        if len(value) <= limit:
                            return value
                        return value[:limit].rstrip() + "..."

                    context_lines = []
                    if attempt and attempt > 1:
                        context_lines.append(
                            f"This is attempt #{attempt}. The previous attempt did not produce a complete {description_type} description."
                        )
                    if missing == 'long' and existing_short:
                        context_lines.append(
                            "Keep the long description aligned with this approved short description (do NOT rewrite it):\n"
                            f"{_trim(existing_short)}"
                        )
                    if missing == 'short' and existing_long:
                        context_lines.append(
                            "Generate the short bullet points so they match this already approved long description (do NOT contradict it):\n"
                            f"{_trim(existing_long)}"
                        )
                    if previous_errors:
                        context_lines.append(f"Previous issues to avoid: {_trim(str(previous_errors), 200)}")

                    if context_lines:
                        base_context += "\n" + "\n".join(context_lines) + "\n"
                except Exception as context_error:
                    logger.debug(f"Retry context could not be attached to prompt for {product_name}: {context_error}")

            if category:
                base_context += f"\nCategory: {category}\nUse the category only if it is relevant to the product's therapeutic use or context.\n"

            if description_type == 'short':
                return (
                    base_context +
                    f"You are helping me generate standardized product descriptions for an e-commerce platform selling pharmaceutical and healthcare products.\n\n"
                    f"Generate a SHORT DESCRIPTION for '{product_name}' with these exact requirements:\n\n"
                    f"✅ Short Description:\n"
                    f"• 4 concise small bullet points, no punctuation at the end\n\n"
                     f"• Do NOT use numbers or numbering (like 1., 2., etc.) after the bullet points—just plain bullet points.\n\n"
                    f"⚠ Very Important: DO NOT change the format, tone, or style. Continue exactly as in previous descriptions generated.\n\n"
                    f"Format each bullet point as:\n"
                    f"• First benefit\n"
                    f"• Second benefit\n"
                    f"• Third benefit\n"
                    f"• Fourth benefit\n\n"
                    f"Generate ONLY the 4 small bullet points for '{product_name}'. Use professional and simple pharmaceutical language."
                )
            elif description_type == 'long':
                return (
                    base_context +
                    f"You are helping me generate standardized product descriptions for an e-commerce platform selling pharmaceutical and healthcare products.\n\n"
                    f"Generate a LONG DESCRIPTION for '{product_name}' with these exact requirements:\n\n"
                    f"💊 Long Description:\n"
                    f"EXACTLY 5-6 complete lines (approximately 120-150 words maximum)\n"
                    f"Keep it concise and focused\n"
                    f"SEO-optimized, professional, and easy to understand\n"
                    f"Consistent tone and length\n"
                    f"MUST end with a complete sentence - do NOT stop mid-sentence\n\n"
                    f"Dont start with Titles for every product , just pure description of the product, no symbols too\n"
                    f"no bullet points\n"
                    f"⚠ Very Important: DO NOT change the format, tone, or style. Continue exactly as in previous descriptions generated.\n\n"
                    f"⚠ CRITICAL: Keep it SHORT and CONCISE (120-150 words max). Complete ALL sentences. Do not stop mid-sentence. Generate EXACTLY 5-6 complete lines for '{product_name}'. Make it SEO-optimized with consistent tone."
                )
            else:
                logger.error(f"Invalid description type: {description_type}")
                return ""

        except Exception as e:
            logger.error(f"Error generating prompt for {product_name}: {str(e)}")
            return f"Generate a professional description for {product_name}."
    
    async def _call_mistral(self, prompt: str, max_retries: int = 5, model_override: str = None) -> str:
        """
        Call Mistral/OpenRouter API with enhanced retry logic and exponential backoff.
        Allows model override for other OpenRouter models.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        model_id = model_override if model_override else self.mistral_model
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert pharmaceutical content specialist with comprehensive knowledge of medications, medical devices, supplements, and healthcare products. You possess deep expertise in drug mechanisms, therapeutic applications, dosage forms, and pharmaceutical marketing. Your role is to create highly accurate, professionally written, and compliant content for healthcare e-commerce platforms. Always ensure factual accuracy, use appropriate medical terminology, and focus on therapeutic benefits and key product features. Your content must meet the highest pharmaceutical industry standards for accuracy and professionalism."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,  # Reduced for faster processing while maintaining quality
            "temperature": 0.02,  # Further reduced for maximum accuracy and consistency
            "top_p": 0.9,        # Increased for better coverage
            "frequency_penalty": 0.3,  # Increased to reduce repetition
            "presence_penalty": 0.2   # Fine-tuned for diverse, accurate content
        }
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout for speed
                    response = await client.post(
                        self.mistral_api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        raw_content = result["choices"][0]["message"]["content"].strip()
                        
                        # Check if response is empty or contains only whitespace/control tokens
                        if not raw_content or len(raw_content) < 10:
                            logger.warning(f"Empty or very short API response (attempt {attempt + 1}), retrying...")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 + attempt)
                                continue
                            return ""
                        
                        cleaned = self._clean_response(raw_content)
                        # Verify cleaned content is not empty
                        if not cleaned or len(cleaned.strip()) < 10:
                            logger.warning(f"Content became empty after cleaning (attempt {attempt + 1}), retrying...")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 + attempt)
                                continue
                            return ""
                        
                        return cleaned
                    elif response.status_code == 429:  # Rate limit
                        # Enhanced rate limiting strategy
                        wait_time = min(60, (2 ** attempt) + (attempt * 2))  # Cap at 60 seconds
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status_code in [502, 503, 504]:  # Server errors
                        wait_time = 5 + (attempt * 2)
                        logger.warning(f"Server error {response.status_code}, retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return ""
                        
            except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 + (attempt * 3)
                    await asyncio.sleep(wait_time)
                    continue
                return ""
            except Exception as e:
                logger.error(f"Mistral API call failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 3 + (attempt * 2)
                    await asyncio.sleep(wait_time)
                    continue
                return ""
        
        logger.error(f"Failed to get response from Mistral after {max_retries} attempts")
        return ""
    
    async def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Enhanced Gemini API call with improved accuracy settings.
        
        Args:
            prompt (str): Prompt to send to Gemini
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Generated text or empty string if failed
        """
        for attempt in range(max_retries):
            try:
                # Enhanced prompt with context for better accuracy
                enhanced_prompt = (
                    f"As a pharmaceutical content specialist, provide a highly accurate, "
                    f"professional response to the following request:\n\n{prompt}\n\n"
                    f"Important: Use precise medical terminology, focus on factual benefits, "
                    f"and ensure content is suitable for healthcare professionals and patients."
                    f"normal people should also be able to understand it."
                )
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.gemini_model.generate_content(enhanced_prompt)
                )
                
                if response.text:
                    raw_content = response.text.strip()
                    return self._clean_response(raw_content)
                else:
                    logger.warning("Gemini returned empty response")
                    return ""
                    
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "rate" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini rate limit hit, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return ""
        
        return ""


class BatchProcessor:
    """
    Enhanced batch processor optimized for large-scale pharmaceutical product processing.
    Includes intelligent rate limiting, caching, and error recovery.
    """
    
    def __init__(self, llm_client: LLMClient, batch_size: int = 8):  # Increased from 5 to 8 for more parallelism
        """
        Initialize batch processor with optimized settings for large datasets.
        
        Args:
            llm_client (LLMClient): LLM client instance
            batch_size (int): Number of products to process concurrently (increased for speed)
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.cache = {}  # Simple in-memory cache for repeated products
        self.rate_limiter = RateLimiter(base_delay=0.3, max_delay=10.0)  # Lower base delay and max delay
        self.retry_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def process_products(
        self,
        products: list,
        progress_callback=None,
        stop_check=None,
        existing_results: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        Process list of products in batches with enhanced progress tracking and error recovery.
        Optimized for large datasets (10,000+ products).
        
        Args:
            products (list): List of product names
            progress_callback (callable): Optional callback for progress updates
            stop_check (callable): Optional function to check if processing should stop
            existing_results (dict, optional): Previously captured results for context-aware retries
            
        Returns:
            list: List of dictionaries with product info and descriptions
        """
        existing_results = existing_results or {}
        results = []
        total_products = len(products)
        logger.info(f"Starting batch processing of {total_products} products with batch size {self.batch_size}")
        
        for i in range(0, total_products, self.batch_size):
            # Check for stop request before processing each batch
            if stop_check and stop_check():
                logger.info(f"Processing stopped by user request at {len(results)}/{total_products}")
                break
                
            batch = products[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: products {i+1}-{min(i+len(batch), total_products)}")
            
            # Apply rate limiting before each batch
            await self.rate_limiter.wait_if_needed()
            
            batch_results = await self._process_batch_with_retry(batch, existing_results)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.debug(f"Batch returned exception: {result}")
                    continue
                results.append(result)

            # Update progress with detailed info
            if progress_callback:
                success_count = len([r for r in results if isinstance(r, dict) and r.get('status') == 'success'])
                incomplete_count = len([r for r in results if isinstance(r, dict) and r.get('status') != 'success'])
                processed_so_far = min(i + len(batch), total_products)
                progress = min(100, int((processed_so_far / total_products) * 100))
                if progress_callback.__code__.co_argcount >= 4:
                    progress_callback(progress, success_count, total_products, incomplete_count)
                else:
                    progress_callback(progress, success_count, total_products)
            
            # Adaptive delay based on API performance - reduced for speed
            await asyncio.sleep(self._calculate_delay(incomplete_count, success_count))

        success_total = len([r for r in results if isinstance(r, dict) and r.get('status') == 'success'])
        incomplete_total = len([r for r in results if isinstance(r, dict) and r.get('status') != 'success'])
        logger.info(
            f"Batch processing complete: {success_total} successful, {incomplete_total} incomplete/failed"
        )
        return results
        
    def _calculate_delay(self, failed_count: int, success_count: int) -> float:
        """Calculate adaptive delay based on success/failure ratio - optimized for speed."""
        if failed_count == 0:
            return 0.1  # Minimal delay when everything works
        
        failure_ratio = failed_count / max(success_count, 1)
        if failure_ratio > 0.5:
            return 1.0  # Reduced delay even for many failures
        elif failure_ratio > 0.2:
            return 0.7  # Reduced medium delay
        else:
            return 0.3  # Reduced short delay

    def _build_retry_context(
        self,
        product_name: str,
        existing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create retry context payloads for short and long description prompts."""
        normalized = (product_name or "").lower().strip()
        if not normalized:
            normalized = str(uuid.uuid4())  # Fallback to unique key to avoid collisions

        prior_meta = self.retry_metadata.get(normalized, {}) if normalized else {}
        base_attempt = max(1, int(prior_meta.get('attempt', 0)) + 1)

        if existing_context and isinstance(existing_context, dict):
            try:
                base_attempt = max(base_attempt, int(existing_context.get('attempt', 1)) + 1)
            except Exception:
                base_attempt = max(base_attempt, 2)

        existing_short = ((existing_context or {}).get('short_description') or '').strip()
        existing_long = ((existing_context or {}).get('long_description') or '').strip()
        previous_errors = (existing_context or {}).get('errors') or prior_meta.get('last_error')

        if existing_context is not None and isinstance(existing_context, dict):
            short_completed = bool(existing_context.get('short_completed'))
            long_completed = bool(existing_context.get('long_completed'))
        else:
            short_completed = bool(prior_meta.get('short_completed'))
            long_completed = bool(prior_meta.get('long_completed'))

        meta = {
            'attempt': base_attempt,
            'existing_short': existing_short,
            'existing_long': existing_long,
            'previous_errors': previous_errors,
            'short_completed': short_completed,
            'long_completed': long_completed
        }

        short_context = {
            'attempt': base_attempt,
            'existing_short': existing_short,
            'existing_long': existing_long,
            'previous_errors': previous_errors,
            'missing': None
        }

        long_context = dict(short_context)

        if existing_context:
            if not short_completed:
                short_context['missing'] = 'short'
            if not long_completed:
                long_context['missing'] = 'long'

        self.retry_metadata[normalized] = {
            'attempt': base_attempt,
            'timestamp': time.time(),
            'short_completed': short_completed,
            'long_completed': long_completed
        }

        return {
            'meta': meta,
            'short': short_context,
            'long': long_context
        }
    
    async def _retry_failed_products_fast(self, failed_products: list, stop_check=None) -> list:
        """
        Fast retry for critically failed products with minimal overhead.
        
        Args:
            failed_products (list): List of failed product names
            stop_check (callable): Function to check for stop request
            
        Returns:
            list: Results from retry attempts
        """
        retry_results = []
        
        for product in failed_products[:10]:  # Limit retries
            if stop_check and stop_check():
                break
                
            try:
                # Single fast retry with minimal timeout
                result = await asyncio.wait_for(
                    self._process_single_product_fast(product),
                    timeout=15.0
                )
                retry_results.append(result)
                
            except Exception as e:
                logger.debug(f"Retry failed for {product}: {str(e)}")
                # Don't add failed retries to avoid duplicates
                continue
                
            # Minimal delay between retries
            await asyncio.sleep(0.1)
        
        return retry_results
    
    async def _process_batch_with_retry(
        self,
        batch: list,
        existing_results: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        Optimized batch processing with parallel execution and smart retry logic.
        
        Args:
            batch (list): List of product names to process
            existing_results (dict, optional): Previously captured results to reuse partial data
            
        Returns:
            list: Processed results for the batch
        """
        batch_results = []
        
        # Create all tasks for parallel execution
        tasks = []
        for product in batch:
            # product is a dict: {"product_name": ..., "category": ...}
            product_name = str(product.get("product_name", "")).strip()
            cache_key = product_name.lower()

            cached_result = self.cache.get(cache_key)
            if cached_result and cached_result.get('status') == 'success':
                batch_results.append(cached_result)
                continue

            context = existing_results.get(cache_key)

            task = self._process_single_product_fast(product, existing_context=context)
            tasks.append((product, task, cache_key))
        
        # Process remaining tasks in parallel with reduced timeout
        if tasks:
            try:
                # Run all tasks concurrently with timeout
                task_results = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task, _ in tasks], return_exceptions=True),
                    timeout=18.0
                )
                
                # Process results
                for i, (product, _, cache_key) in enumerate(tasks):
                    if i < len(task_results):
                        result = task_results[i]
                        if isinstance(result, Exception):
                            # Handle failed products
                            result = {
                                'product_name': product.get('product_name', str(product)),
                                'short_description': '',
                                'long_description': '',
                                'status': 'failed',
                                'error': str(result)
                            }
                        else:
                            # Cache successful results
                            if result.get('status') == 'success':
                                self.cache[cache_key] = result
                        batch_results.append(result)
                
            except asyncio.TimeoutError:
                logger.warning(f"Batch timeout for {len(tasks)} products, creating empty results")
                # Create failed results for timeout
                for product, _, _ in tasks:
                    batch_results.append({
                        'product_name': product.get('product_name', str(product)),
                        'short_description': '',
                        'long_description': '',
                        'status': 'failed',
                        'error': 'Timeout'
                    })
        
        return batch_results
    
    async def _process_single_product_fast(
        self,
        product_info: dict,
        existing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fast single product processing with optimized concurrent description generation.
        
        Args:
            product_name (str): Name of the product
            
        Returns:
            dict: Product information with generated descriptions
        """
        try:
            product_name = product_info.get("product_name", "")
            category = product_info.get("category", None)
            # Generate both descriptions concurrently for speed
            retry_payload = self._build_retry_context(product_name, existing_context)

            short_task = self.llm_client.generate_description(
                product_name,
                'short',
                category,
                retry_context=retry_payload['short']
            )
            long_task = self.llm_client.generate_description(
                product_name,
                'long',
                category,
                retry_context=retry_payload['long']
            )

            # Wait for both with timeout
            short_desc, long_desc = await asyncio.wait_for(
                asyncio.gather(short_task, long_task, return_exceptions=True),
                timeout=15.0
            )
            
            # Handle any exceptions in description generation
            if isinstance(short_desc, Exception):
                short_desc = ''
                logger.warning(f"Short description failed for {product_name}: {short_desc}")
            
            if isinstance(long_desc, Exception):
                long_desc = ''
                logger.warning(f"Long description failed for {product_name}: {long_desc}")
            
            # Clean and validate descriptions using the improved cleaning method
            short_desc = str(short_desc).strip() if short_desc else ''
            long_desc = str(long_desc).strip() if long_desc else ''
            
            # Apply the same thorough cleaning as the main method
            if short_desc:
                short_desc = self.llm_client._clean_response(short_desc)
                
                # Ensure bullet format for short descriptions
                lines = short_desc.split('\n')
                bullet_lines = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # Remove any asterisks first
                        line = line.replace('*', '')
                        
                        # Add bullet if missing
                        if not line.startswith('•'):
                            line = '• ' + line
                        
                        bullet_lines.append(line)
                short_desc = '\n'.join(bullet_lines)
            
            if long_desc:
                # Apply thorough cleaning to long description
                long_desc = self.llm_client._clean_response(long_desc)
                # EXTRA asterisk removal for long descriptions
                long_desc = long_desc.replace('*', '')
                
                # Check if long description is incomplete (doesn't end with proper punctuation)
                if long_desc and not long_desc.rstrip().endswith(('.', '!', '?')):
                    logger.warning(f"Long description for '{product_name}' may be incomplete (doesn't end with punctuation): ...{long_desc[-50:]}")
            
            # Require BOTH descriptions to be present for success
            short_completed = bool(short_desc and short_desc.strip())
            long_completed = bool(long_desc and long_desc.strip())
            has_both = short_completed and long_completed

            result = {
                'product_name': product_name,
                'short_description': short_desc,
                'long_description': long_desc,
                'status': 'success' if has_both else 'failed',
                'attempt': retry_payload['meta']['attempt'],
                'short_completed': short_completed,
                'long_completed': long_completed
            }

            if not has_both:
                missing_parts = []
                if not short_completed:
                    missing_parts.append('short')
                if not long_completed:
                    missing_parts.append('long')
                if missing_parts and 'error' not in result:
                    result['error'] = f"Missing {' & '.join(missing_parts)} description"

            if existing_context and 'errors' in existing_context and existing_context['errors']:
                result.setdefault('errors', existing_context['errors'])

            # Log if only one description is missing
            if short_desc and not long_desc:
                logger.warning(f"Missing long description for {product_name}")
            elif long_desc and not short_desc:
                logger.warning(f"Missing short description for {product_name}")

            normalized = product_name.lower().strip()
            self.retry_metadata[normalized] = {
                'attempt': retry_payload['meta']['attempt'],
                'timestamp': time.time(),
                'short_completed': short_completed,
                'long_completed': long_completed,
                'last_status': result['status'],
                'last_error': result.get('error') or result.get('errors')
            }
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Product processing timeout: {product_name}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed',
                'error': 'Processing timeout'
            }
        except Exception as e:
            logger.error(f"Error processing product {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _process_batch(self, batch: list) -> list:
        """
        Process a single batch of products concurrently.
        
        Args:
            batch (list): List of product names to process
            
        Returns:
            list: Processed results for the batch
        """
        tasks = []
        for product in batch:
            tasks.append(self._process_single_product(product))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_product_enhanced(self, product_name: str) -> Dict[str, Any]:
        """
        Enhanced single product processing with better error handling and validation.
        
        Args:
            product_name (str): Name of the product
            
        Returns:
            dict: Product information with generated descriptions
        """
        # Check cache first
        cache_key = product_name.lower().strip()
        if cache_key in self.cache:
            logger.debug(f"Using cached result for: {product_name}")
            return self.cache[cache_key]
        
        try:
            # Generate descriptions with timeout and validation
            short_desc = await asyncio.wait_for(
                self.llm_client.generate_description(product_name, 'short'),
                timeout=120.0  # 2 minute timeout
            )
            
            long_desc = await asyncio.wait_for(
                self.llm_client.generate_description(product_name, 'long'),
                timeout=180.0  # 3 minute timeout for longer descriptions
            )
            
            # Validate descriptions
            short_valid = short_desc and len(short_desc.strip()) > 10
            long_valid = long_desc and len(long_desc.strip()) > 20
            
            # Determine status
            if short_valid and long_valid:
                status = 'success'
                self.rate_limiter.record_success()
            elif short_valid or long_valid:
                status = 'partial'
            else:
                status = 'failed'
                logger.warning(f"Both descriptions failed for: {product_name}")
            
            result = {
                'product_name': product_name,
                'short_description': short_desc if short_valid else '',
                'long_description': long_desc if long_valid else '',
                'status': status
            }
            
            # Cache successful and partial results
            if status in ['success', 'partial']:
                self.cache[cache_key] = result
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing {product_name}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }
        except Exception as e:
            logger.error(f"Error processing {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }

    async def _process_single_product(self, product_name: str) -> Dict[str, Any]:
        """
        Process a single product and generate both descriptions.
        
        Args:
            product_name (str): Name of the product
            
        Returns:
            dict: Product information with generated descriptions
        """
        # Check cache first
        cache_key = product_name.lower().strip()
        if cache_key in self.cache:
            logger.info(f"Using cached result for: {product_name}")
            return self.cache[cache_key]
        
        try:
            # Generate both short and long descriptions concurrently
            short_task = self.llm_client.generate_description(product_name, 'short')
            long_task = self.llm_client.generate_description(product_name, 'long')
            
            short_desc, long_desc = await asyncio.gather(short_task, long_task)
            
            result = {
                'product_name': product_name,
                'short_description': short_desc,
                'long_description': long_desc,
                'status': 'success' if short_desc and long_desc else 'partial'
            }
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }
    
    async def _retry_failed_products(self, failed_products: list, stop_check=None) -> list:
        """
        Retry failed products individually with more conservative approach.
        
        Args:
            failed_products (list): List of failed product names
            stop_check (callable): Optional function to check if processing should stop
            
        Returns:
            list: Results from retry attempts
        """
        retry_results = []
        
        for product_name in failed_products[:50]:  # Limit retries
            if stop_check and stop_check():
                break
                
            # Wait longer between individual retries
            await asyncio.sleep(2.0)
            
            try:
                result = await self._process_single_product_enhanced(product_name)
                if result.get('status') != 'failed':
                    retry_results.append(result)
                    logger.info(f"Successfully retried: {product_name}")
                else:
                    logger.warning(f"Retry failed for: {product_name}")
            except Exception as e:
                logger.error(f"Exception during retry for {product_name}: {str(e)}")
                
        return retry_results
