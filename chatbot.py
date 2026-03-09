import json
import nltk
import requests
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import wikipediaapi
import os
import re
import ast
import operator

# Optional OpenAI integration: only used if `OPENAI_API_KEY` is set in env
try:
    import openai
except Exception:
    openai = None

# Hugging Face placeholder - we will call the Inference API when `HF_TOKEN` is set
HF_CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_HF_MODEL = os.environ.get('HF_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')

# Try to load required NLTK data; if not present, it will be downloaded in app.py
lemmatizer = WordNetLemmatizer()

SAFE_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

SKILL_QUESTION_BANK = {
    'python': [
        'What are lists, tuples, sets, and dictionaries in Python?',
        'Explain the difference between a list and a tuple in Python.',
        'What are decorators in Python?',
        'How does exception handling work in Python?',
    ],
    'flask': [
        'What is Flask and why would you use it?',
        'How do routing, templates, and request handling work in Flask?',
        'How do you build and consume REST APIs in Flask?',
    ],
    'django': [
        'What is the difference between Django and Flask?',
        'How does Django ORM work?',
    ],
    'sql': [
        'What is the difference between INNER JOIN, LEFT JOIN, and RIGHT JOIN?',
        'What is normalization in databases?',
        'How do GROUP BY and HAVING work in SQL?',
    ],
    'mysql': [
        'How do you optimize slow MySQL queries?',
        'What indexes are and when would you use them in MySQL?',
    ],
    'html': [
        'What is the role of semantic HTML?',
        'What is the difference between block and inline elements?',
    ],
    'css': [
        'What is the difference between Flexbox and Grid?',
        'How does CSS specificity work?',
    ],
    'javascript': [
        'What is the difference between var, let, and const?',
        'Explain promises and async/await in JavaScript.',
    ],
    'react': [
        'What are props and state in React?',
        'What is the use of hooks such as useState and useEffect?',
    ],
    'pandas': [
        'What is a DataFrame in pandas?',
        'How do you handle missing values in pandas?',
    ],
    'numpy': [
        'What is the difference between a Python list and a NumPy array?',
        'Why is NumPy faster for numerical computation?',
    ],
    'machine learning': [
        'What is the difference between supervised and unsupervised learning?',
        'What is overfitting and how do you reduce it?',
        'How do you evaluate a machine learning model?',
    ],
    'deep learning': [
        'What is the difference between machine learning and deep learning?',
        'What is a neural network?',
        'What are activation functions in deep learning?',
    ],
    'data structures': [
        'What is the difference between a stack and a queue?',
        'When would you use an array versus a linked list?',
    ],
    'algorithms': [
        'What is time complexity and what does Big O notation mean?',
        'Explain the difference between linear search and binary search.',
    ],
    'git': [
        'What is the difference between git merge and git rebase?',
        'How do you resolve merge conflicts in Git?',
    ],
    'github': [
        'How do pull requests and code reviews work in GitHub?',
        'How do you collaborate on a shared repository using GitHub?',
    ],
}

GENERIC_PROJECT_QUESTIONS = [
    'Explain one project from your resume and your exact role in it.',
    'What problem was the project solving, and who were the users?',
    'What technology stack did you choose for the project and why?',
    'What was the biggest technical challenge in the project and how did you solve it?',
    'How did you design the backend, database, or API structure for the project?',
    'What improvements would you make if you had more time on that project?',
]

GENERIC_HR_QUESTIONS = [
    'Tell me about yourself.',
    'Walk me through your resume.',
    'Why do you want to work for this company?',
    'What are your strengths and weaknesses?',
    'Why should we hire you?',
    'Describe a challenge you faced and how you handled it.',
    'Where do you see yourself in the next 3 to 5 years?',
    'How do you handle pressure or tight deadlines?',
    'Describe a time you worked in a team.',
    'What motivates you in your career?',
]

class ChatBot:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file
        self.intents = []
        self.patterns = []
        self.responses = []
        self.tags = []
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_lemmatize, stop_words='english')
        self.tfidf_matrix = None
        self._load_intents()
        self._train_model()
        
        # Initialize Wikipedia API client (no auth required)
        self.wiki = wikipediaapi.Wikipedia(user_agent='PyChatbot/1.0 (educational-project)', language='en')
        
    def _load_intents(self):
        """Loads the intents from the JSON file."""
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for intent in data['intents']:
            for pattern in intent['patterns']:
                self.patterns.append(pattern)
                self.responses.append(intent['responses'])
                self.tags.append(intent['tag'])
                self.intents.append(intent)

    def tokenize_and_lemmatize(self, text):
        """Tokenizes and lemmatizes the input text."""
        try:
            tokens = nltk.word_tokenize(text.lower())
        except LookupError:
            tokens = text.lower().split()
        return [lemmatizer.lemmatize(word) for word in tokens]

    def _is_knowledge_query(self, text):
        """Heuristic to detect factual/general queries that should prefer generative fallback."""
        normalized = text.strip().lower()
        starters = (
            'what is', 'who is', 'where is', 'when is', 'why', 'how',
            'what does', 'what are', 'can you explain', 'define', 'explain', 'tell me about', 'name one', 'list',
            'difference between', 'compare', 'application of', 'applications of'
        )
        return normalized.endswith('?') or normalized.startswith(starters)

    def _split_compound_questions(self, text):
        """Split a message containing multiple questions into smaller prompts."""
        normalized = re.sub(r'\s+', ' ', text.strip())
        if not normalized:
            return []

        parts = re.split(
            r'(?<=[?.!])\s+(?=(?:what|how|why|who|where|when|define|explain|tell|write|name|compare|list|can)\b)',
            normalized,
            flags=re.IGNORECASE,
        )

        questions = []
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                questions.append(cleaned)
        return questions

    def _extract_quoted_prompts(self, text):
        """Extract multiple quoted prompts from a single message."""
        matches = re.findall(r'"([^"]+)"', text)
        cleaned = [match.strip() for match in matches if match.strip()]
        return cleaned if len(cleaned) > 1 else []

    def _safe_eval_math(self, expression):
        """Safely evaluate a small arithmetic expression."""
        def _eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in SAFE_MATH_OPERATORS:
                return SAFE_MATH_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_MATH_OPERATORS:
                return SAFE_MATH_OPERATORS[type(node.op)](_eval_node(node.operand))
            raise ValueError('Unsupported expression')

        parsed = ast.parse(expression, mode='eval')
        return _eval_node(parsed.body)

    def _get_math_response(self, text):
        """Return an answer for simple arithmetic questions if present."""
        normalized = text.strip().lower().rstrip('?.!')
        match = re.search(r'(?:what is|calculate|solve)\s+([-+*/%()\d\s.]+)$', normalized)
        if not match:
            return None

        expression = match.group(1).strip()
        if not re.fullmatch(r'[-+*/%()\d\s.]+', expression):
            return None

        try:
            result = self._safe_eval_math(expression)
        except Exception:
            return None

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"The answer is {result}."

    def _build_model_messages(self, user_input, context_text=None, context_name=None):
        """Build chat messages for generative backends, optionally grounding on uploaded file content."""
        system_content = (
            "You are a helpful assistant. Answer the exact user question clearly and concisely. "
            "If a term has multiple meanings, use the user's context."
        )

        if context_text:
            excerpt = context_text[:12000]
            user_content = (
                f"Use the uploaded file content to answer the question. "
                f"If the answer is not in the file, say that briefly.\n\n"
                f"File: {context_name or 'uploaded file'}\n"
                f"Content:\n{excerpt}\n\n"
                f"Question: {user_input}"
            )
        else:
            user_content = user_input

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _extract_candidate_name(self, context_text):
        """Best-effort name extraction from uploaded resume text."""
        lines = [line.strip() for line in context_text.splitlines() if line.strip()]
        for line in lines[:8]:
            if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", line):
                return line
        return None

    def _extract_skills_from_context(self, context_text):
        """Extract known technical skills from uploaded text using a curated vocabulary."""
        lowered = context_text.lower()
        ordered_skills = []
        for skill in SKILL_QUESTION_BANK:
            if skill in lowered:
                ordered_skills.append(skill)
        return ordered_skills

    def _extract_project_lines(self, context_text):
        """Extract probable project titles or project bullets from resume text."""
        lines = [line.strip(' -:\t') for line in context_text.splitlines() if line.strip()]
        project_lines = []
        in_project_section = False

        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in ('project', 'projects', 'academic project', 'personal project')):
                in_project_section = True
                if len(line.split()) > 1 and 'project' not in lowered[:12]:
                    project_lines.append(line)
                continue

            if in_project_section and re.fullmatch(r'[A-Z][A-Za-z\s]+', line) and len(line.split()) <= 6:
                project_lines.append(line)
                continue

            if in_project_section and len(project_lines) < 4 and len(line.split()) <= 12:
                project_lines.append(line)

            if in_project_section and len(project_lines) >= 4:
                break

        unique_lines = []
        seen = set()
        for line in project_lines:
            normalized = line.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        return unique_lines[:4]

    def _get_contextual_file_response(self, user_input, context_text, context_name=None):
        """Answer common uploaded-file questions without relying on external models."""
        normalized = user_input.strip().lower()
        candidate_name = self._extract_candidate_name(context_text)
        skills = self._extract_skills_from_context(context_text)
        asks_for_interview_questions = 'interview' in normalized and any(
            term in normalized for term in ('question', 'questions', 'cv', 'resume', 'uploaded file', 'possible')
        )
        asks_for_technical_questions = 'technical question' in normalized or 'technical questions' in normalized
        asks_for_project_questions = 'project' in normalized and 'question' in normalized
        asks_for_hr_questions = 'hr' in normalized and 'question' in normalized

        if asks_for_technical_questions or asks_for_interview_questions:
            questions = []
            for skill in skills[:6]:
                questions.extend(SKILL_QUESTION_BANK.get(skill, []))

            if asks_for_interview_questions:
                questions.extend(GENERIC_PROJECT_QUESTIONS)

            if not questions:
                questions = [
                    'Tell me about yourself and your recent projects.',
                    'What technical skills are you most confident in?',
                    'Explain one project you built and the challenges you solved.',
                    'How do you debug a problem in your code?',
                    'What have you learned recently that improved your development skills?',
                ]

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are likely technical interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are likely technical interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if asks_for_project_questions:
            project_lines = self._extract_project_lines(context_text)
            questions = list(GENERIC_PROJECT_QUESTIONS)

            for skill in skills[:5]:
                questions.append(f'How did you use {skill.title()} in your project work?')

            for project in project_lines:
                questions.append(f'Explain the project "{project}" in detail.')
                questions.append(f'What challenges did you face while building "{project}"?')

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are project-related interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are project-related interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if asks_for_hr_questions:
            questions = list(GENERIC_HR_QUESTIONS)
            if candidate_name:
                questions.insert(1, f'Introduce yourself as {candidate_name} in a concise and professional way.')

            if skills:
                primary_skills = ', '.join(skill.title() for skill in skills[:5])
                questions.append(f'How would you explain your strongest skills: {primary_skills}?')

            project_lines = self._extract_project_lines(context_text)
            for project in project_lines[:2]:
                questions.append(f'Which project are you most proud of, such as "{project}", and why?')

            unique_questions = []
            seen = set()
            for question in questions:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)

            intro = 'Here are HR interview questions based on the uploaded CV:'
            if candidate_name:
                intro = f'Here are HR interview questions for {candidate_name} based on the uploaded CV:'

            lines = [intro]
            for index, question in enumerate(unique_questions[:15], start=1):
                lines.append(f'{index}. {question}')
            return '\n'.join(lines)

        if 'skill' in normalized or 'technology' in normalized:
            if skills:
                formatted = ', '.join(skill.title() for skill in skills[:12])
                return f'The uploaded file mentions these technical skills: {formatted}.'
            return 'I could not confidently identify technical skills in the uploaded file.'

        if 'name' in normalized and candidate_name:
            return f'The candidate name in the uploaded file is {candidate_name}.'

        if 'summary' in normalized or 'summarize' in normalized:
            if skills:
                skill_summary = ', '.join(skill.title() for skill in skills[:8])
                if candidate_name:
                    return f'{candidate_name} appears to have experience with: {skill_summary}.'
                return f'The uploaded file highlights experience with: {skill_summary}.'

        return None

    def _has_extra_topic_tokens(self, user_input, matched_pattern):
        """Detect when a question includes topic words not present in the matched canned pattern."""
        ignore_tokens = {
            'what', 'is', 'are', 'who', 'where', 'when', 'why', 'how',
            'define', 'explain', 'tell', 'me', 'about', 'the', 'a', 'an',
            'in', 'of', 'to', 'for', 'language', '?'
        }
        user_tokens = {t for t in self.tokenize_and_lemmatize(user_input) if t not in ignore_tokens}
        pattern_tokens = {t for t in self.tokenize_and_lemmatize(matched_pattern) if t not in ignore_tokens}
        return len(user_tokens - pattern_tokens) > 0

    def _train_model(self):
        """Fits the TF-IDF vectorizer on the intent patterns."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def _normalize_user_input(self, text):
        """Normalize common abbreviations and typos before matching/fallback."""
        normalized = text.strip()
        normalized = re.sub(r'^\s*waht\b', 'what', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bdeep\s+lern(ing)?\b', 'deep learning', normalized, flags=re.IGNORECASE)

        typo_replacements = {
            r'\bquueu\b': 'queue',
            r'\bqeue\b': 'queue',
            r'\barrary\b': 'array',
            r'\balogrithm\b': 'algorithm',
            r'\boperting\b': 'operating',
        }
        for pattern, replacement in typo_replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Expand common technical abbreviations in question-style queries.
        abbreviation_replacements = {
            r'\bdl\b': 'deep learning',
            r'\bllm\b': 'large language model',
            r'\bdsa\b': 'data structures and algorithms',
            r'\bos\b': 'operating system',
            r'\bml\b': 'machine learning',
            r'\boops\b': 'object oriented programming',
        }
        if re.search(r'\b(what is|define|explain|about|tell me about)\b', normalized, flags=re.IGNORECASE):
            for pattern, replacement in abbreviation_replacements.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        return normalized

    def get_generative_response(self, user_input, context_text=None, context_name=None):
        """Uses Wikipedia's free API (no auth needed) to answer general knowledge questions."""
        if context_text:
            local_response = self._get_contextual_file_response(user_input, context_text, context_name=context_name)
            if local_response:
                return local_response

            messages = self._build_model_messages(user_input, context_text=context_text, context_name=context_name)

            if openai and os.environ.get('OPENAI_API_KEY'):
                try:
                    openai.api_key = os.environ.get('OPENAI_API_KEY')
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=350,
                        temperature=0.2,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as oe:
                    print(f"OpenAI contextual fallback error: {oe}")

            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                try:
                    model = os.environ.get('HF_MODEL', DEFAULT_HF_MODEL)
                    hf_headers = {"Authorization": f"Bearer {hf_token}", "Accept": "application/json"}
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 320,
                        "temperature": 0.2
                    }
                    hf_resp = requests.post(HF_CHAT_API_URL, headers=hf_headers, json=payload, timeout=30)
                    if hf_resp.status_code == 200:
                        data = hf_resp.json()
                        choices = data.get('choices', []) if isinstance(data, dict) else []
                        if choices and isinstance(choices[0], dict):
                            content = choices[0].get('message', {}).get('content')
                            if content:
                                return content.strip()
                    else:
                        print(f"HuggingFace contextual chat API returned {hf_resp.status_code}: {hf_resp.text}")
                except Exception as he:
                    print(f"HuggingFace contextual fallback error: {he}")

            return "I uploaded the file, but I could not generate an answer from it right now."

        try:
            # Clean the query: remove code/programming noise words for better Wikipedia search
            # Keep language keywords (e.g., "python") to preserve question context.
            noise_words = r'\b(code|program|write|example|show me|give me)\b'
            cleaned_input = re.sub(noise_words, '', user_input, flags=re.IGNORECASE).strip()
            search_query = cleaned_input if len(cleaned_input) > 2 else user_input
            print(f"Fallback triggered. Searching Wikipedia for: '{search_query}' (original: '{user_input}')")
            
            # Step 1: Use Wikipedia OpenSearch to find the best matching article title
            headers = {"User-Agent": "PyChatbot/1.0 (educational-project; contact@example.com)"}
            search_res = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": search_query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json"
                },
                headers=headers,
                timeout=8
            )
            search_data = search_res.json()
            
            # Step 2: Fetch article summary if a match was found
            if len(search_data) >= 2 and len(search_data[1]) > 0:
                article_title = search_data[1][0]
                print(f"Best Wikipedia match: '{article_title}'")
                page = self.wiki.page(article_title)
                
                if page.exists() and page.summary:
                    # Return first 3 sentences for a concise, natural answer
                    sentences = page.summary.split('. ')
                    short_answer = '. '.join(sentences[:3]).strip()
                    if not short_answer.endswith('.'):
                        short_answer += '.'
                    return f"📚 {short_answer}"

        except Exception as e:
            print(f"Wikipedia fallback error: {e}")

        # If Wikipedia has no match or fails, try OpenAI (if available)
        if openai and os.environ.get('OPENAI_API_KEY'):
            try:
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self._build_model_messages(user_input),
                    max_tokens=250,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content.strip()
                return text
            except Exception as oe:
                print(f"OpenAI fallback error: {oe}")

        # Next fallback: Hugging Face chat-completions API (if HF_TOKEN available)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            try:
                model = os.environ.get('HF_MODEL', DEFAULT_HF_MODEL)
                hf_headers = {"Authorization": f"Bearer {hf_token}", "Accept": "application/json"}
                payload = {
                    "model": model,
                    "messages": self._build_model_messages(user_input),
                    "max_tokens": 220,
                    "temperature": 0.2
                }
                hf_resp = requests.post(HF_CHAT_API_URL, headers=hf_headers, json=payload, timeout=30)
                if hf_resp.status_code == 200:
                    data = hf_resp.json()
                    choices = data.get('choices', []) if isinstance(data, dict) else []
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get('message', {})
                        content = message.get('content')
                        if content:
                            return content.strip()
                else:
                    print(f"HuggingFace chat API returned {hf_resp.status_code}: {hf_resp.text}")
            except Exception as he:
                print(f"HuggingFace fallback error: {he}")

        return "I wasn't able to find specific information on that. Could you try rephrasing your question?"

    def _get_single_response(self, user_input, context_text=None, context_name=None):
        """Return a response for a single prompt/question."""
        math_response = self._get_math_response(user_input)
        if math_response:
            return math_response

        if context_text:
            return self.get_generative_response(user_input, context_text=context_text, context_name=context_name)

        normalized_input = self._normalize_user_input(user_input)
        user_tfidf = self.vectorizer.transform([normalized_input])
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        best_match_idx = np.argmax(cosine_similarities)
        best_score = cosine_similarities[best_match_idx]
        best_tag = self.tags[best_match_idx]
        
        # Log the similarity score for debugging
        print(f"Input: '{user_input}' | Normalized: '{normalized_input}' | Match: '{best_tag}' | Score: {best_score:.4f}")
        
        # Only use intent-based response if similarity is above a strict threshold
        small_talk_tags = {
            'greeting', 'goodbye', 'thanks', 'about', 'capabilities', 'project_details', 'joke'
        }
        is_knowledge_query = self._is_knowledge_query(normalized_input)
        matched_pattern = self.patterns[best_match_idx]
        has_extra_topic_tokens = self._has_extra_topic_tokens(normalized_input, matched_pattern)

        # For factual queries, require stronger confidence before returning canned intents.
        if is_knowledge_query and (best_tag in small_talk_tags or best_score < 0.78 or has_extra_topic_tokens):
            return self.get_generative_response(normalized_input)

        if best_score > 0.62:
            return np.random.choice(self.responses[best_match_idx])

        # Default to generative fallback for low-confidence matches.
        return self.get_generative_response(normalized_input)

    def get_response(self, user_input, context_text=None, context_name=None):
        """Given user input, find the best matching response."""
        quoted_prompts = self._extract_quoted_prompts(user_input)
        if quoted_prompts:
            answers = []
            for index, prompt in enumerate(quoted_prompts[:8], start=1):
                answer = self._get_single_response(prompt, context_text=context_text, context_name=context_name)
                answers.append(f"{index}. {answer}")
            return "\n\n".join(answers)

        questions = self._split_compound_questions(user_input)
        if len(questions) > 1:
            answers = []
            for index, question in enumerate(questions[:6], start=1):
                answer = self._get_single_response(question, context_text=context_text, context_name=context_name)
                answers.append(f"{index}. {answer}")
            return "\n\n".join(answers)

        return self._get_single_response(user_input, context_text=context_text, context_name=context_name)

# Example usage (for testing)
if __name__ == "__main__":
    bot = ChatBot()
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'quit':
            break
        print("Bot:", bot.get_response(user_msg))
