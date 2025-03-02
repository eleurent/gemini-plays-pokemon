import os
import google.generativeai as genai
import dataclasses


PROMPT_PREFIX = """You are playing Pokemon Red. You have access to the following actions:

[A, B, UP, DOWN, LEFT, RIGHT]

What is your next action?
"""
@dataclasses.dataclass
class GeminiAgent():
    model_name: str = 'models/gemini-2.0-flash-latest'

    def __post_init__(self):
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def act(self, frame):
        contents = self.prompt(frame)
        response = self.model.generate_content(contents)
        return response.text
    
    def prompt(self, frame):
        pass