import os
import google.generativeai as genai
import dataclasses
import PIL.Image


PROMPT_PREFIX = """You are playing Pokemon Red."""

PROMPT = """You have access to the following actions:

[A, B, UP, DOWN, LEFT, RIGHT]

What is your next action? Reply in this format:
Action: <the_action>
"""




@dataclasses.dataclass
class GeminiAgent():
    model_name: str = 'models/gemini-2.0-flash-001'
    valid_actions: list[str] = (
        'DOWN',
        'LEFT',
        'RIGHT',
        'UP',
        'A',
        'B',
        'START',
    )

    def __post_init__(self):
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)
        self.valid_actions = list(self.valid_actions)

    def act(self, frame):
        contents = self.prompt(frame)
        response = self.model.generate_content(contents)
        return self.parse(response)
    
    def parse(self, response):
        action = response.text.split('Action: ')[1].strip()
        return self.valid_actions.index(action)
    
    def prompt(self, frame):
        frame = PIL.Image.fromarray(frame)
        return (frame, PROMPT_PREFIX)