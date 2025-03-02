import os
import google.generativeai as genai
import dataclasses
import PIL.Image
import collections


PROMPT_PREFIX = """You are playing Pokemon Red. Use the default player names, don't use a custom name."""

PROMPT = """You have access to the following actions:

[A, B, UP, DOWN, LEFT, RIGHT]

What is your next action? Start with a justification, and then reply in this format:
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
    max_history_len: int = 10

    def __post_init__(self):
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)
        self.valid_actions = list(self.valid_actions)
        self.frames_history = collections.deque(maxlen=self.max_history_len)
        self.responses_history = collections.deque(maxlen=self.max_history_len)

    def act(self, frame):
        frame = PIL.Image.fromarray(frame)
        contents = self.prompt(frame)
        response = self.model.generate_content(contents)
        self.frames_history.append(frame)
        self.responses_history.append(response.text)
        return self.parse(response), response.text
    
    def parse(self, response):
        action = response.text.split('Action: ')[1].strip()
        return self.valid_actions.index(action)
    
    def prompt(self, frame):
        user_prompts = [PROMPT] * len(self.frames_history)
        turns = list(zip(self.frames_history, user_prompts, self.responses_history))
        flattened_turns = tuple([item for sublist in turns for item in sublist])
        return (PROMPT_PREFIX,) + flattened_turns + (frame, PROMPT)