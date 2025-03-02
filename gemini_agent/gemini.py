import os
import google.generativeai as genai
import dataclasses
import PIL.Image
import collections


PROMPT_PREFIX = """You are playing Pokemon Red. Use the default player names, don't use a custom name. """

HISTORY_PROMPT = """
ALSO, here is the history of your previous actions:
"""

PROMPT = """
NOW HERE IS THE SITUATION.
You have access to the following actions:

[A, B, UP, DOWN, LEFT, RIGHT]

What is your next action? Start with a justification, and then reply in this format:
Action: <the_action>

(If you notice that you played the same action multiple times without success, maybe try a different action).
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
    max_history_len: int = 3

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
        try:
            response = self.model.generate_content(contents)
            self.frames_history.append(frame)
            self.responses_history.append(response.text)
            return self.parse(response), response.text
        except Exception as e:
            return self.valid_actions.index('A'), f"Error: {e}"
    
    def parse(self, response):
        action = response.text.split('Action: ')[1].strip()
        if action not in self.valid_actions:
            action = 'A'
        return self.valid_actions.index(action)
    
    def prompt(self, frame):
        turns = list(zip(self.frames_history, self.responses_history))
        flattened_turns = tuple([item for sublist in turns for item in sublist])
        return (PROMPT_PREFIX,) + (frame, PROMPT, HISTORY_PROMPT) + flattened_turns + (PROMPT, frame)