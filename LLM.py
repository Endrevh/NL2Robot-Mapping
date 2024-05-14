from openai import OpenAI

class LargeLanguageModel:
    def __init__(self, api_key=None, model=None):
        if api_key is None:
            # Defaults to getting API key from environment variable: os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)            
        
        if model is None:
            self.model = "gpt-3.5-turbo"
        else:
            self.model = model

        self.messages = []

    @property
    def chat_messages(self):
        # Convert the list of messages into the required format for ChatGPT API
        return [{"role": message["role"], "content": message["content"]} for message in self.messages]

    def add_message(self, role, content):
        # Add a new message to the list
        self.messages.append({"role": role, "content": content})

    def generate_response(self, message):
        # Add user message to the list
        self.add_message("user", message)

        # Send messages to ChatGPT API and get a response
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.chat_messages
        )
        
        gpt_response = response.choices[0].message.content

        # Add assistant message to messages list
        self.add_message("assistant", gpt_response)

        return gpt_response
    

# Example usage
#API_key = None # Replace with your API key if you have one
#lang_model = None
#LLM = LargeLanguageModel(api_key=API_key, model=lang_model)
"""LLM.add_message("system", "You are my assistant and are in control of a robot with a camera mounted on the end-effector. The camera can be used to scan for items on a workbench. \\" +
                "The robot is able to perform the following actions: [A: Pick up item 'X' from the workbench. B: Give item 'X' to the operator. C: Equip item 'X'. D: Move end-effector to view workbench from new camera angle. E: Capture image from current camera angle].\\" +
                "I, the operator, will tell you which items are visible on the workbench, and provide you with a task that I need done." + 
                "Your job is to create a sequence of actions from the list above which fullfils the task. \\" +
                "Your response should formatted as a list of the selected actions, and which items from the workbench you decide to use (if relevant). If no actions apply, return 'None'.")"""

#user_input = input("Enter your request: ")
#message = "Visible items: 1. Hammer 2. Glue 3. Screwdriver 4. Caliper 5. Scissors. " + user_input
#message = "Visible items: 1: Tie. 2: Frisbee. 3: Bottle. 4: Fork. 5: Knife. 6: Scissors. 7: Toothbrush. " + user_input
#message = "Visible items: 1: Tie. 2: Frisbee. 3: Bottle. 4: Scissors. 5: Toothbrush. " + user_input

#response = LLM.generate_response(message)
#print("Generated Response:", response)
#print("Messages:", LLM.messages)