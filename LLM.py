from openai import OpenAI

class LargeLanguageModel:
    def __init__(self, api_key=None, model=None):
        if api_key is None:
            # Defaults to getting API key from environment variable: os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)            
        
        if model is None:
            self.model = "gpt-3.5-turbo-0125"
        else:
            self.model = model

        self.messages = []
        self.add_message("system", "You control a robot with a camera on the end-effector.\\" + 
                        "Available actions: [A: Pick up item 'X', B: Equip tool 'X'. C: Hand item 'X' to user. D: Move end-effector to view the workbench from a new camera angle. E: Capture image]. \\" + 
                        "You must select an appropriate sequence of actions according to the user input. \\" + 
                        "Prompt example: 'Visible items: 1. Hammer 2. Glue 3. Screwdriver 4. Caliper 5. Scissors. Give me a tool to fix this chair, and double-check the workbench for hidden items.' \\" +
                        "Your reply: 'Sequence of actions: [B(screwdriver),C(screwdriver),D,E] \\" + 
                        "If no actions are appropriate, return 'None'.\\")

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
LLM = LargeLanguageModel()
user_input = input("Enter your request: ")
message = "Visible items: 1. Hammer 2. Glue 3. Screwdriver 4. Caliper 5. Scissors. " + user_input
response = LLM.generate_response(message)
print("Generated Response:", response)