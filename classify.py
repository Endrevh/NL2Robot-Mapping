from openai import OpenAI


def get_class_from_prompt(user_input, available_classes):
    # Construct a prompt that instructs GPT to generate the most fitting class
    prompt = f"Assign the text below to an appropriate class from {available_classes}. If none apply, return 'None'.\n\n{user_input}\nClass:"
    
    # Use GPT-3.5. Defaults to getting API key from environment variable: os.environ.get("OPENAI_API_KEY")
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages= [{"role": "user", "content": f"{prompt}"}],
        max_tokens=50
    )

    # Extract the generated class from GPT's response
    generated_class = response.choices[0].message.content
    return generated_class

# Example usage
user_input = input("Enter your request: ")
available_classes = ['hammer', 'screwdriver', 'glue', 'scissors']
generated_class = get_class_from_prompt(user_input, available_classes)
print("Generated Class:", generated_class)
