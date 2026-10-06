from importlib.metadata import version
from dotenv import load_dotenv

# env file loaded with all keys are available now.
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print("LangChain Core version:", version("langchain-core"))
print("LangGraph version:", version("langgraph"))

# key = os.getenv("OPENAI_API_KEY")

# if key:
#     print("API key loaded! Starts with:", key[:7])
# else:
#     print("No API key found. Check your .env file.")


def main():

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say, 'setup complete!' in one word.")
    print(f"Response from ChatOpenAI:, {response.content}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
