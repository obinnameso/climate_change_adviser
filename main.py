from helpers import *

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("Mistral API key is not set. Please set the 'mistral_api_key' environment variable.")



def chatbot():
    
    try:
        # Load and process documents
        file_path = "data/Aristotle_on_catharsis.pdf"
        print("Loading documents...")
        docs = load_documents(file_path)

        print("Chunking documents...")
        documents = chunk_documents(docs)
        print(f"Created {len(documents)} chunks.")

        # Create vector store
        print("Creating vector store...")
        vector_store = create_vector_store(documents)

        # Create retrieval chain
        print("Setting up retrieval chain...")
        retrieval_chain = create_retrieval_chain_with_prompt(vector_store, API_KEY)

        # Initialize conversation history
        conversation_history = ""

        print("\nLily is ready. Type 'end' to terminate the conversation.")
        while True:
            # Get user input
            user_input = input("\nYou: ")

            if user_input.strip().lower() == "end":
                print("\nLily: It was nice chatting with you. Goodbye!")
                break

            # Combine conversation history with the current input
            conversation_with_history = f"{conversation_history}\nUser: {user_input}"

            # Get the response from the retrieval chain
            response = retrieval_chain.invoke({"input": conversation_with_history})

            if "answer" not in response:
                print("\nLily: I'm sorry, I couldn't process your question.")
            else:
                answer = response["answer"]
                print(f"\nLily: {answer}")

                # Append the latest interaction to the conversation history
                conversation_history += f"\nUser: {user_input}\nLily: {answer}"

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    chatbot()