# Importing necessary libraries and modules
import os
import argparse
import logging
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from typing import List, Tuple, Union
from consts import PINECONE_SETTINGS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

# Setting up command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device_type",
    type=str,
    default="cpu",
    choices=["cuda", "cpu", "hip"],  # Argument for device type
    help="The compute power that you have",
)

# Function to load a model from OpenAI
def load_model():
    """
    Select a model from huggingface.
    """
    language_model = OpenAI(
        model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY
    )

    return language_model

# Main function
def main():
    # Parse command line arguments
    args = parser.parse_args()
    logging.info(f"Running on: {args.device_type}")

    # Create embeddings for the documents
    logging.info("Creating embedding for the documents")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": args.device_type}
    )

    index_name = "flowise"

    # Initialize Pinecone
    logging.info("Consume Pinecone VectorDB ")
    pinecone.init(**PINECONE_SETTINGS)
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    # Load the language model
    llm = load_model()

    # Create a RetrievalQA object
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    # Open a log file to write user queries
    with open("user_input.log", "w") as file:
        while True:
            # Get user query
            query = input("\nEnter a question: ")

            # Write the query to the log file
            file.write(query + "\n")

            # Break the loop if the user types "quit"
            if query == "quit":
                break

            # Get the answer to the query
            answer = qa.run(query=query)

            # Print the question and answer
            print(f"\n\n > Question:")
            print(query)
            print(f"\n\n > Answer:")
            print(answer)

# Entry point of the script
if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )
        # Call the main function
        main()
    except:
        # Raise an error if something goes wrong
        raise RuntimeError("Something went wrong")
