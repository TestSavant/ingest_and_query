# Importing necessary libraries and modules
import os
from typing import List, Tuple, Union
import logging

# Importing custom modules
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

# Importing constants
from consts import pinecone_db, DOCUMENT_MAP, SOURCE_DIRECTORY
import argparse

# Setting up command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default=SOURCE_DIRECTORY)  # Argument for source directory
parser.add_argument(
    "--device_type",
    type=str,
    default="cpu",
    choices=["cuda", "cpu", "hip"],  # Argument for device type
    help="The compute power that you have",
)

# Function to load a single document
def load_single_document(file_path: str) -> Document:
    """
    Load one document from the source documents directory
    """
    # Get file extension
    file_extension = os.path.splitext(file_path)[1]
    try:
        # Get the appropriate loader class for the file extension
        loader_class = DOCUMENT_MAP[file_extension]
    except KeyError:
        # Raise error if file extension is not supported
        raise KeyError(f"File extension {file_extension} is not supported")
    finally:
        pass

    # Instantiate the loader and load the document
    loader = loader_class(file_path)
    return loader.load()[0]

# Function to load all documents in a directory
def load_document(source_dir: str) -> List[Document]:
    """
    Load all documents from the source documents directory
    """
    # List all files in the directory
    all_files = os.listdir(source_dir)
    documents = []

    # Load each document
    for file in all_files:
        source_file_path = os.path.join(source_dir, file)
        documents.append(load_single_document(source_file_path))

    return documents

# Main function
def main():
    # Parse command line arguments
    args = parser.parse_args()

    # Load documents
    logging.info(f"Loading documents from {args.source_dir}")
    documents = load_document(args.source_dir)

    # Split documents into chunks and process text
    logging.info("Splitting documents into chunks and processing text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    # Create embeddings for the documents
    logging.info("Creating embedding for the documents")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": args.device_type}
    )

    # Create a vector store index
    index_name = "flowise"
    Pinecone.from_documents(
        texts,
        embeddings,
        index_name=index_name,
    )

    logging.info("Finished create vectorDB index")

# Entry point of the script
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # Call the main function
    main()
