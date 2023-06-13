# ingest.py
This script is primarily used for loading documents from a specified directory, splitting the documents into chunks, creating embeddings for the chunks using a HuggingFace model, and storing the embeddings in a Pinecone vector database. The script takes two command-line arguments: the source directory for the documents and the device type for the HuggingFace model.

# query.py
This script is primarily used for creating a question-answering system. It uses a HuggingFace model to create embeddings for documents, and these embeddings are stored in a Pinecone vector database. The script also uses an OpenAI language model to generate answers to user queries. The user can input their queries in the console, and the script will print the corresponding answers. The script takes a command-line argument for the device type for the HuggingFace model. The user queries are also logged in a file named "user_input.log".


# Here's a description of each of the imported modules and functions:

os: This is a standard Python library that provides functions for interacting with the operating system. It includes functions for file and directory manipulation, reading environment variables, and more.

argparse: This is a standard Python library used for writing user-friendly command-line interfaces. It parses command-line arguments and generates help and usage messages.

logging: This is a standard Python library used for generating logging messages. It provides a flexible framework for emitting log messages from Python programs.

langchain.docstore.document.Document: This is a class from the langchain package to process and store documents

langchain.llms.OpenAI: This is another class from the langchain package. It represents an OpenAI language model.

langchain.embeddings.HuggingFaceInstructEmbeddings: This class from the langchain package represents a HuggingFace model used for generating embeddings.

langchain.vectorstores.Pinecone: This class from the langchain package represents a Pinecone vector database.

typing.List, typing.Tuple, typing.Union: These are classes from the typing module in the Python standard library. They are used for type hinting, which helps in static type checking and improves readability.

consts.PINECONE_SETTINGS: This is a constant from the consts module. It contains settings for Pinecone.

langchain.chains.question_answering.load_qa_chain: This function from the langchain package loads a question-answering model or pipeline.

dotenv.load_dotenv: This function from the python-dotenv package loads environment variables from a .env file into the system environment.

langchain.chains.RetrievalQA: This class from the langchain package represents a retrieval-based question-answering model.

pinecone: This is the Pinecone library, which provides functions for interacting with Pinecone, a vector database service.

langchain.text_splitter.RecursiveCharacterTextSplitter: This class from the langchain package represents a text splitter that splits text into chunks in a recursive manner.



