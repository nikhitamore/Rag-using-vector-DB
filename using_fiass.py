import os
import yaml
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load configuration from file
config_file_path = "/Users/nikhitamore/Documents/Rag-using-vector-DB/config.yaml"  # Update with your configuration file path

def load_config(config_file_path):
    """
    Load configuration from YAML file.

    :param config_file_path: str - Path to the configuration file.
    :return: dict - Configuration dictionary.
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file_path)

# Extract configuration values
md_files_folder = config.get('md_files_folder', '')
vector_database_path = config.get('vector_database_path', '')
pretrained_model = config.get('rag_model', {}).get('pretrained_model', '')
example_query = config.get('example_query', '')

# Initialize RAG model
RAG = RAGPretrainedModel.from_pretrained(pretrained_model)

def read_md_files(folder_path):
    """
    Read and return the content of all Markdown files in a folder.

    :param folder_path: str - Path to the folder containing Markdown files.
    :return: list - List of tuples containing filename and content.
    """
    md_files_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                md_files_content.append((filename, content))
    return md_files_content

def index_md_files(md_files_content):
    """
    Index Markdown files using RAG model.

    :param md_files_content: list - List of tuples containing filename and content.
    """
    for filename, content in md_files_content:
        RAG.index(
            collection=[content],
            document_ids=[filename],
            document_metadatas=[{"entity": "document", "source": "Markdown"}],
            index_name=filename,
            max_document_length=180,  # You can adjust this according to your document length
            split_documents=True
        )

def query_index(query, k=10):
    """
    Query the indexed documents using RAG model.

    :param query: str - Query string.
    :param k: int - Number of documents to retrieve.
    :return: list - List of search results.
    """
    results = RAG.search(query=query, k=k)
    return results

# Read Markdown files
md_files_content = read_md_files(md_files_folder)

# Index Markdown files
index_md_files(md_files_content)

# Generate vector database
loader = TextLoader(md_files_folder)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")  # Specify your desired Hugging Face model here
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

# Query the indexed documents using the example query
results = query_index(example_query)
print(results)
