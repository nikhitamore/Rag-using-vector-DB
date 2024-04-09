import os
import yaml
from ragatouille import RAGPretrainedModel

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
            max_document_length=200,  # You can adjust this according to your document length
            split_documents=True
        )


