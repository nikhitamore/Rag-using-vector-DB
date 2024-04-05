import os
import markdown
import yaml
import fiass
from ragatouille import RAGPretrainedModel
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Function to read MD files from a folder
def read_md_files(folder_path):
    md_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                md_files.append({"filename": file, "content": f.read()})
    return md_files

# Function to convert MD content to plain text
def md_to_text(md_content):
    return markdown.markdown(md_content)

# Function to generate vectors for documents and store them in a FIASS database
def generate_vector_database(md_files, vector_db_path):
    vector_db = FAISS(vector_db_path)  # Initialize FAISS vector store
    for file_data in md_files:
        # Convert MD content to plain text
        text_content = md_to_text(file_data["content"])
        # Split text into tokens
        token_text_splitter = TokenTextSplitter()
        tokens = token_text_splitter.split(text_content)
        # Generate embeddings for tokens
        embeddings = HuggingFaceEmbeddings("bert-base-uncased").embed(tokens)  # Use BERT embeddings
        # Add vectors to the database with metadata
        vector_db.add(file_data["filename"], embeddings)
    return vector_db

# Function to search for the best similar answer using RAGATOUILLE
def search_similar_answer(query, rag_model, vector_db):
    results = rag_model.search(query=query, vector_db=vector_db, k=1)
    return results

def main():
    # Load configuration from YAML file
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Access variables from the configuration
    md_files_folder = config["md_files_folder"]
    vector_database_path = config["vector_database_path"]
    rag_model_pretrained_model = config["rag_model"]["pretrained_model"]
    example_query = config["example_query"]

    # Initialize RAGATOUILLE model
    rag_model = RAGPretrainedModel.from_pretrained(rag_model_pretrained_model)

    # Read MD files
    md_files = read_md_files(md_files_folder)

    # Generate vector database
    vector_db = generate_vector_database(md_files, vector_database_path)

    # Search for similar answer to example query
    result = search_similar_answer(example_query, rag_model, vector_db)
    if result:
        print("Best similar answer:", result[0]["document_metadata"]["filename"])
    else:
        print("No similar answer found.")

if __name__ == "__main__":
    main()
