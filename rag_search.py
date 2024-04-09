from ragatouille import RAGPretrainedModel

query = "What is workflow 3.0"
RAG = RAGPretrainedModel.from_index("/Users/nikhitamore/Documents/Rag-using-vector-DB/.ragatouille/colbert/indexes/Introduction to SPAship Workflow 3.0.md")
results = RAG.search(query, k=3)
print("&************",results)