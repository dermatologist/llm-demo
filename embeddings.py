from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings), len(embeddings[0]))

embedded_query = embeddings_model.embed_query("Neurology?")
print(f" Neurology {embedded_query[:5]}\n")

embedded_query = embeddings_model.embed_query("Dermatology?")
print(f" Dermatology {embedded_query[:5]}\n")