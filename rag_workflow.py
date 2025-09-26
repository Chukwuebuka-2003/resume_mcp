import os
from google import genai
from google.genai import types

import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use PyMuPDF as the primary PDF library
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not found. Please install it with: pip install PyMuPDF")
    exit(1)

# Load environment variables from .env file
load_dotenv()

# Configure the Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# CONFIGURATION
RESUME_PDF_PATH = "data/Chukwuebuka Micheal Ezeokeke's Resume.pdf"

if os.path.exists(RESUME_PDF_PATH):
    PDF_PATH = RESUME_PDF_PATH
    print(f"Using resume PDF: {PDF_PATH}")
else:
    PDF_PATH = RESUME_PDF_PATH
    print(f"Warning: Resume PDF not found. Using default path: {PDF_PATH}")
    print("Please ensure the PDF file exists at this location.")

CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")  # Path to store the Chroma database

# Create data and chroma_db directories if they don't exist
os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

COLLECTION_NAME = "resume_collection"  # Name of the collection in ChromaDB
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"  # Updated embedding model
GENERATIVE_MODEL = "gemini-2.5-flash"  # Gemini as LLM for answer generation

# INGESTION: LOAD AND PROCESS THE PDF
def load_and_split_pdf(file_path):
    """
    Loads a PDF, extracts text using PyMuPDF, and splits it into manageable chunks.
    """
    print(f"Loading and splitting PDF: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return []

    try:
        doc = fitz.open(file_path)
        text = ""

        print(f"PDF has {len(doc)} pages")

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
                    print(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
                else:
                    print(f"No text found on page {page_num + 1}")
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                continue

        doc.close()

        if not text.strip():
            print("Warning: No text extracted from the PDF.")
            return []

        print(f"Total extracted text: {len(text)} characters")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        print(f"Successfully split the document into {len(chunks)} chunks.")

        if chunks:
            print(f"First chunk preview: {chunks[0][:200]}...")

        return chunks

    except Exception as e:
        print(f"Error loading PDF with PyMuPDF: {e}")
        return []

def setup_chroma_db():
    """
    Sets up a persistent ChromaDB client and creates/gets a collection.
    """
    print("Setting up ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"ChromaDB collection '{COLLECTION_NAME}' is ready.")
        return collection
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        raise

def embed_and_store(chunks, collection):
    """
    Embeds text chunks and stores them in the ChromaDB collection.
    """
    if not chunks:
        print("No chunks to embed and store.")
        return

    print(f"Embedding and storing {len(chunks)} chunks...")

    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Collection already contains {existing_count} items. Clearing existing data to re-index...")
            existing_data = collection.get()
            if existing_data['ids']:
                collection.delete(ids=existing_data['ids'])
                print("Cleared existing data from the collection.")

        embeddings = []
        valid_chunks = []

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                print(f"Skipping empty chunk {i+1}")
                continue

            try:
                print(f"Embedding chunk {i+1}/{len(chunks)}")
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings.append(result.embeddings[0].values)
                valid_chunks.append(chunk)
            except Exception as e:
                print(f"Error embedding chunk {i+1}: {e}")
                continue

        if not embeddings:
            print("No valid embeddings generated.")
            return

        ids = [f"chunk_{i}" for i in range(len(valid_chunks))]
        metadatas = [{'chunk_index': i, 'chunk_length': len(chunk)} for i, chunk in enumerate(valid_chunks)]

        collection.add(
            embeddings=embeddings,
            documents=valid_chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully embedded and stored {len(valid_chunks)} chunks in ChromaDB.")

    except Exception as e:
        print(f"An error occurred during embedding or storage: {e}")
        raise

def retrieve_relevant_chunks(question, collection, n_results=3):
    """
    Retrieves the most relevant text chunks from ChromaDB for a given question.
    """
    print(f"Retrieving relevant chunks for question: '{question}'")

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        question_embedding = result.embeddings[0].values
        print("Query embedding generated successfully")

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )

        retrieved_chunks = results['documents'][0] if results['documents'] else []

        if retrieved_chunks and any(chunk is None for chunk in retrieved_chunks):
            print("Some documents are None, filtering them out...")
            retrieved_chunks = [chunk for chunk in retrieved_chunks if chunk is not None]

        if not retrieved_chunks:
            print("No relevant chunks found.")
            return []

        print(f"Found {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def generate_answer(question, context_chunks):
    """
    Uses Gemini to generate an answer based on the question and retrieved context.
    """
    print("Generating answer with Gemini...")

    if not context_chunks:
        return "No relevant context found to answer your question."

    try:
        separator = "\n---\n"
        context_text = separator.join(context_chunks)

        instruction = f"""You are an AI assistant helping to answer questions about a resume/CV document.
        Use only the information from the provided context to answer the question accurately and concisely.
        If the context lacks sufficient information to answer, reply with: "The information is not available in the provided context."
        Do not make assumptions or use external knowledge beyond the document.

        The context contains information from a resume including personal details, work experience, education, skills, and projects.

        Context:
        {context_text}"""

        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=instruction
            ),
            contents=question
        )
        print("Successfully generated answer.")
        return response.text

    except Exception as e:
        print(f"An error occurred during answer generation: {e}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

def main():
    """
    The main function to run the RAG workflow for the resume.
    """
    print("Starting Resume RAG workflow...")

    text_chunks = load_and_split_pdf(PDF_PATH)
    if text_chunks:
        collection = setup_chroma_db()
        embed_and_store(text_chunks, collection)
    else:
        print("Could not process the PDF. Please check the file path and content.")
        return

    print("\n--- Resume Indexed. Ready for Questions ---")
    print("Type 'exit' to quit the program.")

    while True:
        try:
            user_question = input("\nPlease ask a question about the resume: ").strip()
            if user_question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if not user_question:
                print("Please enter a valid question.")
                continue

            relevant_chunks = retrieve_relevant_chunks(user_question, collection)
            answer = generate_answer(user_question, relevant_chunks)

            print("\n--- Answer ---")
            print(answer)
            print("--------------\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
