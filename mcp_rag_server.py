import sys
import os
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Import the RAG workflow logic as a library
import rag_workflow as rag

# Ensure PyMuPDF is available
try:
    import fitz  # PyMuPDF
    print("PyMuPDF is available")
except ImportError:
    print("PyMuPDF not found. Please install it with: pip install PyMuPDF")
    sys.exit(1)

# Use the same paths as in rag_workflow.py for consistency
PDF_PATH = rag.PDF_PATH
CHROMA_DB_PATH = rag.CHROMA_DB_PATH

# SETUP THE RAG SYSTEM ON STARTUP

def setup_rag_system():
    """
    Initializes the entire RAG system: loads the PDF, sets up ChromaDB,
    and indexes the document if it's not already indexed.
    Returns the ChromaDB collection object needed for querying.
    """
    print("--- Initializing RAG System for MCP Server ---")

    # Check if PDF file exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}. Please check the file path.", file=sys.stderr)
        return None

    # Load and split the PDF
    try:
        text_chunks = rag.load_and_split_pdf(PDF_PATH)
    except Exception as e:
        print(f"Error loading PDF: {e}", file=sys.stderr)
        return None

    # If the PDF is empty or cannot be read, we cannot proceed.
    if not text_chunks:
        print("Error: Could not load text chunks from PDF. Please check if the PDF contains readable text.", file=sys.stderr)
        return None

    # Set up the persistent ChromaDB
    try:
        collection = rag.setup_chroma_db()
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}", file=sys.stderr)
        return None

    # Embed and store the chunks if needed
    try:
        rag.embed_and_store(text_chunks, collection)
    except Exception as e:
        print(f"Error embedding and storing chunks: {e}", file=sys.stderr)
        return None

    print("--- RAG System is Ready ---")
    return collection

# Initialize the RAG system and get the collection object
# This code runs once when the server starts.
rag_collection = setup_rag_system()

# Check if initialization was successful
if rag_collection is None:
    print("Failed to initialize RAG system. Server cannot start properly.", file=sys.stderr)
    print("Please check your PDF path, API keys, and dependencies.", file=sys.stderr)
    # Don't exit immediately - let the server start but tool will return error messages


# DEFINE THE MCP SERVER AND TOOLS

# Instantiate an MCP server client
mcp: FastMCP = FastMCP("RAG Document Q&A Server")

# Define the RAG Q&A functionality as a tool
@mcp.tool()
def ask_document(question: str) -> str:
    """
    Answers questions about a resume/CV document based on the content of the indexed PDF.
    Use this tool to find information about the person's background, experience, skills, education,
    projects, contact information, and other details contained within their resume.
    """
    print(f"\n[MCP Tool Call] Received question: '{question}'")

    # Check if RAG system was properly initialized
    if rag_collection is None:
        error_msg = "RAG system is not properly initialized. Please check server logs for details."
        print(f"[MCP Tool Call] Error: {error_msg}")
        return error_msg

    try:
        # Retrieve relevant chunks from the document
        relevant_chunks = rag.retrieve_relevant_chunks(
            question=question,
            collection=rag_collection,
            n_results=3  # You can configure the number of chunks to retrieve
        )

        if not relevant_chunks or all(chunk is None or chunk.strip() == "" for chunk in relevant_chunks):
            return "No relevant information found in the document for your question."

        # Generate an answer using the retrieved context
        answer = rag.generate_answer(
            question=question,
            context_chunks=relevant_chunks
        )

        print(f"[MCP Tool Call] Generated answer: '{answer[:100]}...'")
        return answer

    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"[MCP Tool Call] Error: {error_msg}")
        return error_msg


#  3. RUN THE SERVER

if __name__ == "__main__":
    print("Starting MCP server. Listening for requests on stdio...")
    print(f"PDF Path: {PDF_PATH}")
    print(f"ChromaDB Path: {CHROMA_DB_PATH}")

    # The server will now listen for JSON-RPC messages on standard input
    # and send responses to standard output.
    mcp.run(transport="stdio")
