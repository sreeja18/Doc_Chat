from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_all_pdf(pdf_directory, vectorstore=None):
    """
    Process PDFs in a directory, skipping PDFs that are already in the vector store.
    
    Args:
        pdf_directory: Path to PDFs
        vectorstore: optional, VectorStore instance for checking already indexed PDFs
    
    Returns:
        list of documents
    """
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    # Track already indexed PDFs if vectorstore is provided
    existing_files = set()
    if vectorstore is not None:
        try:
            existing = vectorstore.collection.get(include=["metadatas"])
            for meta in existing["metadatas"]:
                if meta and "source_file" in meta:
                    existing_files.add(meta["source_file"])
        except Exception as e:
            print(f"Warning: Could not fetch existing files: {e}")

    for pdf_file in pdf_files:
        # Skip PDFs already in DB
        if pdf_file.name in existing_files:
            print(f"Skipping {pdf_file.name} (already indexed)")
            continue

        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\nTotal docs loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs