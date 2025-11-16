import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def build_rag_pipeline():
    """
    Builds the RAG pipeline using LCEL (LangChain Expression Language).
    """
    print("Building RAG pipeline...")

    # Load the provided text file (speech.txt)
    try:
        loader = TextLoader('speech.txt')
        documents = loader.load()
    except FileNotFoundError:
        print("Error: speech.txt not found. Make sure the file is in the same directory.")
        sys.exit(1)
    
    # split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # create embeddings
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # store embeddings in a local Chroma vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    # Initialize the LLM (Ollama with Mistral)
    llm = Ollama(model="mistral")
    
    # Define the retriever
    retriever = vector_store.as_retriever()

    # Define the Prompt Template
    # This instructs the LLM how to answer based only on the context.
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Define the Output Parser
    # This converts the LLM's response into a simple string.
    output_parser = StrOutputParser()

    # Create the LCEL RAG chain
    # This chain defines the flow of data.
    
    # This setup object retrieves the context and passes the question
    # through simultaneously.
    setup = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    
    # This is the full pipeline
    # 1. 'setup' runs, getting context and the question
    # 2. The output {context: ..., question: ...} is "piped" to the prompt
    # 3. The formatted prompt is "piped" to the llm
    # 4. The llm's output is "piped" to the output_parser
    chain = setup | prompt | llm | output_parser

    print("Pipeline built successfully. You can now ask questions.")
    return chain

def main():
    """
    Main function to run the command-line Q&A system.
    """
    qa_chain = build_rag_pipeline()
    
    print("\n--- Ambedkar Q&A System ---")
    print("Type 'exit' to quit.")
    
    while True:
        # Get user question
        question = input("\nAsk a question: ")
        
        if question.lower().strip() == 'exit':
            print("Goodbye!")
            break
            
        if not question.strip():
            print("Please enter a question.")
            continue
            
        # Generate an answer
        try:
            response = qa_chain.invoke(question)
            
            print("\nAnswer:")
            print(response)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()