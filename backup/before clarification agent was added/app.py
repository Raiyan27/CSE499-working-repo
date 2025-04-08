import streamlit as st
import os
import fitz  # PyMuPDF
from typing import List
from typing_extensions import TypedDict
from operator import itemgetter
from dotenv import load_dotenv
load_dotenv()

# LangChain Imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

# LangGraph Imports
from langgraph.graph import END, StateGraph

# --- Constants ---
DATA_DIR = "./data"  # Directory for PDF files
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "Legal_data"
GRAPH_IMAGE_PATH = "agent_graph.png" # Path to save the graph image

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return None

def process_pdfs_in_directory(directory_path):
    """Processes all PDFs in a directory, extracts text, and splits into documents."""
    all_docs = []
    if not os.path.exists(directory_path):
        st.warning(f"Directory not found: {directory_path}")
        return []

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.warning(f"No PDF files found in {directory_path}")
        return []

    progress_bar = st.progress(0, text="Processing PDFs...")
    total_files = len(pdf_files)
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(directory_path, filename)
        # st.write(f"Processing {filename}...") # Can be verbose
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            # Using RecursiveCharacterTextSplitter as SemanticChunker needs embeddings model initialized first
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            docs = text_splitter.create_documents([pdf_text], metadatas=[{"source": filename}]) # Add source metadata
            all_docs.extend(docs)
        progress_bar.progress((i + 1) / total_files, text=f"Processing: {filename} ({i+1}/{total_files})")

    progress_bar.empty() # Clear the progress bar
    if not all_docs:
        st.warning("No text could be extracted from the PDFs.")
    else:
        st.info(f"Processed {total_files} PDF files, extracted {len(all_docs)} document chunks.")
    return all_docs

# --- Caching Functions for Streamlit ---

# Cache the embeddings model initialization
@st.cache_resource
def get_embedding_model(openai_api_key):
    # Setting the API key globally within the function might be redundant if set elsewhere,
    # but ensures it's available when this cached function runs.
    os.environ['OPENAI_API_KEY'] = openai_api_key
    try:
        st.write("Initializing Embeddings Model...")
        model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
        st.write("Embeddings Model Initialized.")
        return model
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings: {e}")
        return None

# Cache the Vector DB loading/creation
@st.cache_resource(ttl="1h") # Cache for 1 hour
def load_or_create_vector_db(_embeddings_model):
    if not _embeddings_model:
        st.error("Embeddings model not available. Cannot load/create Vector DB.")
        return None

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.info(f"Loading existing vector database from {PERSIST_DIR}...")
        try:
            vector_db = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
                embedding_function=_embeddings_model
            )
            # Quick check to see if the collection is accessible
            count = vector_db._collection.count()
            st.success(f"Vector database loaded successfully with {count} documents.")
            return vector_db
        except Exception as e:
            st.error(f"Error loading existing database: {e}. Will try to recreate.")
            # Potentially clear the directory if loading fails critically
            # import shutil
            # shutil.rmtree(PERSIST_DIR)
            # Note: Automatically recreating might delete user data unexpectedly.
            # It's often safer to prompt the user or require manual deletion.
            st.warning("Manual deletion of the './chroma_db' directory might be required if recreation fails.")


    st.info(f"Creating new vector database in {PERSIST_DIR}...")
    # Ensure data directory exists or handle it
    if not os.path.exists(DATA_DIR):
         os.makedirs(DATA_DIR)
         st.warning(f"Created data directory: {DATA_DIR}. Please add PDF files there and rerun.")
         return None # Stop if data dir was just created empty

    docs = process_pdfs_in_directory(DATA_DIR)
    if not docs:
        st.error("No documents processed from PDF files. Cannot create vector database.")
        return None

    st.info(f"Embedding {len(docs)} document chunks...")
    progress_bar_embed = st.progress(0, text="Embedding documents...")
    try:
        # Chroma.from_documents doesn't easily show progress, this is a placeholder
        # For large datasets, consider batching and updating progress manually if possible.
        vector_db = Chroma.from_documents(
            collection_name=COLLECTION_NAME,
            documents=docs,
            embedding=_embeddings_model,
            persist_directory=PERSIST_DIR
        )
        progress_bar_embed.progress(1.0, text="Embedding complete.")
        progress_bar_embed.empty()
        st.success("New vector database created and persisted.")
        return vector_db
    except Exception as e:
        progress_bar_embed.empty()
        st.error(f"Failed to create vector database: {e}")
        return None

# --- Agentic RAG Components (Functions) ---

# Pydantic model for grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Cache the agent graph compilation
# Depends on API keys implicitly via components
@st.cache_resource(ttl="1h")
def get_agentic_rag_app(_retriever, openai_api_key, tavily_api_key):
    if not _retriever:
       st.error("Retriever not available. Cannot build agent.")
       return None
    if not openai_api_key or not tavily_api_key:
        st.error("API keys not set. Cannot build agent.")
        return None

    # Ensure API keys are set in the environment for LangChain components
    # Although passed as args, some components might implicitly check env vars
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['TAVILY_API_KEY'] = tavily_api_key

    # --- LLMs and Tools ---
    try:
        st.write("Initializing LLMs and Tools...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        # Ensure Tavily client is initialized correctly
        tv_search = TavilySearchResults(max_results=2, search_depth='advanced', max_tokens=10000, tavily_api_key=tavily_api_key)
        st.write("LLMs and Tools Initialized.")
    except Exception as e:
        st.error(f"Failed to initialize LLMs or Tools: {e}")
        return None

    # --- Prompts ---
    # Grader Prompt
    SYS_PROMPT_GRADER = """You are an expert grader assessing relevance of a retrieved document to a user question.
                 Follow these instructions for grading:
                   - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                   - Your grade should be strictly 'yes' or 'no' to indicate whether the document is relevant to the question or not.""" # Added strictness
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT_GRADER),
        ("human", "Retrieved document:\n{document}\nUser question:\n{question}"),
    ])

    # RAG Prompt
    SYS_PROMPT_RAG = """You are an assistant for question-answering Bangladeshi Legal tasks.
             Use the following pieces of retrieved context to answer the question.
             If no context is present or if you don't know the answer based on the provided documents, state that clearly.
             Do not make up an answer.
             Provide a detailed and specific answer based *only* on the context provided.
             If context includes filenames, cite them like this: (Source: filename.pdf). If the context is from a web search, cite it as (Source: Web Search).
             Question:
             {question}
             Context:
             {context}
             Answer:"""
    rag_prompt_template = ChatPromptTemplate.from_template(SYS_PROMPT_RAG)

    # Web Search Re-writer Prompt
    SYS_WEB_SEARCH_PROMPT = """You are an expert question re-writer. Your task is to convert the user's question into an effective query optimized for a web search engine like Google or Tavily.
                   Focus on keywords and clear phrasing. Remove conversational elements.
                   Example Input: "Can you tell me about the inheritance laws for non-Muslims in Bangladesh according to the documents?"
                   Example Output: "inheritance laws non-Muslims Bangladesh"
                   """
    re_write_prompt_web = ChatPromptTemplate.from_messages([
        ("system", SYS_WEB_SEARCH_PROMPT),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved web search query."),
    ])

    # Initial Query Re-writer Prompt
    SYS_INITIAL_QUERY_PROMPT = """You are a question re-writer specializing in legal queries for a vector database retrieval system containing Bangladeshi laws. Your task is to:
                - Analyze the input question to understand its core legal intent.
                - Rewrite the question to be precise, unambiguous, and use terminology likely present in legal documents.
                - Focus on keywords and legal concepts relevant to Bangladesh. Remove conversational filler.
                - Do NOT expand the question into multiple sub-questions. Keep it as a single, optimized query for retrieval.

                Example Input: "What should I do if my landlord is evicting me unfairly?"
                Example Output: "Legal grounds for tenant eviction under Bangladeshi law"

                Example Input: "Tell me about company registration."
                Example Output: "Procedure for company registration in Bangladesh"

                Rewrite the following question for optimal legal document retrieval in Bangladesh:"""
    re_write_initial_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_INITIAL_QUERY_PROMPT),
        ("human", "Initial question:\n{question}"),
    ])

    # --- Chains ---
    doc_grader = (grade_prompt | structured_llm_grader)
    question_rewriter_web = (re_write_prompt_web | llm | StrOutputParser())
    initial_query_rewriter = (re_write_initial_prompt | llm | StrOutputParser())

    def format_docs_with_sources(docs: List[Document]) -> str:
        """Formats documents for the RAG prompt, including sources."""
        if not docs:
            return "No relevant context found."
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content
            formatted.append(f"--- Context from: {source} ---\n{content}")
        return "\n\n".join(formatted)


    qa_rag_chain = (
        {
            "context": itemgetter('context') | RunnableLambda(format_docs_with_sources), # Pass Document objects directly
            "question": itemgetter('question')
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    # --- Graph State ---
    class GraphState(TypedDict):
        question: str           # The original or rewritten question for the current step
        original_question: str  # Store the initial user question
        generation: str         # The final generated answer
        web_search_needed: str  # "Yes" or "No"
        documents: List[Document] # List of relevant documents (can include DB docs and web results)

    # --- Nodes ---
    def rewrite_initial_query(state):
        """Rewrites the initial user question for better DB retrieval."""
        print("---NODE: Initial Query Rewriter---")
        original_question = state["question"]
        better_question = initial_query_rewriter.invoke({"question": original_question})
        print(f"Original: {original_question}")
        print(f"Rewritten for DB: {better_question}")
        # Keep the original question for context, use the better question for retrieval
        return {"question": better_question, "original_question": original_question}

    def retrieve(state):
        """Retrieves documents from the vector database based on the current question."""
        print("---NODE: Retrieve from Vector DB---")
        question = state["question"] # Use the potentially rewritten question
        print(f"Retrieving for: {question}")
        try:
             # Ensure retriever is invoked correctly
            documents = _retriever.invoke(question)
            print(f"Retrieved {len(documents)} documents.")
            return {"documents": documents, "question": question} # Pass potentially rewritten question
        except Exception as e:
            print(f"---ERROR during retrieval: {e}---")
            st.warning(f"Error during document retrieval: {e}")
            return {"documents": [], "question": question} # Return empty list on error

    def grade_documents_node(state):
        """Grades retrieved documents for relevance and decides if web search is needed."""
        print("---NODE: Grade Documents---")
        question = state["original_question"] # Grade against the *original* question's intent
        documents = state["documents"]
        print(f"Grading {len(documents)} documents against: {question}")
        filtered_docs = []
        web_search_needed = "No" # Default to No

        if not documents:
             print("---No documents retrieved, definitely needs web search.---")
             web_search_needed = "Yes"
        else:
            all_irrelevant = True
            for d in documents:
                doc_content = d.page_content
                doc_source = d.metadata.get("source", "Unknown Source")
                try:
                    # Ensure the input matches the grader prompt keys
                    score = doc_grader.invoke({"question": question, "document": doc_content})
                    grade = score.binary_score.lower().strip() # Normalize the grade

                    if grade == "yes":
                        print(f"---GRADE: Relevant (Source: {doc_source})---")
                        filtered_docs.append(d)
                        all_irrelevant = False # Found at least one relevant doc
                    else:
                        print(f"---GRADE: Not Relevant (Source: {doc_source})---")
                        # We still might not need web search if *other* docs are relevant
                except Exception as e:
                    print(f"---ERROR Grading Document (Source: {doc_source}): {e}---")
                    # Option: Treat grading errors as needing web search? Or just skip the doc?
                    # Let's assume error means uncertainty -> maybe web search is needed
                    web_search_needed = "Yes" # Set to Yes if grading fails for any doc

            # Decide on web search *after* grading all docs
            if all_irrelevant and documents: # If we retrieved docs, but *none* were relevant
                print("---All retrieved documents were graded irrelevant.---")
                web_search_needed = "Yes"
            elif not documents: # Handles the initial check again, just to be sure
                 web_search_needed = "Yes"


        print(f"---Web Search Decision: {web_search_needed}---")
        # Return the *filtered* relevant documents and the decision
        return {"documents": filtered_docs, "question": state["question"], "web_search_needed": web_search_needed}


    def rewrite_query_web(state):
        """Rewrites the original question for effective web search."""
        print("---NODE: Rewrite Query for Web Search---")
        original_question = state["original_question"] # Rewrite the original question
        web_query = question_rewriter_web.invoke({"question": original_question})
        print(f"Original: {original_question}")
        print(f"Rewritten for Web: {web_query}")
        # Update the question field to the web query for the next step
        # Keep existing *relevant* documents from the DB retrieval
        return {"documents": state["documents"], "question": web_query}

    def web_search(state):
        """Performs web search using Tavily."""
        print("---NODE: Web Search---")
        web_query = state["question"] # This is the web-optimized query now
        original_documents = state["documents"] # Keep relevant DB docs
        print(f"Searching web for: {web_query}")

        try:
            # Invoke Tavily tool. It expects a string query directly or a dict with "query" key.
            # Check Tavily documentation/Langchain wrapper for exact input format.
            # Assuming direct string input works or the wrapper handles it:
            docs_dict_list = tv_search.invoke(web_query) # Pass the query string

            web_results_docs = []
            if docs_dict_list and isinstance(docs_dict_list, list):
                 # Convert Tavily results (dictionaries) into Langchain Document objects
                 content_limit = 3000 # Max characters per web result content
                 for doc_dict in docs_dict_list:
                     content = doc_dict.get("content", "")[:content_limit]
                     if content: # Only add if there's content
                         metadata = {"source": "Web Search", "url": doc_dict.get("url", "N/A")}
                         web_results_docs.append(Document(page_content=content, metadata=metadata))

            if web_results_docs:
                print(f"---Web Search added {len(web_results_docs)} results.---")
                # Combine relevant DB docs with new web results
                combined_documents = original_documents + web_results_docs
                return {"documents": combined_documents, "question": state["original_question"]} # Revert question for generation
            else:
                print("---Web Search returned no usable results.---")
                # Proceed with only the original documents (if any)
                return {"documents": original_documents, "question": state["original_question"]}

        except Exception as e:
            print(f"---ERROR During Web Search: {e}---")
            st.warning(f"Web search failed: {e}")
            # Return original documents if web search fails
            return {"documents": original_documents, "question": state["original_question"]}


    def generate_answer(state):
        """Generates the final answer using the gathered context."""
        print("---NODE: Generate Answer---")
        question_for_llm = state["original_question"] # Use original question for LLM
        documents = state["documents"] # Use combined DB + Web docs
        print(f"Generating answer for: {question_for_llm}")
        print(f"Using {len(documents)} documents as context.")

        if not documents:
            print("---No documents available for generation.---")
            # Provide a specific message if no context could be found
            generation = "I could not find relevant information in the local documents or via web search to answer your question."
            return {"documents": documents, "question": question_for_llm, "generation": generation}

        try:
            # Prepare context for the RAG chain correctly
            generation = qa_rag_chain.invoke({"context": documents, "question": question_for_llm})
            print("---Generation Complete---")
            return {"documents": documents, "question": question_for_llm, "generation": generation}
        except Exception as e:
            print(f"---ERROR During Generation: {e}---")
            st.error(f"Error during answer generation: {e}")
            return {"documents": documents, "question": question_for_llm, "generation": f"Sorry, an error occurred while generating the answer: {e}"}


    # --- Conditional Edges ---
    def decide_to_generate(state):
        """Decides whether to proceed to generation or initiate web search."""
        print("---EDGE: Decide to Generate or Web Search---")
        web_search_needed = state["web_search_needed"]

        if web_search_needed == "Yes":
            print("---DECISION: Routing to Web Search Path---")
            return "rewrite_query_web" # Route to web search rewrite node
        else:
            # Only generate if web search wasn't needed (i.e., DB docs were sufficient)
            print("---DECISION: Routing to Generate Answer (from DB Docs)---")
            return "generate_answer" # Route directly to generation

    # --- Build Graph ---
    st.write("Building Agent Graph...")
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("query_rewriter", rewrite_initial_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query_web", rewrite_query_web)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)

    # Set entry point
    graph.set_entry_point("query_rewriter")

    # Add edges
    graph.add_edge("query_rewriter", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",      # Source node
        decide_to_generate,     # Function to decide the next node
        {
            "rewrite_query_web": "rewrite_query_web", # If decision is "rewrite_query_web", go here
            "generate_answer": "generate_answer",     # If decision is "generate_answer", go here
        },
    )
    graph.add_edge("rewrite_query_web", "web_search")
    graph.add_edge("web_search", "generate_answer") # Always generate after attempting web search
    graph.add_edge("generate_answer", END) # Final step

    try:
        st.write("Compiling Agent Graph...")
        # Compile the graph
        agentic_rag_compiled = graph.compile()
        st.success("Agent graph compiled successfully.")

        # --- Save graph image ---
        try:
            # Ensure necessary dependencies are installed:
            # pip install mermaid-py playwright
            # playwright install
            st.write(f"Attempting to save graph image to {GRAPH_IMAGE_PATH}...")
            # Check if the get_graph method exists and has draw_mermaid_png
            if hasattr(agentic_rag_compiled, 'get_graph') and hasattr(agentic_rag_compiled.get_graph(), 'draw_mermaid_png'):
                image_bytes = agentic_rag_compiled.get_graph().draw_mermaid_png()
                with open(GRAPH_IMAGE_PATH, "wb") as f:
                    f.write(image_bytes)
                st.info(f"Agent graph visualization saved as {GRAPH_IMAGE_PATH}")
            else:
                st.warning("Could not generate graph image: Method not found. LangGraph version might differ or dependencies missing.")
        except ImportError as img_e:
             st.warning(f"Could not save agent graph image due to missing dependencies: {img_e}. Try `pip install mermaid-py playwright` and `playwright install`.")
        except Exception as img_e:
            st.warning(f"Could not save agent graph image: {img_e}")
        # --- End of image saving ---

        return agentic_rag_compiled
    except Exception as e:
        st.error(f"Failed to compile agent graph: {e}")
        # Also print traceback for detailed debugging if needed
        import traceback
        traceback.print_exc()
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="Legal RAG Chatbot (Bangladesh)", layout="wide")
st.title("⚖️ Bangladeshi Legal Assistant (Agentic RAG)")

# --- API Key Input & Resource Initialization ---
with st.sidebar:
    st.header("Configuration")
    # Use text_input for keys, allowing users to paste them if not in .env
    openai_api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    tavily_api_key_input = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))

    keys_provided = openai_api_key_input and tavily_api_key_input

    if not keys_provided:
        st.warning("Please enter your OpenAI and Tavily API keys in the sidebar to proceed.")
        st.stop() # Stop execution if keys are not provided

    # Initialize resources only if keys are present
    st.markdown("---")
    st.subheader("Status")
    # Use a placeholder for status updates during init
    status_placeholder = st.empty()
    status_placeholder.info("Initializing resources... (This might take a moment on first run or data changes)")

    # Pass keys directly to cached functions
    embed_model = get_embedding_model(openai_api_key_input)
    vector_db = load_or_create_vector_db(embed_model) # Pass the embed_model instance

    # Initialize retriever and agent app after DB is ready
    similarity_retriever = None
    app = None
    if vector_db:
        try:
            # Configure the retriever
            similarity_retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.45} # Increased K slightly, adjusted threshold
                # search_kwargs={"k": 3, "score_threshold": 0.5} # Alternative setting
            )
            st.success("Retriever is ready.")

            # Get the compiled agent app (this also saves the graph image)
            app = get_agentic_rag_app(similarity_retriever, openai_api_key_input, tavily_api_key_input)
            if not app:
                 st.error("Agent application failed to compile. Check errors above.")

        except Exception as e:
            st.error(f"Error setting up retriever or agent: {e}")

    else:
        st.error("Vector Database could not be initialized. Cannot proceed.")


    if app and similarity_retriever:
         status_placeholder.success("All resources initialized successfully!")
    else:
         status_placeholder.error("Initialization failed. Check console/log for errors.")
         st.stop() # Stop if essential components failed


    st.markdown("---")
    st.markdown("Powered by LangChain, LangGraph, OpenAI, ChromaDB, Tavily & Streamlit")
    # Display the saved graph image in the sidebar if it exists
    if os.path.exists(GRAPH_IMAGE_PATH):
        st.sidebar.subheader("Agent Workflow")
        st.sidebar.image(GRAPH_IMAGE_PATH, use_column_width=True)


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me any question about the loaded Bangladeshi legal documents."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...🧠 Processing your request...")

        if app: # Check if the agent app compiled successfully
             try:
                # Prepare the input for the LangGraph app
                # The initial state requires the 'question' key
                inputs = {"question": prompt}

                # --- Invoke the LangGraph app ---
                # Using invoke gets the final state result directly
                final_state = app.invoke(inputs)

                # Extract the final generation from the state
                full_response = final_state.get("generation", "Sorry, I couldn't generate a response after processing.")

                # Update the placeholder with the final response
                message_placeholder.markdown(full_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # --- (Optional) Stream events for debugging/detailed feedback ---
                # comment out the invoke block above and uncomment this section
                # to see step-by-step node outputs in the console/terminal
                # response_chunks = []
                # final_generation = "Processing..."
                # print("\n--- Streaming Graph Execution ---")
                # for output_chunk in app.stream(inputs, {"recursion_limit": 10}): # Add recursion limit
                #     # stream() yields dictionaries with node names as keys
                #     for node_name, output_value in output_chunk.items():
                #         print(f"Output from node '{node_name}':")
                #         # Print the relevant parts of the state from this node's output
                #         if isinstance(output_value, dict):
                #             print(f"  Keys: {list(output_value.keys())}")
                #             # Optionally print specific values like 'question' or 'generation' if they exist
                #             if 'question' in output_value: print(f"  Question: {output_value['question']}")
                #             if 'generation' in output_value:
                #                  print(f"  Generation (partial/final): {output_value['generation']}")
                #                  final_generation = output_value['generation'] # Capture the last generation
                #             if 'documents' in output_value: print(f"  Documents processed: {len(output_value['documents'])}")

                #         # You could potentially update the UI incrementally here if needed
                #         # message_placeholder.markdown(f"Processing step: {node_name}...") # Example UI update

                # print("--- End Streaming ---")

                # # After streaming, get the final answer (might be captured in the loop)
                # if not final_generation or final_generation == "Processing...":
                #      final_generation = "Sorry, processing finished but couldn't generate a final answer."

                # message_placeholder.markdown(final_generation)
                # st.session_state.messages.append({"role": "assistant", "content": final_generation})
                # --- End of Streaming Section ---


             except Exception as e:
                 st.error(f"An error occurred during agent execution: {e}")
                 # Print full traceback to console for debugging
                 import traceback
                 traceback.print_exc()
                 error_message = f"Sorry, I encountered an error processing your request. Please check the logs or try again. Error: {e}"
                 message_placeholder.markdown(error_message)
                 st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            error_message = "The chatbot application is not ready. Initialization might have failed. Please check sidebar status and logs."
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})