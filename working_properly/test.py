import streamlit as st
import os
import fitz  # PyMuPDF
from typing import List, Tuple, Optional
from typing_extensions import TypedDict
from operator import itemgetter
import io # For image saving

# LangChain Imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.document_loaders import PyPDFDirectoryLoader # Using custom loader now
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

# LangGraph Imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver # For potential future stateful memory

# --- Constants ---
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "Legal_data"
GRAPH_IMAGE_PATH = "agent_graph.png"

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
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
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(directory_path, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            # Add filename as metadata for source tracking
            docs = text_splitter.create_documents([pdf_text], metadatas=[{"source": filename}])
            all_docs.extend(docs)
        progress_bar.progress((i + 1) / len(pdf_files), text=f"Processing: {filename}")

    progress_bar.empty()
    if not all_docs:
        st.warning("No text could be extracted from the PDFs.")
    return all_docs

# --- Caching Functions ---

@st.cache_resource
def get_embedding_model(openai_api_key):
    # No need to set os.environ here if passing key directly
    try:
        return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings: {e}. Please check your API key.")
        st.stop() # Stop execution if embeddings fail

@st.cache_resource(ttl="2h") # Cache DB for 2 hours
def load_or_create_vector_db(_embeddings_model):
    # Renamed internal variable to avoid conflict
    embeddings_func = _embeddings_model
    if not embeddings_func:
        st.error("Embeddings model not available. Cannot load/create Vector DB.")
        return None

    # Ensure the persist directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Check if the directory is not empty and potentially contains a valid Chroma DB
    # A more robust check might involve trying to load metadata or count items
    db_exists = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

    if db_exists:
        st.info(f"Attempting to load existing vector database from {PERSIST_DIR}...")
        try:
            vector_db = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings_func
            )
            # Perform a quick check, e.g., count items
            count = vector_db._collection.count()
            st.success(f"Vector database with {count} items loaded successfully.")
            return vector_db
        except Exception as e:
            st.warning(f"Could not load existing database: {e}. Will attempt to recreate.")
            # Consider clearing the directory if loading fails persistently
            # import shutil
            # shutil.rmtree(PERSIST_DIR)

    st.info(f"Creating new vector database in {PERSIST_DIR}...")
    docs = process_pdfs_in_directory(DATA_DIR)
    if not docs:
        st.error("No documents processed from PDF files. Cannot create vector database.")
        return None

    try:
        vector_db = Chroma.from_documents(
            collection_name=COLLECTION_NAME,
            documents=docs,
            embedding=embeddings_func,
            persist_directory=PERSIST_DIR # Chroma handles persistence automatically here
        )
        count = vector_db._collection.count()
        st.success(f"New vector database created and persisted with {count} items.")
        return vector_db
    except Exception as e:
        st.error(f"Failed to create vector database: {e}")
        return None

# --- Agentic RAG Components ---

# Pydantic models
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class ClarityDecision(BaseModel):
    """Decision on whether to ask for clarification or proceed with web search."""
    decision: str = Field(description="Either 'clarification' or 'web_search'")

class ClarificationRequest(BaseModel):
    """Question to ask the user for clarification."""
    question_to_user: str = Field(description="A clear question asking the user to provide more specific details.")

class RefinedQuery(BaseModel):
    """A refined query based on user feedback."""
    refined_question: str = Field(description="The user's original question refined with their feedback.")


# Cache the agent graph compilation
@st.cache_resource(ttl="2h")
def get_agentic_rag_app(_retriever, openai_api_key, tavily_api_key):
    if not _retriever:
       st.error("Retriever not available. Cannot build agent.")
       return None
    if not openai_api_key or not tavily_api_key:
        st.error("API keys not set. Cannot build agent.")
        return None

    # Set keys for components that use environment variables implicitly if not passed directly
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['TAVILY_API_KEY'] = tavily_api_key

    # --- LLMs and Tools ---
    try:
        # Main LLM for generation and rewriting
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        # Specialized LLM for structured outputs (can be the same model)
        structured_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

        # Structured LLM callers
        structured_llm_grader = structured_llm.with_structured_output(GradeDocuments)
        structured_llm_clarity_decision = structured_llm.with_structured_output(ClarityDecision)
        structured_llm_clarification_request = structured_llm.with_structured_output(ClarificationRequest)
        structured_llm_refine_query = structured_llm.with_structured_output(RefinedQuery)

        tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, tavily_api_key=tavily_api_key)
    except Exception as e:
        st.error(f"Failed to initialize LLMs or Tools: {e}")
        return None

    # --- Prompts ---
    SYS_PROMPT_GRADER = """You are an expert grader assessing relevance of a retrieved document to a user question.
                 Follow these instructions for grading:
                   - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                   - Your grade should be either 'yes' or 'no'."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT_GRADER),
        ("human", "Retrieved document:\n{document}\nUser question:\n{question}"),
    ])

    # Updated RAG Prompt with Memory Placeholder
    SYS_PROMPT_RAG = """You are an assistant for question-answering Bangladeshi Legal tasks.
             Use the following pieces of retrieved context and the conversation history to answer the question.
             If no context is present or if you don't know the answer, just say that you don't know the answer based on the provided documents or web search.
             Do not make up the answer unless it is there in the provided context.
             Give a detailed answer and to the point answer with regard to the question.
             Include the source document filename(s) in your answer where possible, like '(Source: filename.pdf)' or '(Source: Web Search)'.
             Consider the chat history to understand the context of the current question, especially if it's a follow-up.
             Answer the LATEST user question based on the provided context and history.
             """
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT_RAG),
        MessagesPlaceholder(variable_name="chat_history"), # For memory
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
    ])


    SYS_WEB_SEARCH_PROMPT = """Act as a question re-writer. Convert the following input question to a better version that is optimized for web search.
                   Reason about the underlying semantic intent."""
    re_write_prompt_web = ChatPromptTemplate.from_messages([
        ("system", SYS_WEB_SEARCH_PROMPT),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved question."),
    ])

    SYS_INITIAL_QUERY_PROMPT = """Act as a question re-writer specializing in legal queries related to Bangladeshi laws. Rewrite the question to be more precise and optimized for database retrieval.
                Examples:
                Input: "What about inheritance?" Output: "What are the laws governing property inheritance in Bangladesh?"
                Input: "complaint against company" Output: "What is the legal procedure for filing a complaint against a company in Bangladesh?"
                Now, rewrite the following question."""
    re_write_initial_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_INITIAL_QUERY_PROMPT),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved question."),
    ])

    # Prompt for deciding clarification vs web search
    SYS_CLARITY_DECISION_PROMPT = """The user asked: '{question}'. Initial document retrieval failed or yielded irrelevant results.
    Is the question potentially too vague or ambiguous to proceed effectively with a web search?
    Decide if we should ask the user for clarification first ('clarification') or attempt a web search directly ('web_search')."""
    clarity_decision_prompt = ChatPromptTemplate.from_template(SYS_CLARITY_DECISION_PROMPT)

    # Prompt for generating the clarification question
    SYS_CLARIFICATION_REQUEST_PROMPT = """The user asked: '{question}'. This question was deemed too vague after initial retrieval failed.
    Formulate a specific question to ask the user back, prompting them for more details or context so you can better understand their need."""
    clarification_request_prompt = ChatPromptTemplate.from_template(SYS_CLARIFICATION_REQUEST_PROMPT)

    # Prompt for refining query with user feedback
    SYS_REFINE_QUERY_PROMPT = """The user's original question was: '{original_question}'.
    It was deemed too vague, and they provided the following clarification/feedback: '{user_feedback}'.
    Combine the original question and the feedback into a new, refined question suitable for retrieval or web search."""
    refine_query_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_REFINE_QUERY_PROMPT),
        ("human", "Original Question: {original_question}\nUser Feedback: {user_feedback}\n\nRefined Question:")
    ])

    # --- Chains ---
    doc_grader = grade_prompt | structured_llm_grader
    question_rewriter_web = re_write_prompt_web | llm | StrOutputParser()
    initial_query_rewriter = re_write_initial_prompt | llm | StrOutputParser()
    clarity_decision_chain = clarity_decision_prompt | structured_llm_clarity_decision
    clarification_request_chain = clarification_request_prompt | structured_llm_clarification_request
    refine_query_chain = refine_query_prompt | structured_llm_refine_query

    def format_docs_with_sources(docs: List[Document]):
        if not docs:
            return "No relevant documents found."
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content
            formatted.append(f"--- Document {i+1} (Source: {source}) ---\n{content}")
        return "\n\n".join(formatted)

    # RAG Chain now includes chat_history
    qa_rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_with_sources(x["context"])
        )
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    # --- Graph State ---
    class GraphState(TypedDict):
        question: str
        generation: Optional[str]
        documents: List[Document]
        chat_history: Optional[List[Tuple[str, str]]] # List of (human, ai) tuples
        # HITL State
        original_question: Optional[str] # Store the initial question if clarification is needed
        user_feedback: Optional[str]     # Store user's clarification
        clarification_needed: bool       # Flag to signal HITL needed
        clarification_question: Optional[str] # Question to ask user
        # Control Flow State
        web_search_needed: str # 'Yes' or 'No'

    # --- Nodes ---
    def route_request(state: GraphState):
        """Routes the request based on whether user feedback is present."""
        print("---ROUTING REQUEST---")
        if state.get("user_feedback"):
            print("---Routing to Incorporate Feedback---")
            return "incorporate_feedback"
        else:
            print("---Routing to Initial Query Rephraser---")
            # Store the input question as the original question for potential HITL
            return {"original_question": state["question"]} , "query_rephraser" # Pass state update and route


    def rewrite_initial_query(state: GraphState):
        """Rewrites the initial query for better retrieval."""
        print("---INITIAL QUERY REPHRASER---")
        question = state["question"]
        better_question = initial_query_rewriter.invoke({"question": question})
        print(f"Initial Rewritten Query: {better_question}")
        return {"question": better_question, "original_question": state.get("original_question", question)} # Keep original Q

    def incorporate_feedback(state: GraphState):
        """Refines the query using user feedback."""
        print("---INCORPORATING USER FEEDBACK---")
        original_question = state.get("original_question")
        user_feedback = state.get("user_feedback")
        if not original_question or not user_feedback:
            print("---Warning: Missing original question or feedback for refinement---")
            # Fallback: use the latest question if feedback is missing somehow
            return {"question": state["question"]}

        refined_result = refine_query_chain.invoke({
            "original_question": original_question,
            "user_feedback": user_feedback
        })
        better_question = refined_result.refined_question
        print(f"Query Refined with Feedback: {better_question}")
        # Clear feedback fields after use
        return {"question": better_question, "user_feedback": None, "original_question": None}


    def retrieve(state: GraphState):
        """Retrieves documents from the vector store."""
        print("---RETRIEVAL FROM VECTOR DB---")
        question = state["question"]
        documents = _retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents.")
        return {"documents": documents}

    def grade_documents_node(state: GraphState):
        """Grades the relevance of retrieved documents."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search_needed = "No" # Assume relevant initially

        if not documents:
             print("---NO DOCUMENTS RETRIEVED---")
             web_search_needed = "Yes"
        else:
            all_irrelevant = True
            for d in documents:
                try:
                    score = doc_grader.invoke({"question": question, "document": d.page_content})
                    grade = score.binary_score
                    doc_source = d.metadata.get("source", "Unknown Source")
                    if grade.lower() == "yes":
                        print(f"---GRADE: DOCUMENT RELEVANT ({doc_source})---")
                        filtered_docs.append(d)
                        all_irrelevant = False # Found at least one relevant doc
                    else:
                        print(f"---GRADE: DOCUMENT NOT RELEVANT ({doc_source})---")
                        # Don't set web_search_needed to Yes here yet, wait until all docs graded
                except Exception as e:
                    print(f"---ERROR GRADING DOCUMENT ({doc_source}): {e}---")
                    # Treat grading error as potentially irrelevant? Or keep the doc? For now, let's exclude.
            
            if all_irrelevant:
                 print("---ALL RETRIEVED DOCUMENTS GRADED IRRELEVANT---")
                 web_search_needed = "Yes"
            elif not filtered_docs and documents: # Some docs retrieved but ALL failed grading (e.g., errors)
                 print("---ALL RETRIEVED DOCUMENTS FAILED GRADING---")
                 web_search_needed = "Yes"


        print(f"---Web Search Needed Decision: {web_search_needed}---")
        # Return only relevant documents
        return {"documents": filtered_docs, "web_search_needed": web_search_needed}

    def decide_clarification_or_websearch(state: GraphState):
        """Decides whether to ask for clarification or proceed with web search."""
        print("---DECIDING CLARIFICATION OR WEB SEARCH---")
        question = state["question"]
        try:
            decision_result = clarity_decision_chain.invoke({"question": question})
            decision = decision_result.decision.lower()
            print(f"---Decision: {decision}---")
            if decision == "clarification":
                return "request_clarification"
            else: # Default to web search if decision is unclear
                return "rewrite_query_web"
        except Exception as e:
            print(f"---ERROR during clarification decision: {e}. Defaulting to web search.---")
            return "rewrite_query_web"


    def request_clarification(state: GraphState):
        """Generates the question to ask the user for clarification."""
        print("---REQUESTING CLARIFICATION FROM USER---")
        question = state.get("original_question", state["question"]) # Use original if available
        try:
            clarification_result = clarification_request_chain.invoke({"question": question})
            clarification_q = clarification_result.question_to_user
            print(f"---Clarification Question: {clarification_q}---")
            # Signal that the graph should end here and wait for user input
            return {"clarification_needed": True, "clarification_question": clarification_q}
        except Exception as e:
            print(f"---ERROR generating clarification question: {e}. Cannot proceed with HITL.---")
            # Fallback: maybe try web search anyway? Or end with error? Let's end.
            return {"generation": "Sorry, I need more information, but encountered an error asking for it."}


    def rewrite_query_web(state: GraphState):
        """Rewrites the query for a web search."""
        print("---REWRITE QUERY FOR WEB SEARCH---")
        question = state["question"]
        better_question = question_rewriter_web.invoke({"question": question})
        print(f"Rewritten Query for Web: {better_question}")
        return {"question": better_question} # Update question for web search node

    def web_search(state: GraphState):
        """Performs a web search and adds results to documents."""
        print("---WEB SEARCH---")
        question = state["question"]
        existing_documents = state.get("documents", []) # Keep any relevant docs from initial retrieval
        try:
            print(f"---Searching Tavily for: {question}---")
            docs_results = tv_search.invoke(question) # Tavily tool expects string input directly
            web_results_content = "\n\n".join([d.get("content", "") for d in docs_results])

            if web_results_content:
                web_results_doc = Document(page_content=web_results_content, metadata={"source": "Web Search"})
                print(f"---Appending Web Search Results ({len(web_results_content)} chars)---")
                combined_documents = existing_documents + [web_results_doc]
                return {"documents": combined_documents}
            else:
                print("---WEB SEARCH RETURNED NO CONTENT---")
                return {"documents": existing_documents} # Return existing docs if search empty
        except Exception as e:
            print(f"---ERROR DURING WEB SEARCH: {e}---")
            return {"documents": existing_documents} # Return existing docs if search fails


    def generate_answer(state: GraphState):
        """Generates the final answer using RAG."""
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        # Prepare chat history from state if available
        history = state.get("chat_history", [])
        langchain_messages = []
        for human, ai in history:
            langchain_messages.append(HumanMessage(content=human))
            langchain_messages.append(AIMessage(content=ai))

        if not documents:
            print("---NO DOCUMENTS AVAILABLE FOR GENERATION---")
            # Handle no documents case - maybe answer from history or say dunno
            generation = "I couldn't find relevant information in the provided documents or via web search to answer your question."
        else:
            try:
                generation = qa_rag_chain.invoke(
                    {"context": documents, "question": question, "chat_history": langchain_messages}
                )
                print("---GENERATION COMPLETE---")
            except Exception as e:
                print(f"---ERROR DURING GENERATION: {e}---")
                generation = f"Sorry, an error occurred while generating the answer: {e}"

        return {"generation": generation, "clarification_needed": False} # Ensure clarification flag is reset

    # --- Conditional Edges ---
    def decide_action_after_grading(state: GraphState):
        """Decides the next step after grading documents."""
        print("---ASSESSING GRADED DOCUMENTS---")
        web_search_needed = state["web_search_needed"]
        documents = state["documents"] # These are already filtered

        if web_search_needed == "No":
            # We have relevant documents from the DB
            print("---DECISION: Generate answer from DB docs---")
            return "generate_answer"
        else:
            # No relevant docs from DB, need to decide between clarification and web search
            print("---DECISION: DB retrieval failed/irrelevant, check if clarification needed---")
            return "decide_clarification_or_websearch"


    # --- Build Graph ---
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("route_request", route_request)
    workflow.add_node("query_rephraser", rewrite_initial_query)
    workflow.add_node("incorporate_feedback", incorporate_feedback)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("decide_clarification_or_websearch", decide_clarification_or_websearch)
    workflow.add_node("request_clarification", request_clarification)
    workflow.add_node("rewrite_query_web", rewrite_query_web)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)

    # Define edges
    workflow.set_entry_point("route_request")

    workflow.add_edge("route_request", "query_rephraser") # This edge seems redundant if route_request returns the target?
    # Let's handle routing purely via conditional edges from route_request if needed, or ensure route_request passes state correctly.
    # The current route_request function returns a tuple (state_update, target_node), LangGraph doesn't support this directly.
    # Option 1: Make route_request just return the target node name string.
    # Option 2: Use conditional edges from route_request. Let's try Option 2.

    # Revised routing logic: route_request just determines target
    def route_request_conditional(state: GraphState):
        print("---ROUTING REQUEST---")
        if state.get("user_feedback"):
            print("---Routing to Incorporate Feedback---")
            return "incorporate_feedback"
        else:
            print("---Routing to Initial Query Rephraser---")
            return "query_rephraser"

    workflow.add_node("route_logic", route_request_conditional) # Use this for conditional edge source
    workflow.set_entry_point("route_logic")

    # Store original question before rephrasing (if not coming from feedback loop)
    def store_original_question(state: GraphState):
        # Only store if it's not already set (i.e., not in feedback loop)
        if not state.get("original_question"):
             return {"original_question": state["question"]}
        return {} # No change if original_question exists

    workflow.add_node("store_q", store_original_question)

    workflow.add_conditional_edges(
        "route_logic",
        lambda x: x, # Pass the string returned by route_request_conditional
        {
            "incorporate_feedback": "incorporate_feedback",
            "query_rephraser": "store_q", # Go to store_q first
        }
    )
    workflow.add_edge("store_q", "query_rephraser")

    # Continue graph flow
    workflow.add_edge("query_rephraser", "retrieve")
    workflow.add_edge("incorporate_feedback", "retrieve") # After feedback, retrieve again
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_action_after_grading, # Function returns target node name
        {
            "generate_answer": "generate_answer",
            "decide_clarification_or_websearch": "decide_clarification_or_websearch",
        },
    )

    workflow.add_conditional_edges(
        "decide_clarification_or_websearch",
        lambda x: x, # Pass the string returned by the decision node
         {
            "request_clarification": "request_clarification",
            "rewrite_query_web": "rewrite_query_web",
        },
    )

    workflow.add_edge("rewrite_query_web", "web_search")
    workflow.add_edge("web_search", "generate_answer")

    # End points
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("request_clarification", END) # End here to wait for user


    # Compile the graph
    try:
        agentic_rag_compiled = workflow.compile()
        st.success("Agent graph compiled successfully.")

        # --- Save Graph Image ---
        try:
            png_bytes = agentic_rag_compiled.get_graph().draw_mermaid_png()
            with open(GRAPH_IMAGE_PATH, "wb") as f:
                f.write(png_bytes)
            print(f"--- Saved Graph Image to {GRAPH_IMAGE_PATH} ---")
        except Exception as img_err:
            st.warning(f"Could not save graph image: {img_err}")

        return agentic_rag_compiled
    except Exception as e:
        st.error(f"Failed to compile agent graph: {e}")
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="Legal RAG Chatbot (Bangladesh)", layout="wide")
st.title("⚖️ Bangladeshi Legal Assistant (Agentic RAG)")

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me any question about the loaded Bangladeshi legal documents!"}]
if "clarification_pending" not in st.session_state:
    st.session_state.clarification_pending = False
if "original_question" not in st.session_state:
    st.session_state.original_question = None

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key", value=os.environ.get("OPENAI_API_KEY", ""))
    tavily_api_key = st.text_input("Tavily API Key", type="password", key="tavily_key", value=os.environ.get("TAVILY_API_KEY", ""))

    keys_provided = openai_api_key and tavily_api_key
    if not keys_provided:
        st.warning("Please enter your OpenAI and Tavily API keys to initialize the assistant.")
        st.stop()

    # Initialize resources only if keys are present
    with st.spinner("Initializing resources... (can take a minute on first run)"):
        embed_model = get_embedding_model(openai_api_key)
        vector_db = load_or_create_vector_db(embed_model)

        if vector_db:
            # Adjust retriever settings if needed
            similarity_retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.55} # May need tuning
            )
            st.success("Retriever is ready.")
            # Compile agent app
            compiled_app = get_agentic_rag_app(similarity_retriever, openai_api_key, tavily_api_key)
        else:
            st.error("Vector Database initialization failed. Cannot proceed.")
            compiled_app = None
            st.stop()

    if compiled_app and os.path.exists(GRAPH_IMAGE_PATH):
        st.success("Agent Ready!")
        st.markdown("---")
        st.subheader("Agent Workflow Graph")
        st.image(GRAPH_IMAGE_PATH)
    elif not compiled_app:
         st.error("Agent compilation failed.")
         st.stop()

    st.markdown("---")
    st.markdown("Powered by LangChain, LangGraph, OpenAI, ChromaDB, Tavily & Streamlit")


# --- Chat Interface ---

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt_placeholder = "Ask your legal question..."
if st.session_state.clarification_pending:
    prompt_placeholder = "Please provide clarification..."

if prompt := st.chat_input(prompt_placeholder):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare graph input
    inputs = {}
    if st.session_state.clarification_pending:
        # User is providing feedback
        inputs = {
            "question": st.session_state.original_question, # Use original question
            "user_feedback": prompt # The new input is feedback
        }
        st.session_state.clarification_pending = False # Reset flag
        st.session_state.original_question = None
    else:
        # It's a new question
        inputs = {"question": prompt}

    # Add chat history to the input state for the graph
    # Keep only last N turns to avoid excessive token usage? (e.g., last 5 turns * 2 = 10 messages)
    history_turns = 5
    past_messages = st.session_state.messages[- (history_turns * 2) -1 : -1] # Get previous messages excluding the current user input
    chat_history_tuples = []
    human_msg = None
    for msg in past_messages:
         if msg["role"] == "user":
             human_msg = msg["content"]
         elif msg["role"] == "assistant" and human_msg is not None:
             chat_history_tuples.append((human_msg, msg["content"]))
             human_msg = None # Reset for next pair
    inputs["chat_history"] = chat_history_tuples

    # Display thinking message and invoke graph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...🧠")

        if compiled_app:
             try:
                # Config for tracing (optional)
                config = RunnableConfig(recursion_limit=50) # Increase if graph is deep

                # Invoke the graph
                response = compiled_app.invoke(inputs, config=config)

                # Check if clarification is needed
                if response.get("clarification_needed"):
                    clarification_q = response.get("clarification_question", "Could you please provide more details?")
                    full_response = clarification_q
                    # Set state for next interaction
                    st.session_state.clarification_pending = True
                    # Store the *original* question that led to clarification
                    st.session_state.original_question = inputs.get("question") # The question that was initially asked
                else:
                    # Get the final generation
                    full_response = response.get("generation", "Sorry, I couldn't generate a response.")
                    st.session_state.clarification_pending = False # Ensure reset if successful

                message_placeholder.markdown(full_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

             except Exception as e:
                 st.error(f"An error occurred during agent execution: {e}")
                 error_message = f"Sorry, I encountered an error: {e}"
                 message_placeholder.markdown(error_message)
                 st.session_state.messages.append({"role": "assistant", "content": error_message})
                 # Reset clarification state on error
                 st.session_state.clarification_pending = False
                 st.session_state.original_question = None
        else:
            error_message = "The chatbot application is not ready. Please check configuration."
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a clear button for chat history (optional)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Ask me a new question!"}]
    st.session_state.clarification_pending = False
    st.session_state.original_question = None
    st.rerun()