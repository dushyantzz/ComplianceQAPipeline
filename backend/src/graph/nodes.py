import json
import os
import logging
import re  # <--- Added Regex for cleaning
from typing import Dict, Any, List

from openai import NotFoundError as OpenAINotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

# Import the State schema .
from backend.src.graph.state import VideoAuditState, ComplianceIssue

# Import the Service
from backend.src.services.video_indexer import VideoIndexerService

# Configure Logger
logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)

_KEYWORD_QUERY_MAX_CHARS = 12000


def _keyword_search_rules(query_text: str, k: int = 3) -> List[Document]:
    """
    Full-text search on Azure AI Search when no embedding deployment is available.
    Uses the same default content field as LangChain's AzureSearch ("content").
    """
    endpoint = (os.getenv("AZURE_SEARCH_ENDPOINT") or "").rstrip("/")
    key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not (endpoint and key and index_name):
        logger.warning("Azure Search configuration missing; cannot retrieve rules.")
        return []

    q = query_text[:_KEYWORD_QUERY_MAX_CHARS] if query_text else "*"
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
    )
    try:
        results = client.search(search_text=q, top=k, select=["id", "content"])
    except Exception as e:
        logger.error("Azure AI Search keyword query failed: %s", e)
        return []

    docs: List[Document] = []
    for r in results:
        docs.append(
            Document(
                page_content=(r.get("content") or ""),
                metadata={"id": r.get("id")},
            )
        )
    return docs


# --- NODE 1: THE INDEXER ---
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Downloads YouTube video, uploads to Azure VI, and extracts insights.
    """
    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")
    
    logger.info(f"--- [Node: Indexer] Processing: {video_url} ---")
    
    local_filename = "temp_audit_video.mp4"
    
    try:
        vi_service = VideoIndexerService()
        
        # 1. DOWNLOAD
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL for this test.")

        # 2. UPLOAD
        azure_video_id = vi_service.upload_video(local_path, video_name=video_id_input)
        logger.info(f"Upload Success. Azure ID: {azure_video_id}")
        
        # 3. CLEANUP
        if os.path.exists(local_path):
            os.remove(local_path)

        # 4. WAIT
        raw_insights = vi_service.wait_for_processing(azure_video_id)
        
        # 5. EXTRACT
        clean_data = vi_service.extract_data(raw_insights)
        
        logger.info("--- [Node: Indexer] Extraction Complete ---")
        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "", 
            "ocr_text": []
        }

# --- NODE 2: THE COMPLIANCE AUDITOR ---
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Performs Retrieval-Augmented Generation (RAG) to audit the content.
    """
    logger.info("--- [Node: Auditor] querying Knowledge Base & LLM ---")
    
    transcript = state.get("transcript", "")
    
    if not transcript:
        logger.warning("No transcript available. Skipping Audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No Transcript)."
        }

    # Initialize Clients (deployment names must match Azure AI Foundry / OpenAI resource)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )

    embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"

    docs: List[Document] = []
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        dims_env = os.getenv("AZURE_SEARCH_VECTOR_DIMENSIONS")
        dims = int(dims_env) if dims_env else len(embeddings.embed_query("ping"))

        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query,
            vector_search_dimensions=dims,
        )
        docs = vector_store.similarity_search(query_text, k=3)
    except OpenAINotFoundError:
        logger.warning(
            "Embedding deployment %r not found on this OpenAI resource (404). "
            "Using keyword search. Fix: deploy embeddings on the SAME resource as "
            "AZURE_OPENAI_ENDPOINT, or set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to the real name.",
            embedding_deployment,
        )
        docs = _keyword_search_rules(query_text, k=3)

    # RAG Retrieval (vector or keyword fallback)
    
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])
    
    # --- UPDATED PROMPT WITH STRICT SCHEMA ---
    system_prompt = f"""
    You are a Senior Brand Compliance Auditor.
    
    OFFICIAL REGULATORY RULES:
    {retrieved_rules}
    
    INSTRUCTIONS:
    1. Analyze the Transcript and OCR text below.
    2. Identify ANY violations of the rules.
    3. Return strictly JSON in the following format:
    
    {{
        "compliance_results": [
            {{
                "category": "Claim Validation",
                "severity": "CRITICAL",
                "description": "Explanation of the violation..."
            }}
        ],
        "status": "FAIL", 
        "final_report": "Summary of findings..."
    }}

    If no violations are found, set "status" to "PASS" and "compliance_results" to [].
    """

    user_message = f"""
    VIDEO METADATA: {state.get('video_metadata', {})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR): {ocr_text}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        # --- FIX: Clean Markdown if present (```json ... ```) ---
        content = response.content
        if "```" in content:
            # Regex to find JSON inside code blocks
            content = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL).group(1)
            
        audit_data = json.loads(content.strip())
        
        return {
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated.")
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        # Log the raw response to see what went wrong
        logger.error(f"Raw LLM Response: {response.content if 'response' in locals() else 'None'}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL"
        }