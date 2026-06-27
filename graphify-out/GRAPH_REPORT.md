# Graph Report - .  (2026-06-27)

## Corpus Check
- Corpus is ~5,227 words - fits in a single context window. You may not need a graph.

## Summary
- 75 nodes · 86 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.85)
- Token cost: 1,250 input · 850 output

## Community Hubs (Navigation)
- [[_COMMUNITY_LangGraph Workflow Nodes and State|LangGraph Workflow Nodes and State]]
- [[_COMMUNITY_FastAPI Server and Telemetry|FastAPI Server and Telemetry]]
- [[_COMMUNITY_Video Indexer Service Integration|Video Indexer Service Integration]]
- [[_COMMUNITY_Azure Search and Document Ingestion|Azure Search and Document Ingestion]]
- [[_COMMUNITY_YouTube Ad Specifications|YouTube Ad Specifications]]
- [[_COMMUNITY_CLI Entry Point and Simulation|CLI Entry Point and Simulation]]
- [[_COMMUNITY_Compliance Pipeline Root Package|Compliance Pipeline Root Package]]

## God Nodes (most connected - your core abstractions)
1. `VideoIndexerService` - 10 edges
2. `index_video_node()` - 7 edges
3. `VideoAuditState` - 7 edges
4. `YouTube Ad Specs` - 7 edges
5. `audit_content_node()` - 6 edges
6. `ComplianceQAPipeline` - 5 edges
7. `AuditRequest` - 4 edges
8. `AuditResponse` - 4 edges
9. `audit_video()` - 4 edges
10. `_keyword_search_rules()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `YouTube Ad Specs` --semantically_similar_to--> `Azure AI Search`  [INFERRED] [semantically similar]
  backend/data/youtube-ad-specs.pdf → README.md
- `FTC Endorsement Guides` --semantically_similar_to--> `Azure AI Search`  [INFERRED] [semantically similar]
  backend/data/1001a-influencer-guide-508_1.pdf → README.md
- `index_video_node()` --references--> `Azure Video Indexer`  [INFERRED]
  backend/src/graph/nodes.py → README.md
- `index_docs()` --references--> `Azure AI Search`  [INFERRED]
  backend/scripts/index_documents.py → README.md
- `index_video_node()` --calls--> `VideoIndexerService`  [EXTRACTED]
  backend/src/graph/nodes.py → backend/src/services/video_indexer.py

## Import Cycles
- None detected.

## Communities (9 total, 1 thin omitted)

### Community 0 - "LangGraph Workflow Nodes and State"
Cohesion: 0.16
Nodes (15): Any, Document, audit_content_node(), index_video_node(), _keyword_search_rules(), Performs Retrieval-Augmented Generation (RAG) to audit the content., Full-text search on Azure AI Search when no embedding deployment is available., Downloads YouTube video, uploads to Azure VI, and extracts insights. (+7 more)

### Community 1 - "FastAPI Server and Telemetry"
Cohesion: 0.16
Nodes (14): audit_video(), AuditRequest, AuditResponse, ComplianceIssue, health_check(), Main API endpoint that triggers the compliance audit workflow.          HTTP M, # NOTE: In production, you'd use:, Simple endpoint to verify the API is running.          Used by:     - Load ba (+6 more)

### Community 2 - "Video Indexer Service Integration"
Cohesion: 0.17
Nodes (7): Parses the JSON into our State format., Generates an ARM Access Token., Exchanges ARM token for Video Indexer Account Token., Downloads a YouTube video to a local file., Uploads a LOCAL FILE to Azure Video Indexer., Polls status until complete., VideoIndexerService

### Community 3 - "Azure Search and Document Ingestion"
Cohesion: 0.17
Nodes (11): Disclosure Requirements, FTC Endorsement Guides, Material Connection, Azure AI Search, Azure OpenAI, Azure Video Indexer, ComplianceQAPipeline, LangChain (+3 more)

### Community 4 - "YouTube Ad Specifications"
Cohesion: 0.29
Nodes (7): Bumper Ads, In-Feed Video Ads, Masthead Ads, Non-Skippable In-Stream Ads, Outstream Ads, Skippable In-Stream Ads, YouTube Ad Specs

### Community 5 - "CLI Entry Point and Simulation"
Cohesion: 0.50
Nodes (3): Main Execution Entry Point for Brand Guardian AI.  This file is the "control c, Simulates a Video Compliance Audit request.          This function orchestrate, run_cli_simulation()

## Knowledge Gaps
- **11 isolated node(s):** `complianceqapipeline`, `Azure OpenAI`, `LangGraph`, `LangChain`, `Material Connection` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `index_video_node()` connect `LangGraph Workflow Nodes and State` to `Video Indexer Service Integration`, `Azure Search and Document Ingestion`?**
  _High betweenness centrality (0.276) - this node is a cross-community bridge._
- **Why does `Azure Video Indexer` connect `Azure Search and Document Ingestion` to `LangGraph Workflow Nodes and State`?**
  _High betweenness centrality (0.220) - this node is a cross-community bridge._
- **What connects `Reads PDFs from backend/data, chunks them, and uploads vectors to Azure AI Searc`, `Defines the expected structure of incoming API requests.          Pydantic val`, `Defines the structure of a single compliance violation.          Used inside A` to the rest of the system?**
  _33 weakly-connected nodes found - possible documentation gaps or missing edges._