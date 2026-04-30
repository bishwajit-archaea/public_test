"""Search agent using dual embeddings (GraphCodeBERT + Sentence-BERT)."""

from typing import Dict, Any, List
from opensearchpy import OpenSearch
from app.services.opensearch.dual_embedder import DualEmbedder
import time
import json
from app.core.logger import logger
from app.core.config import settings_object as settings
import asyncio
from app.services.opensearch import opensearch_client, OpenSearch
import shutil
from pathlib import Path

class DualSearchAgent:
    """
    Search agent with dual embedding support.
    """
    
    def __init__(
        self,
        index_name: str = settings.OPENSEARCH_INDEX,
    ):
        self.index_name = index_name
        self.client: OpenSearch = opensearch_client
        
        # Initialize embedder (lazy loading)
        self._embedder = None
        
        # Create index if not exists
        self.create_index()
        self._get_embedder()
    
    def _get_embedder(self):
        """Get or create embedder (lazy loading)."""
        if self._embedder is None:
            logger.info("📦 Loading embedding models (one-time)...")
            self._embedder = DualEmbedder(use_graphcodebert=True, use_sentencebert=True)
        return self._embedder

    def create_index(self):
        # CREATE INDEX IF NOT EXISTS
        if not self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' does not exist. Creating...")
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        # File metadata
                        "dataset": {
                            "type": "keyword"  # ✅ This is already keyword type
                        },
                        "file_path": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "file_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        
                        # Chunk metadata
                        "type": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "line_count": {"type": "integer"},
                        
                        # Code and description
                        "code": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "description": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        
                        # Identifiers for keyword search
                        "identifiers": {
                            "type": "keyword"
                        },
                        
                        # Dual vector embeddings
                        "code_embedding": {
                            "type": "knn_vector",
                            "dimension": 768,  # GraphCodeBERT
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "text_embedding": {
                            "type": "knn_vector",
                            "dimension": 384,  # Sentence-BERT
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"Index '{self.index_name}' created.")

    async def delete_index_all_data(self):
        # Delete all documents but keep the index
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"match_all": {}}}
        )
        logger.info(f"All documents from index '{self.index_name}' deleted.")
        # remove tree directories folders not the main directory
        await self._cleanup_trees_folders()
        self.create_index()
        logger.info(f"Index '{self.index_name}' created.")
        return True

    async def _cleanup_trees_folders(self):
        """Remove all subdirectories and files in TREES_JSON_FOLDER but keep the main directory"""
        trees_folder = Path(settings.TREES_JSON_FOLDER)
        
        if trees_folder.exists() and trees_folder.is_dir():
            # Remove all contents (files and subdirectories) but keep the main folder
            for item in trees_folder.iterdir():
                if item.is_file():
                    item.unlink()  # Delete file
                    logger.info(f"Deleted file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)  # Delete directory and all its contents
                    logger.info(f"Deleted directory: {item}")
            
            logger.info(f"Cleaned up all contents in trees folder: {trees_folder}")
        else:
            logger.warning(f"Trees folder does not exist: {trees_folder}")
    
    async def keyword_search(
        self,
        query: str,
        current_dataset: str,
        max_results: int = 10,
        search_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Keyword search on code, description, and identifiers.
        """
        if search_fields is None:
            search_fields = ["code", "description", "identifiers", "name", "file_path"]
        
        start_time = time.time()
        
        body = {
            "size": max_results,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": search_fields,
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "dataset": current_dataset  # ✅ FIXED: Remove .keyword
                            }
                        }
                    ]
                }
            },
            "_source": ["file_path", "file_name", "name", "type", "start_line", 
                    "end_line", "code", "description", "identifiers", "dataset"]
        }
        logger.info(f"🔎 Keyword Search Body: {json.dumps(body)}")
        
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "search_type": "keyword",
            "total_hits": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"] if response["hits"]["hits"] else 0,
            "search_time_ms": round(search_time_ms, 2),
            "results": [
                {"score": hit["_score"], **hit["_source"]}
                for hit in response["hits"]["hits"]
            ]
        }
    
    async def code_semantic_search(
        self,
        query: str,
        current_dataset: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Semantic search using code embeddings (GraphCodeBERT).
        Best for code-to-code similarity.
        """
        embedder = self._get_embedder()
        
        # Generate code embedding
        query_embedding = embedder.generate_code_embedding(query)
        
        start_time = time.time()
        
        response = self.client.search(
            index=self.index_name,
            body={
                "size": max_results,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "code_embedding": {
                                        "vector": query_embedding,
                                        "k": max_results
                                    }
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "dataset": current_dataset  # ✅ FIXED: Remove .keyword
                                }
                            }
                        ]
                    }
                },
                "_source": ["file_path", "file_name", "name", "type", "start_line",
                        "end_line", "code", "description", "dataset"]
            }
        )
                
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "search_type": "code_semantic",
            "total_hits": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"] if response["hits"]["hits"] else 0,
            "search_time_ms": round(search_time_ms, 2),
            "results": [
                {"score": hit["_score"], **hit["_source"]}
                for hit in response["hits"]["hits"]
            ]
        }
    
    async def text_semantic_search(
        self,
        query: str,
        current_dataset: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Semantic search using text embeddings (Sentence-BERT).
        Best for natural language queries.
        """
        embedder = self._get_embedder()
        
        # Generate text embedding
        start_time = time.time()
        query_embedding = embedder.generate_text_embedding(query)
        
        body = {
            "size": max_results,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "text_embedding": {
                                    "vector": query_embedding,
                                    "k": max_results
                                }
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "dataset": current_dataset  # ✅ FIXED: Remove .keyword
                            }
                        }
                    ]
                }
            },
            "_source": ["file_path", "file_name", "name", "type", "start_line",
                    "end_line", "code", "description", "dataset"]
        }
        
        logger.info(f"🔎 Text Semantic Search - Dataset Filter: {current_dataset}")
        
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "search_type": "text_semantic",
            "total_hits": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"] if response["hits"]["hits"] else 0,
            "search_time_ms": round(search_time_ms, 2),
            "results": [
                {"score": hit["_score"], **hit["_source"]}
                for hit in response["hits"]["hits"]
            ]
        }
    
    async def hybrid_search(
        self,
        query: str,
        current_dataset: str,
        max_results: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        use_text_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Hybrid search combining keyword and semantic search.
        """
        logger.info(f"🔍 Hybrid Search - Query: '{query}', Dataset: '{current_dataset}'")
        
        # Get both results
        if use_text_semantic:
            keyword_task = asyncio.create_task(self.keyword_search(query, current_dataset, max_results * 2))
            semantic_task = asyncio.create_task(self.text_semantic_search(query, current_dataset, max_results * 2))
        else:
            keyword_task = asyncio.create_task(self.keyword_search(query, current_dataset, max_results * 2))
            semantic_task = asyncio.create_task(self.code_semantic_search(query, current_dataset, max_results * 2))

        keyword_results, semantic_results = await asyncio.gather(keyword_task, semantic_task)

        # Merge and score
        seen = {}
        
        # Add keyword results
        kw_max = keyword_results['max_score'] or 1.0
        for result in keyword_results['results']:
            key = f"{result['file_path']}:{result['start_line']}"
            seen[key] = {
                **result,
                'combined_score': (result['score'] / kw_max) * keyword_weight,
                'source': 'keyword'
            }
        
        # Add/boost with semantic results
        sem_max = semantic_results['max_score'] or 1.0
        for result in semantic_results['results']:
            key = f"{result['file_path']}:{result['start_line']}"
            sem_score = (result['score'] / sem_max) * semantic_weight
            
            if key in seen:
                seen[key]['combined_score'] += sem_score
                seen[key]['source'] = 'both'
            else:
                seen[key] = {
                    **result,
                    'combined_score': sem_score,
                    'source': 'semantic'
                }
        
        # Sort and limit
        combined = sorted(seen.values(), key=lambda x: x['combined_score'], reverse=True)
        
        return {
            "query": query,
            "search_type": "hybrid",
            "total_hits": len(combined),
            "results": combined[:max_results],
            "keyword_hits": keyword_results['total_hits'],
            "semantic_hits": semantic_results['total_hits']
        }
    
    def format_results(self, results: Dict[str, Any], max_display: int = 5):
        """Pretty print results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"SEARCH RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Query: '{results['query']}'")
        logger.info(f"Type: {results['search_type']}")
        logger.info(f"Total hits: {results['total_hits']}")
        
        if 'keyword_hits' in results:
            logger.info(f"  - Keyword: {results['keyword_hits']}")
            logger.info(f"  - Semantic: {results['semantic_hits']}")
        
        if 'search_time_ms' in results:
            logger.info(f"Search time: {results['search_time_ms']}ms")
        
        logger.info(f"\n{'-'*80}")
        
        for i, hit in enumerate(results['results'][:max_display], 1):
            score = hit.get('combined_score', hit.get('score', 0))
            source = hit.get('source', 'N/A')
            
            logger.info(f"\n{i}. Score: {score:.4f}", end='')
            if source != 'N/A':
                logger.info(f" [{source}]", end='')
            logger.info()
            
            logger.info(f"   📁 {hit['file_name']}")
            logger.info(f"   🏷️  {hit.get('name', 'N/A')} ({hit.get('type', 'N/A')})")
            logger.info(f"   📍 Lines {hit['start_line']}-{hit['end_line']}")
            logger.info(f"   📝 {hit.get('description', 'N/A')[:100]}...")
            logger.info(f"   💻 {hit['code'][:80]}...")
        
        logger.info(f"\n{'='*80}\n")


def test_dual_search_agent(query: str = "function that handles authentication"):
    """Test the dual search agent."""
    agent = DualSearchAgent()
    
    logger.info("🤖 Dual Search Agent Test")
    logger.info("="*80 + "\n")
    
    # Test different search types
    logger.info(f"Testing query: '{query}'\n")
    
    # 1. Keyword search
    logger.info("1️⃣  KEYWORD SEARCH")
    results = agent.keyword_search(query, max_results=5)
    agent.format_results(results, max_display=3)
    
    # 2. Text semantic search (best for natural language)
    logger.info("\n2️⃣  TEXT SEMANTIC SEARCH (Sentence-BERT)")
    results = agent.text_semantic_search(query, max_results=5)
    agent.format_results(results, max_display=3)
    
    # 3. Hybrid search
    logger.info("\n3️⃣  HYBRID SEARCH")
    results = agent.hybrid_search(query, max_results=5)
    agent.format_results(results, max_display=3)


if __name__ == "__main__":
    test_dual_search_agent()
