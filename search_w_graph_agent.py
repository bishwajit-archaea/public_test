"""
Graph-Enhanced Search Agent

Combines:
1. Hybrid search (keyword + semantic)
2. Code knowledge graph (relationships and context)
3. LLM reranking with graph context
4. Security scanning
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from app.services.llm.search_agent import EnhancedSearchAgent
from app.services.llm.query_expander_call import QueryExpander
from app.core.config import settings_object as settings 
from app.services.llm.code_rerank_call import rerank_with_llm_and_graph
from app.services.embedding.code_graph_builder import CodeGraphBuilder
import asyncio
from app.services.opensearch.opensearch import DualSearchAgent
from app.core.logger import logger


class GraphEnhancedAgent(EnhancedSearchAgent):
    """
    Search agent enhanced with code knowledge graph.
    """
    
    def __init__(
        self,
        client: OpenAI,
        model: str,
        dataset_path: str,
        trees_path: str,
        current_dataset: str,
        search_agent=DualSearchAgent(index_name=settings.OPENSEARCH_INDEX),
        index_name: str = settings.OPENSEARCH_INDEX,
    ):
        """
        Initialize Graph-Enhanced Search Agent.
        
        Args:
            client: OpenAI client
            model: Model name
            dataset_path: Path to dataset
            graph_path: Path to code graph JSON (will build if doesn't exist)
            index_name: OpenSearch index name
        """
        super().__init__(
            client=client,
            model=model,
            dataset_path=dataset_path,
            index_name=index_name,
            search_agent=search_agent,
            current_dataset=current_dataset
        )
        self.trees_path = trees_path
        self.graph_path = f"{self.trees_path}/code_graph.json"
        self.graph_builder = None
        self.graph = None
        self.client = client
        self.search_agent = search_agent
        self.current_dataset = current_dataset
        
        # Initialize query expander
        self.query_expander = QueryExpander(client=self.client,model=self.model)
        
        # Load or build graph
        self._load_or_build_graph()
        
        logger.info(f"✓ Graph-Enhanced Agent ready!\n")
    
    def _load_or_build_graph(self):
        """Load existing graph or build new one."""
        if Path(self.graph_path).exists():
            logger.info(f"📊 Loading code graph from {self.graph_path}...")
            with open(self.graph_path, 'r') as f:
                self.graph = json.load(f)
            logger.info(f"✓ Loaded graph: {len(self.graph['nodes'])} nodes, {len(self.graph['edges'])} edges\n")
        else:
            logger.info(f"📊 Building code graph from {self.dataset_path}...")
            final_dataset_path = f"{self.dataset_path}"
            self.graph_builder = CodeGraphBuilder(self.current_dataset)
            self.graph = self.graph_builder.build_graph(final_dataset_path)
            self.graph_builder.save_graph(self.graph_path)
            self.graph_builder.create_metadata_tree_json(final_dataset_path,self.trees_path)
    
    def _enrich_results_with_graph(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich search results with graph context.
        
        For each result, add:
        - What functions it calls
        - What calls it
        - Related functions in same file
        - Import dependencies
        """
        enriched = []
        
        for result in results:
            enriched_result = result.copy()
            
            # Get function name and file
            func_name = result.get('name')
            file_path = result.get('file_path')
            
            if func_name and func_name != 'anonymous':
                # Get graph context
                context = self._get_function_context(func_name, file_path)
                
                if context:
                    enriched_result['graph_context'] = context
                    
                    # Build a rich description including graph context
                    context_desc = []
                    
                    if context.get('calls'):
                        # Handle list of dicts
                        calls_str = ', '.join([c['name'] for c in context['calls'][:5]])
                        context_desc.append(f"Calls: {calls_str}")
                    
                    if context.get('called_by'):
                        # Handle list of dicts
                        called_by_str = ', '.join([c['name'] for c in context['called_by'][:5]])
                        context_desc.append(f"Called by: {called_by_str}")
                    
                    if context.get('file_imports'):
                        context_desc.append(f"Uses: {', '.join(context['file_imports'][:5])}")
                    
                    if context.get('related_functions'):
                        context_desc.append(f"Related: {', '.join(context['related_functions'][:3])}")
                    
                    if context_desc:
                        enriched_result['enriched_description'] = (
                            result.get('description', '') + '. ' + '. '.join(context_desc)
                        )
            
            enriched.append(enriched_result)
        
        return enriched
    
    def _get_function_context(self, function_name: str, file_path: str = None) -> Dict[str, Any]:
        """Get context for a function from the graph."""
        context = {
            'calls': [],
            'called_by': [],
            'file_imports': [],
            'related_functions': []
        }
        
        # Normalize file path for comparison
        def normalize_path(p):
            if p:
                return str(Path(p)).lower().replace('\\', '/').split('dataset/')[-1] if 'dataset' in str(p) else str(p)
            return None
        
        normalized_search_path = normalize_path(file_path)
        
        # Find the function node
        func_node_id = None
        matched_file_path = None
        for node_id, node in self.graph['nodes'].items():
            if node['type'] == 'function' and node['name'] == function_name:
                node_file_normalized = normalize_path(node['file'])
                if file_path is None or node_file_normalized == normalized_search_path or (normalized_search_path and normalized_search_path in node_file_normalized):
                    func_node_id = node_id
                    matched_file_path = node['file']
                    break
        
        if not func_node_id:
            return context
        
        # Find what this function calls
        for edge in self.graph['edges']:
            if edge['from'] == func_node_id and edge['type'] == 'calls':
                called_node = self.graph['nodes'][edge['to']]
                context['calls'].append({
                    'name': called_node['name'],
                    'file': called_node.get('file')
                })
            
            # Find what calls this function
            if edge['to'] == func_node_id and edge['type'] == 'calls':
                caller_node = self.graph['nodes'][edge['from']]
                context['called_by'].append({
                    'name': caller_node['name'],
                    'file': caller_node.get('file')
                })
        
        # Get file context - try to find the file in graph
        file_to_lookup = matched_file_path or file_path
        if file_to_lookup:
            # Try exact match first
            if file_to_lookup in self.graph['files']:
                file_info = self.graph['files'][file_to_lookup]
            else:
                # Try normalized match
                file_info = None
                for graph_file, info in self.graph['files'].items():
                    if normalize_path(graph_file) == normalized_search_path:
                        file_info = info
                        break
            
            if file_info:
                context['file_imports'] = [imp['name'] for imp in file_info.get('imports', [])]
                context['related_functions'] = [
                    f['name'] for f in file_info.get('functions', [])
                    if f['name'] != function_name
                ][:5]  # Limit to 5
        
        return context
    
    def _expand_context_for_result(self, result: Dict) -> Dict:
        """
        Expand context for a result by fetching related code.
        
        - If partial chunk, fetch all parts of the same function
        - Fetch implementations of called functions (top 3)
        """
        expanded = result.copy()
        
        # 1. Get all parts if this is a partial chunk
        if result.get('is_partial'):
            file_path = result.get('file_path')
            base_name = result.get('name', '').split(' (part ')[0]
            total_parts = result.get('total_parts', 1)
            
            # Fetch all parts from OpenSearch
            all_parts = []
            for part_num in range(1, total_parts + 1):
                part_name = f"{base_name} (part {part_num}/{total_parts})"
                # Search for this specific part
                try:
                    part_results = self.search_agent.client.search(
                            index=self.search_agent.index_name,
                            body={
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"term": {"file_path.keyword": file_path}},
                                            {"term": {"name.keyword": part_name}}
                                        ],
                                        "filter": [
                                            {
                                                "term": {
                                                    "dataset": self.current_dataset
                                                }
                                            }
                                        ]
                                    }
                                },
                                "size": 1
                            }
                        )
                    
                    if part_results['hits']['hits']:
                        part_code = part_results['hits']['hits'][0]['_source'].get('code', '')
                        all_parts.append(f"# Part {part_num}/{total_parts}\n{part_code}")
                except:
                    pass
            
            if all_parts:
                expanded['full_code'] = '\n\n'.join(all_parts)
        
        # 2. Get implementations of called functions
        if result.get('graph_context', {}).get('calls'):
            called_implementations = {}
            for func_name in result['graph_context']['calls'][:3]:  # Top 3
                # Search for this function in the index
                try:
                    func_results = self.search_agent.client.search(
                        index=self.search_agent.index_name,
                        body={
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"name.keyword": func_name}}
                                    ],
                                    "filter": [
                                        {
                                            "term": {
                                                "dataset": self.current_dataset
                                            }
                                        }
                                    ]
                                }
                            },
                            "size": 1
                        }
                    )
                    
                    if func_results['hits']['hits']:
                        func_code = func_results['hits']['hits'][0]['_source'].get('code', '')
                        called_implementations[func_name] = func_code[:500]  # First 500 chars
                except:
                    pass
            
            if called_implementations:
                expanded['called_implementations'] = called_implementations
        
        return expanded
    
    
    async def _hybrid_search_with_reranking(
        self,
        query: str,
        query_type: str,
        max_results: int = 10,
        initial_results: int = 30
    ) -> Dict[str, Any]:
        """
        Override to use query expansion + graph-enhanced reranking.
        """
        start_total = time.time()
        
        # Step 1: Expand query
        logger.info(f"🔍 Expanding query...")
        start_expand = time.time()
        expanded = self.query_expander.expand_query(query, max_variations=2)
        expand_time = time.time() - start_expand
        
        all_queries = [expanded['original']] + expanded.get('variations', [])
        
        logger.info(f"   Original: {expanded['original']}")
        for i, var in enumerate(expanded.get('variations', []), 1):
            logger.info(f"   Variation {i}: {var}")
        #  Step 2: Search with all query variations
        logger.info(f"📊 Running hybrid search with {len(all_queries)} queries...")

        start_search = time.time()
        all_results = []
        seen_ids: Set[str] = set()
    
        # Create async tasks for all queries
        search_tasks = []
        for q in all_queries:
            task = self.search_agent.hybrid_search(
                query=q,
                query_type=query_type,
                current_dataset=self.current_dataset,
                max_results=initial_results,
                use_text_semantic=True
            )
            search_tasks.append(task)

        # Wait for all searches to complete
        hybrid_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        # Process results
        for i, hybrid_results in enumerate(hybrid_results_list):
            if isinstance(hybrid_results, Exception):
                logger.info(f"⚠️  Search failed for query '{all_queries[i]}': {hybrid_results}")
                continue
            
            # Deduplicate results
            for result in hybrid_results.get('results', []):
                result_id = f"{result.get('file_path')}_{result.get('start_line')}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(result)

        taken_results = all_results[:initial_results]
        search_time = time.time() - start_search
        logger.info(f"✓ Found {len(all_results)} unique results and taken {len(taken_results)} results across search_time: {search_time:.2f}s for {len(all_queries)} queries\n")
        
        
        # Step 3: LLM Reranking with graph context
        logger.info(f"🤖 Reranking with GPT + Graph Context...")
        reranked, rerank_metrics = rerank_with_llm_and_graph(
            query,  # Use original query for reranking
            taken_results,
            max_results,
            self.search_agent,
            self.client,
            self.model,
            max_context_size=max_results+5,
            graph=self.graph
        )
        
        total_time = time.time() - start_total
        
        # Construct detailed metrics
        metrics = {
            "total_time": total_time,
            "search_time": search_time,  # Keep for backward compatibility
            "rerank_time": rerank_metrics.get("total_rerank_time", 0), # Keep for backward compatibility
            "breakdown": {
                "query_expansion": expand_time,
                "search_execution": search_time,
                "rerank_total": rerank_metrics.get("total_rerank_time", 0),
                "rerank_details": {
                    "graph_enrichment": rerank_metrics.get("graph_enrichment_time", 0),
                    "llm_api_call": rerank_metrics.get("llm_api_time", 0)
                }
            }
        }
        
        if "error" in rerank_metrics:
            metrics["error"] = rerank_metrics["error"]
        
        return {
            "query": query,
            "expanded_queries": all_queries,
            "search_type": "hybrid_with_expansion_and_reranking",
            "total_hits": len(reranked),
            "results": reranked,
            "original_hits": len(all_results),
            "metrics": metrics
        }
    async def delete_index_all_data(self):
        
        return await self.search_agent.delete_index_all_data()


def main():
    """Test the graph-enhanced search agent."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("❌ Set OPENAI_API_KEY environment variable")
        return
    
    # # agent = GraphEnhancedAgent(model=model_id)
    
    # query = "what is the frequency of the cron for running the dynamic pricing updates?"
    
    # results = agent.search(query, max_results=5)
    # agent.format_results(results)


if __name__ == "__main__":
    main()
