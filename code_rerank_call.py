import time
import json
from typing import List, Dict, Tuple, Any
from pathlib import Path
from app.services.llm.utils import retry_and_log_llm_usage
from app.core.logger import logger
from openai import OpenAI


def rerank_with_llm_and_graph(
    query: str,
    results: List[Dict],
    top_k: int,
    search_agent,
    client,
    model: str,
    max_context_size: int = 10,
    graph=None
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Rerank results using LLM with graph context.
    """
    if not results:
        return [], {}
    
    t_start = time.time()
    
    # Enrich results with graph context
    t_enrich_start = time.time()
    enriched_results = _enrich_results_with_graph(results, graph)
    
    # Expand context for top 10 results (before reranking)
    logger.info(f"  📚 Expanding context for top {max_context_size} results...")
    for i, result in enumerate(enriched_results[:max_context_size]):
        enriched_results[i] = _expand_context_for_result(result, search_agent)
    t_enrich_end = time.time()
    
    # Prepare results for LLM with FULL context
    results_text = _prepare_results_text_for_reranking(enriched_results)
    
    try:
        t_llm_start = time.time()
        
        # Use the decorated LLM call function
        llm_response = rerank_with_llm(
            client=client,
            query=query,
            model=model,
            results_text=results_text,
            top_k=top_k
        )
        
        t_llm_end = time.time()
        
        reranked = _process_llm_reranking_response(llm_response, enriched_results)
        
        logger.info(f"✓ Reranked with graph context to {len(reranked)} results\n")
        
        metrics = {
            "graph_enrichment_time": t_enrich_end - t_enrich_start,
            "llm_api_time": t_llm_end - t_llm_start,
            "total_rerank_time": time.time() - t_start
        }
        return reranked, metrics
        
    except Exception as e:
        logger.info(f"⚠️  Reranking failed: {e}")
        logger.info(f"↪️  Returning enriched results\n")
        # Return enriched results even if reranking fails
        fallback = _create_fallback_results(enriched_results, top_k)
        return fallback, {"error": str(e), "total_rerank_time": time.time() - t_start}


@retry_and_log_llm_usage(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def rerank_with_llm(
    client: OpenAI,
    query: str,
    model: str,
    results_text: str,
    top_k: int
) -> Tuple[Any, Dict]:
    """
    Call LLM for reranking with retry and token logging.
    
    Returns:
        Tuple containing (raw_response, processed_response)
    """
    system_prompt = """You are a code search expert with FULL CODE CONTEXT. You can see the complete functions, not just snippets. Be specific and confident in your explanations - you have all the information needed."""
    
    user_prompt = f"""Query: "{query}"

Results with FULL CONTEXT (complete functions + called implementations):
{results_text}

Rank the top {top_k} most relevant results. You have access to:
1. Full function code (all parts combined)
2. Implementations of called functions
3. Graph relationships (calls, called_by)
4. Complete context to understand what the code actually does

Be SPECIFIC in your explanations - reference actual code, SQL queries, function calls, etc.
Avoid vague words like "may", "might", "could" - you have the full code!

Respond in JSON:
{{
    "ranked_results": [
        {{
            "result_number": 1,
            "relevance_score": 0.0-1.0,
            "why_relevant": "explanation including context"
        }},
        ...
    ]
}}"""

    logger.info(f"\n\n{user_prompt}\n\n")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    llm_response = json.loads(response.choices[0].message.content)
    return response, llm_response


def _prepare_results_text_for_reranking(enriched_results: List[Dict]) -> str:
    """
    Prepare formatted results text for LLM reranking.
    """
    return "\n\n".join([
        f"""Result {i}:
- File: {r.get('file_name', 'N/A')}
- Function: {r.get('name', 'N/A')}
- Description: {r.get('enriched_description', r.get('description', 'N/A'))[:200]}
- Current Chunk Code: {r.get('code', '')[:300]}...
{f"- Full Function Code (all parts): {r.get('full_code', '')[:800]}..." if r.get('full_code') else ""}
{f"- Calls: {', '.join([c['name'] for c in r.get('graph_context', {}).get('calls', [])[:3]])}" if r.get('graph_context', {}).get('calls') else ""}
{f"- Called by: {', '.join([c['name'] for c in r.get('graph_context', {}).get('called_by', [])[:3]])}" if r.get('graph_context', {}).get('called_by') else ""}
{f"- Called Function Implementations: {chr(10).join([f'  {name}: {code[:200]}...' for name, code in r.get('called_implementations', {}).items()])}" if r.get('called_implementations') else ""}"""
        for i, r in enumerate(enriched_results, 1)
    ])


def _process_llm_reranking_response(llm_response: Dict, enriched_results: List[Dict]) -> List[Dict]:
    """
    Process LLM response and create reranked results.
    """
    reranked = []
    for ranked_item in llm_response.get('ranked_results', []):
        result_idx = ranked_item['result_number'] - 1
        if 0 <= result_idx < len(enriched_results):
            result = enriched_results[result_idx].copy()
            result['llm_relevance_score'] = ranked_item.get('relevance_score', 1.0 - (len(reranked) * 0.1))
            result['llm_reasoning'] = ranked_item.get('why_relevant', 'Relevant to query')
            result['original_rank'] = result_idx + 1
            reranked.append(result)
    return reranked


def _create_fallback_results(enriched_results: List[Dict], top_k: int) -> List[Dict]:
    """
    Create fallback results when reranking fails.
    """
    fallback = []
    for i, result in enumerate(enriched_results[:top_k]):
        r = result.copy()
        r['llm_relevance_score'] = 0
        r['llm_reasoning'] = 'Default value Ranked by hybrid search score (graph-enriched)'
        fallback.append(r)
    return fallback


def _expand_context_for_result(result: Dict, search_agent=None) -> Dict:
    """
    Expand context for a result by fetching related code.
    
    - If partial chunk, fetch all parts of the same function
    - Fetch implementations of called functions (top 3)
    """
    expanded = result.copy()
    
    if not search_agent:
        return expanded
    
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
                part_results = search_agent.client.search(
                    index=search_agent.index_name,
                    body={
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"file_path.keyword": file_path}},
                                    {"term": {"name.keyword": part_name}}
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
        # calls is now a list of dicts: [{'name': 'foo', 'file': 'path/to/foo.py'}]
        for call_info in result['graph_context']['calls'][:3]:  # Top 3
            func_name = call_info.get('name')
            file_path = call_info.get('file')
            
            if not func_name:
                continue
                
            # Search for this function in the index
            try:
                must_conditions = [{"term": {"name.keyword": func_name}}]
                if file_path:
                     must_conditions.append({"term": {"file_path.keyword": file_path}})
                
                func_results = search_agent.client.search(
                    index=search_agent.index_name,
                    body={
                        "query": {
                            "bool": {
                                "must": must_conditions,
                                "filter": [
                                    {
                                        "term": {
                                            "dataset": search_agent.current_dataset
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


def _enrich_results_with_graph(results: List[Dict[str, Any]], graph=None) -> List[Dict[str, Any]]:
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
        
        if func_name and func_name != 'anonymous' and graph:
            # Get graph context
            context = _get_function_context(func_name, file_path, graph)
            
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


def _get_function_context(function_name: str, file_path: str = None, graph=None) -> Dict[str, Any]:
    """Get context for a function from the graph."""
    context = {
        'calls': [],
        'called_by': [],
        'file_imports': [],
        'related_functions': []
    }
    
    if not graph:
        return context
    
    # Normalize file path for comparison
    def normalize_path(p):
        if p:
            return str(Path(p)).lower().replace('\\', '/').split('dataset/')[-1] if 'dataset' in str(p) else str(p)
        return None
    
    normalized_search_path = normalize_path(file_path)
    
    # Find the function node
    func_node_id = None
    matched_file_path = None
    for node_id, node in graph['nodes'].items():
        if node['type'] == 'function' and node['name'] == function_name:
            node_file_normalized = normalize_path(node['file'])
            if file_path is None or node_file_normalized == normalized_search_path or (normalized_search_path and normalized_search_path in node_file_normalized):
                func_node_id = node_id
                matched_file_path = node['file']
                break
    
    if not func_node_id:
        return context
    
    # Find what this function calls
    for edge in graph['edges']:
        if edge['from'] == func_node_id and edge['type'] == 'calls':
            called_node = graph['nodes'][edge['to']]
            context['calls'].append({
                'name': called_node['name'],
                'file': called_node.get('file')
            })
        
        # Find what calls this function
        if edge['to'] == func_node_id and edge['type'] == 'calls':
            caller_node = graph['nodes'][edge['from']]
            context['called_by'].append({
                'name': caller_node['name'],
                'file': caller_node.get('file')
            })
    
    # Get file context - try to find the file in graph
    file_to_lookup = matched_file_path or file_path
    if file_to_lookup and 'files' in graph:
        # Try exact match first
        if file_to_lookup in graph['files']:
            file_info = graph['files'][file_to_lookup]
        else:
            # Try normalized match
            file_info = None
            for graph_file, info in graph['files'].items():
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