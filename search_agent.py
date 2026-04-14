"""
Enhanced search agent with security scanning capability.

Combines:
1. Hybrid search (keyword + semantic)
2. LLM reranking
3. Security pattern scanning
"""

import os
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
import json
from app.services.opensearch.opensearch import DualSearchAgent
from app.core.config import settings_object as settings
from app.core.logger import logger
from app.services.llm.code_rerank_call import rerank_with_llm
from pathlib import Path
# from security_scanner import SecurityScanner
# from code_standards_checker import CodeStandardsChecker
# from code_explainer import CodeExplainer
# from dependency_analyzer import DependencyAnalyzer
from app.services.llm.line_highlighter_call import LineHighlighter
# from error_handling_checker import ErrorHandlingChecker
# from pathlib import Path
# from grok_client import grok_client ,model_id   



class EnhancedSearchAgent:
    """
    Search agent with multiple search strategies including security scanning.
    """
    
    def __init__(
        self,
        client:OpenAI,
        model: str,
        dataset_path :str,
        search_agent: DualSearchAgent,
        current_dataset: str,
        index_name: str = settings.OPENSEARCH_INDEX,
    ):
        """
        Initialize Enhanced Search Agent.
        
        Args:
            llm_service: Instance of LLMService
            model:  model to use
            index_name: OpenSearch index name
            dataset_path: Path to code dataset for security scanning
        """
        
        self.client = client
        self.model = model
        self.dataset_path = dataset_path
        self.current_dataset = current_dataset
        
        # Initialize search agent, security scanner, code standards checker, explainer, dependency analyzer, and line highlighter
        self.search_agent = search_agent
        # self.security_scanner = SecurityScanner()
        # self.standards_checker = CodeStandardsChecker()
        # self.error_checker = ErrorHandlingChecker()
        # self.code_explainer = CodeExplainer(api_key=self.api_key, model=self.model)
        # self.dependency_analyzer = DependencyAnalyzer()
        self.line_highlighter = LineHighlighter(client=self.client,model=self.model)
        
        logger.info(f"✓ Enhanced Search Agent initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Index: {index_name}")
        logger.info(f"  Dataset: {dataset_path}\n")
    
    
    
    async def _security_search(self, query: str,query_type: str,max_results: int = 10) -> Dict[str, Any]:
        """
        Perform security-focused search.
        """
        start_time = time.time()
        logger.info(f"🔒 Running security scan...")
        
        # Scan the dataset
        all_findings = []
        dataset_path = Path(self.dataset_path)

        #hybrid search
        hybrid_results =await  self.search_agent.hybrid_search(
            query=query,
            query_type=query_type,
            current_dataset=self.current_dataset,
            max_results=max_results,
            use_text_semantic=True
        )
        # logger.info(f"🔒 Hybrid Search hybrid_results: {hybrid_results}")

        
        return {
            "query": query,
            "query_type":query_type,
            "search_type": "security",
            "total_hits": len(hybrid_results['results']),
            "results": hybrid_results['results'],
            "original_hits": hybrid_results['total_hits'],
            "keyword_hits": hybrid_results.get('keyword_hits', 0),
            "semantic_hits": hybrid_results.get('semantic_hits', 0),
            "metrics": {
                "search_time": round(start_time, 4),
                "rerank_time": round(time.time() - start_time, 4),
                "total_time": round(time.time() - start_time, 4)
            }
        }
    
    
    def _standards_search(self, query: str) -> Dict[str, Any]:
        """
        Perform code standards check.
        """
        start_time = time.time()
        logger.info(f"📋 Running code standards check...")
        
        # Scan the dataset
        all_violations = []
        dataset_path = Path(self.dataset_path)
        
        if dataset_path.exists():
            results = self.standards_checker.check_directory(str(dataset_path))
            
            for file_path, file_violations in results.items():
                for rule_name, violations in file_violations.items():
                    all_violations.extend(violations)
        
        # Filter violations based on query
        if all_violations:
            filtered = self._filter_standards_violations(query, all_violations)
            return {
                "query": query,
                "search_type": "code_standards",
                "total_hits": len(filtered),
                "results": filtered,
                "metrics": {
                    "check_time": round(time.time() - start_time, 4),
                    "total_time": round(time.time() - start_time, 4)
                }
            }
        
        return {
            "query": query,
            "search_type": "code_standards",
            "total_hits": 0,
            "results": [],
            "metrics": {
                "check_time": round(time.time() - start_time, 4),
                "total_time": round(time.time() - start_time, 4)
            }
        }
    
    def _filter_standards_violations(self, query: str, violations: List[Dict]) -> List[Dict]:
        """Filter standards violations based on query."""
        query_lower = query.lower()
        
        # Map query keywords to rule names
        rule_keywords = {
            'debug_code': ['console.log', 'console', 'debugger', 'debug', 'alert'],
            'magic_values': ['magic number', 'magic string', 'hardcoded', 'constant'],
            'naming_conventions': ['naming', 'camelcase', 'variable name', 'function name'],
            'unused_imports': ['unused import', 'import', 'unused'],
            'var_usage': ['var', 'let', 'const'],
        }
        
        # Find relevant rules
        relevant_rules = set()
        for rule, keywords in rule_keywords.items():
            if any(kw in query_lower for kw in keywords):
                relevant_rules.add(rule)
        
        # If no specific rules found, return all
        if not relevant_rules:
            return violations[:20]
        
        # Filter by relevant rules
        filtered = [v for v in violations if v['rule'] in relevant_rules]
        return filtered[:20]
    
    def _explain_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Find relevant code and explain it.
        """
        logger.info(f"💡 Finding and explaining code...")
        
        # First, find relevant code using hybrid search
        logger.info(f"📊 Searching for relevant code...")
        start_search = time.time()
        hybrid_results = self.search_agent.hybrid_search(
            query=query,
            max_results=max_results,
            use_text_semantic=True
        )
        search_time = time.time() - start_search
        
        if not hybrid_results['results']:
            return {
                "query": query,
                "search_type": "explain",
                "total_hits": 0,
                "results": []
            }
        
        logger.info(f"✓ Found {len(hybrid_results['results'])} code snippets")
        logger.info(f"🤖 Generating explanations...\n")
        
        # Explain each result
        explained_results = []
        explained_result_times={}
        explain_start = time.time()
        for i, result in enumerate(hybrid_results['results'][:max_results], 1):
            logger.info(f"  Explaining {i}/{min(max_results, len(hybrid_results['results']))}...")
            
            # Get graph context if available (will be added by GraphEnhancedAgent)
            graph_context = result.get('graph_context')
            
            # Explain the code
            explanation = self.code_explainer.explain_function(
                function_code=result.get('code', ''),
                function_name=result.get('name', 'unknown'),
                file_path=result.get('file_path'),
                graph_context=graph_context
            )

            explained_result_times[i] = explanation['_llm_time']
            
            # Combine result with explanation
            explained_result = result.copy()
            explained_result['explanation'] = explanation
            explained_results.append(explained_result)
            
        explain_end = time.time()   
        explain_time = explain_end - explain_start
        logger.info(f"✓ Generated {len(explained_results)} explanations in {explain_time:.4f} seconds\n")
        
        return {
            "query": query,
            "search_type": "explain",
            "total_hits": len(explained_results),
            "results": explained_results,
            "original_hits": hybrid_results['total_hits'],
            "metrics": {
                "search_time": round(search_time, 4),
                "explain_time": round(explain_time, 4),
                "total_time": round(time.time() - start_search, 4),
                "explained_result_times": explained_result_times
            }
        }
    
    def _document_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Find relevant code and generate documentation.
        """
        start_total = time.time()
        logger.info(f"📝 Finding code and generating documentation...")
        
        # First, find relevant code using hybrid search
        logger.info(f"📊 Searching for relevant code...")
        start_search = time.time()
        hybrid_results = self.search_agent.hybrid_search(
            query=query,
            max_results=max_results,
            use_text_semantic=True
        )
        
        if not hybrid_results['results']:
            return {
                "query": query,
                "search_type": "document",
                "total_hits": 0,
                "results": []
            }
        
        search_time = time.time() - start_search
        logger.info(f"✓ Found {len(hybrid_results['results'])} code snippets")
        logger.info(f"🤖 Generating documentation...\n")
        
        # Generate docs for each result
        start_llm = time.time()
        documented_results = []
        documented_result_times = {}
        for i, result in enumerate(hybrid_results['results'][:max_results], 1):
            logger.info(f"  Documenting {i}/{min(max_results, len(hybrid_results['results']))}...")
            
            # Generate JSDoc
            documentation = self.code_explainer.generate_jsdoc(
                code=result.get('code', ''),
                function_name=result.get('name', 'unknown'),
                context={'file_name': Path(result.get('file_path', '')).name}
            )
            
            # Combine result with documentation
            documented_result = result.copy()
            documented_result_times[i] = documentation['_llm_time']
            documented_result['documentation'] = documentation
            documented_results.append(documented_result)
        
        logger.info(f"✓ Generated documentation for {len(documented_results)} items\n")
        llm_time = time.time() - start_llm
        
        return {
            "query": query,
            "search_type": "document",
            "total_hits": len(documented_results),
            "results": documented_results,
            "original_hits": hybrid_results['total_hits'],
            "metrics": {
                "search_time": round(search_time, 4),
                "document_time": round(llm_time, 4),
                "documented_result_times": documented_result_times,
                "total_time": round(time.time() - start_total, 4)
            }
        }
    
    def _dependencies_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Find code and show its dependency tree.
        """
        start_total = time.time()
        logger.info(f"🌳 Finding code and analyzing dependencies...")
        
        # First, find relevant code using hybrid search
        logger.info(f"📊 Searching for relevant code...")
        start_search = time.time()
        hybrid_results = self.search_agent.hybrid_search(
            query=query,
            max_results=max_results,
            use_text_semantic=True
        )
        
        if not hybrid_results['results']:
            return {
                "query": query,
                "search_type": "dependencies",
                "total_hits": 0,
                "results": []
            }
        
        search_time = time.time() - start_search
        logger.info(f"✓ Found {len(hybrid_results['results'])} code snippets")
        logger.info(f"🔍 Analyzing dependency trees...\n")
        
        # Analyze dependencies for each result
        start_llm = time.time()
        dependency_results = []
        for i, result in enumerate(hybrid_results['results'][:max_results], 1):
            logger.info(f"  Analyzing {i}/{min(max_results, len(hybrid_results['results']))}...")
            
            function_name = result.get('name', 'unknown')
            file_path = result.get('file_path')
            
            # Get full dependency tree
            dep_tree = self.dependency_analyzer.get_full_dependency_tree(
                function_name=function_name,
                file_path=file_path,
                max_depth=3
            )
            
            # Combine result with dependency tree
            dep_result = result.copy()
            dep_result['dependency_tree'] = dep_tree
            dependency_results.append(dep_result)
        
        logger.info(f"✓ Analyzed {len(dependency_results)} dependency trees\n")
        llm_time = time.time() - start_llm
        
        return {
            "query": query,
            "search_type": "dependencies",
            "total_hits": len(dependency_results),
            "results": dependency_results,
            "original_hits": hybrid_results['total_hits'],
            "metrics": {
                "search_time": round(search_time, 4),
                "llm_time": round(llm_time, 4),
                "total_time": round(time.time() - start_total, 4)
            }
        }
    
    def _flow_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Find code and trace its execution flow.
        """
        start_total = time.time()
        logger.info(f"🔄 Finding code and tracing execution flow...")
        
        # First, find relevant code using hybrid search
        logger.info(f"📊 Searching for relevant code...")
        start_search = time.time()
        hybrid_results = self.search_agent.hybrid_search(
            query=query,
            max_results=max_results,
            use_text_semantic=True
        )
        
        if not hybrid_results['results']:
            return {
                "query": query,
                "search_type": "flow",
                "total_hits": 0,
                "results": []
            }
        
        search_time = time.time() - start_search
        logger.info(f"✓ Found {len(hybrid_results['results'])} code snippets")
        logger.info(f"🔍 Tracing execution flows...\n")
        
        # Trace execution flow for each result
        start_llm = time.time()
        flow_results = []
        for i, result in enumerate(hybrid_results['results'][:max_results], 1):
            logger.info(f"  Tracing {i}/{min(max_results, len(hybrid_results['results']))}...")
            
            function_name = result.get('name', 'unknown')
            file_path = result.get('file_path')
            
            # Trace execution flow
            flow = self.dependency_analyzer.trace_execution_flow(
                function_name=function_name,
                file_path=file_path,
                max_depth=5
            )
            
            # Combine result with flow
            flow_result = result.copy()
            flow_result['execution_flow'] = flow
            flow_results.append(flow_result)
        
        logger.info(f"✓ Traced {len(flow_results)} execution flows\n")
        llm_time = time.time() - start_llm
        
        return {
            "query": query,
            "search_type": "flow",
            "total_hits": len(flow_results),
            "results": flow_results,
            "original_hits": hybrid_results['total_hits'],
            "metrics": {
                "search_time": round(search_time, 4),
                "llm_time": round(llm_time, 4),
                "total_time": round(time.time() - start_total, 4)
            }
        }
    
    async def _hybrid_search_with_reranking(
        self,
        query: str,
        query_type: str,
        max_results: int = 10,
        initial_results: int = 30
    ) -> Dict[str, Any]:
        """
        Perform hybrid search with LLM reranking.
        """
        logger.info(f"📊 Running hybrid search...")
        
        # Hybrid search
        start_search = time.time()
        hybrid_results =await  self.search_agent.hybrid_search(
            query=query,
            query_type=query_type,
            current_dataset=self.current_dataset,
            max_results=initial_results,
            use_text_semantic=True
        )
        search_time = time.time() - start_search
        
        logger.info(f"✓ Found {hybrid_results['total_hits']} results")
        logger.info(f"  - Keyword: {hybrid_results.get('keyword_hits', 0)}")
        logger.info(f"  - Semantic: {hybrid_results.get('semantic_hits', 0)}\n")
        
        # LLM Reranking
        logger.info(f"🤖 Reranking with GPT...")
        start_rerank = time.time()
        # reranked = self._rerank_with_llm(query, hybrid_results['results'][:initial_results], max_results)
        reranked = rerank_with_llm(self.client, query,self.model, hybrid_results['results'][:initial_results], max_results)
        rerank_time = time.time() - start_rerank
        
        # Add line highlights to top 3 results only (to save API calls)
        logger.info(f"🔍 Finding exact matching lines in top 3 results...")
        for i, result in enumerate(reranked[:3]):
            self.line_highlighter.add_highlights_to_result(result, query, context_lines=2)
            if result.get('has_line_matches'):
                logger.info(f"   ✓ Found {len(result.get('line_matches', []))} matches in result {i+1}")
        
        return {
            "query": query,
            "query_type":query_type,
            "search_type": "hybrid_with_reranking",
            "total_hits": len(reranked),
            "results": reranked[:1],
            "original_hits": hybrid_results['total_hits'],
            "keyword_hits": hybrid_results.get('keyword_hits', 0),
            "semantic_hits": hybrid_results.get('semantic_hits', 0),
            "metrics": {
                "search_time": round(search_time, 4),
                "rerank_time": round(rerank_time, 4),
                "total_time": round(time.time() - start_search, 4)
            }
        }
     
    async def search(
        self,
        query: str,
        query_type: str,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Smart search that automatically chooses the best strategy.
        
        Args:
            query: Search query
            max_results: Number of results to return
            force_type: Force search type ('security' or 'code'), or None for auto-detect
        """
        logger.info(f"Detected type: {query_type}")
        # # Route to appropriate search
        if query_type == 'security':
            return await self._security_search(query,query_type,max_results)
        elif query_type == 'standards':
            return self._standards_search(query)
        # elif query_type == 'explain':
        #     return self._explain_search(query, max_results=min(max_results, 3))
        # elif query_type == 'document':
        #     return self._document_search(query, max_results=min(max_results, 3))
        # elif query_type == 'dependencies':
        #     return self._dependencies_search(query, max_results=min(max_results, 3))
        # elif query_type == 'flow':
        #     return self._flow_search(query, max_results=min(max_results, 3))
        # else:
        return await  self._hybrid_search_with_reranking(query,query_type, max_results)
    
    def format_results(self, results: Dict[str, Any]):
        """Pretty print results."""
        logger.info(f"{'='*80}")
        logger.info(f"✨ RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Query: '{results['query']}'")
        logger.info(f"Type: {results['search_type']}")
        logger.info(f"Total: {results['total_hits']} results")
        logger.info(f"\n{'-'*80}\n")
        
        if not results['results']:
            logger.info("❌ No results found\n")
            return
        
        # Format based on search type
        if results['search_type'] == 'security_scan':
            self._format_security_results(results['results'])
        elif results['search_type'] == 'code_standards':
            self._format_standards_results(results['results'])
        elif results['search_type'] == 'explain':
            self._format_explain_results(results['results'])
        elif results['search_type'] == 'document':
            self._format_document_results(results['results'])
        elif results['search_type'] == 'dependencies':
            self._format_dependencies_results(results['results'])
        elif results['search_type'] == 'flow':
            self._format_flow_results(results['results'])
        else:
            self._format_code_results(results['results'])
        
        logger.info(f"{'='*80}\n")
    
    def _format_security_results(self, findings: List[Dict]):
        """Format security scan results."""
        for i, finding in enumerate(findings, 1):
            severity_icon = '🔴' if finding['severity'] == 'HIGH' else '🟡' if finding['severity'] == 'MEDIUM' else '🟢'
            
            logger.info(f"{i}. {severity_icon} {finding['severity']} - {finding['type']}")
            logger.info(f"   📁 {Path(finding['file_path']).name}")
            logger.info(f"   📍 Line {finding['line']}")
            logger.info(f"   💻 {finding['full_line']}")
            if finding['secret_value'] and len(finding['secret_value']) > 8:
                masked = finding['secret_value'][:4] + '*' * (len(finding['secret_value']) - 8) + finding['secret_value'][-4:]
                logger.info(f"   🔑 {masked}")
            logger.info()
    
    def _format_standards_results(self, violations: List[Dict]):
        """Format code standards results."""
        for i, v in enumerate(violations, 1):
            severity_icon = '🔴' if v['severity'] == 'high' else '🟡' if v['severity'] == 'medium' else '🟢'
            
            logger.info(f"{i}. {severity_icon} {v['severity'].upper()} - {v['message']}")
            logger.info(f"   📁 {Path(v['file']).name}")
            logger.info(f"   📍 Line {v['line']}")
            logger.info(f"   💻 {v['code'][:100]}")
            logger.info(f"   💡 {v['suggestion']}")
            logger.info()
    
    def _format_explain_results(self, results: List[Dict]):
        """Format code explanation results."""
        for i, result in enumerate(results, 1):
            explanation = result.get('explanation', {})
            
            logger.info(f"\n{i}. 📝 {result.get('file_name', 'N/A')} - {result.get('name', 'N/A')}")
            logger.info(f"   📍 Lines {result.get('start_line', 0)}-{result.get('end_line', 0)}")
            logger.info(f"\n   Summary:")
            logger.info(f"   {explanation.get('summary', 'No summary available')}")
            logger.info(f"\n   Purpose:")
            logger.info(f"   {explanation.get('purpose', 'Unknown')}")
            logger.info(f"\n   How It Works:")
            logger.info(f"   {explanation.get('how_it_works', 'No explanation')[:300]}...")
            
            if explanation.get('inputs'):
                logger.info(f"\n   📥 Inputs: {', '.join(explanation['inputs'][:3])}")
            
            if explanation.get('outputs'):
                logger.info(f"   📤 Outputs: {explanation['outputs'][:100]}")
            
            if explanation.get('dependencies'):
                logger.info(f"   🔗 Dependencies: {', '.join(explanation['dependencies'][:5])}")
            
            if explanation.get('potential_issues'):
                logger.info(f"\n   ⚠️  Potential Issues:")
                for issue in explanation['potential_issues'][:2]:
                    logger.info(f"      • {issue}")
            
            if explanation.get('suggestions'):
                logger.info(f"\n   💡 Suggestions:")
                for suggestion in explanation['suggestions'][:2]:
                    logger.info(f"      • {suggestion}")
            
            logger.info()
    
    def _format_document_results(self, results: List[Dict]):
        """Format documentation generation results."""
        for i, result in enumerate(results, 1):
            doc = result.get('documentation', {})
            
            logger.info(f"\n{i}. 📝 {result.get('file_name', 'N/A')} - {result.get('name', 'N/A')}")
            logger.info(f"   📍 Lines {result.get('start_line', 0)}-{result.get('end_line', 0)}")
            logger.info(f"\n   Generated JSDoc:")
            logger.info(f"   {'-'*70}")
            
            # Show JSDoc
            jsdoc = doc.get('jsdoc', 'No documentation generated')
            for line in jsdoc.split('\n'):
                logger.info(f"   {line}")
            
            logger.info(f"   {'-'*70}")
            
            # Show summary
            if doc.get('summary'):
                logger.info(f"\n   Summary: {doc['summary']}")
            
            # Show parameters
            if doc.get('parameters'):
                logger.info(f"\n   Parameters:")
                for param in doc['parameters']:
                    logger.info(f"      • {param['name']} ({param['type']}): {param['description']}")
            
            # Show returns
            if doc.get('returns'):
                ret = doc['returns']
                logger.info(f"\n   Returns: {ret.get('type', 'unknown')} - {ret.get('description', 'N/A')}")
            
            # Show examples
            if doc.get('examples'):
                logger.info(f"\n   Examples:")
                for j, example in enumerate(doc['examples'][:2], 1):
                    logger.info(f"      {j}. {example[:100]}...")
            
            logger.info()
    
    def _format_dependencies_results(self, results: List[Dict]):
        """Format dependency tree results."""
        for i, result in enumerate(results, 1):
            dep_tree = result.get('dependency_tree', {})
            
            if dep_tree.get('error'):
                logger.info(f"\n{i}. ❌ {result.get('file_name', 'N/A')} - {result.get('name', 'N/A')}")
                logger.info(f"   Error: {dep_tree['error']}")
                continue
            
            logger.info(f"\n{i}. 🌳 {result.get('file_name', 'N/A')} - {dep_tree.get('function', 'N/A')}")
            logger.info(f"   📍 {Path(dep_tree.get('file', '')).name}")
            
            # Show dependencies
            deps = dep_tree.get('dependencies', {})
            logger.info(f"\n   📥 DEPENDENCIES (What this uses): {deps.get('total', 0)}")
            
            if deps.get('total', 0) > 0:
                tree = deps.get('tree', {})
                for level in tree.get('levels', [])[:3]:  # Show first 3 levels
                    logger.info(f"      Level {level['depth']}: {level['count']} items")
                    for item in level['items'][:3]:  # Show first 3 items per level
                        logger.info(f"         • {item['name']} ({Path(item['file']).name})")
                    if level['count'] > 3:
                        logger.info(f"         ... and {level['count'] - 3} more")
            
            # Show dependents
            depts = dep_tree.get('dependents', {})
            logger.info(f"\n   📤 DEPENDENTS (What uses this): {depts.get('total', 0)}")
            
            if depts.get('total', 0) > 0:
                tree = depts.get('tree', {})
                for level in tree.get('levels', [])[:3]:
                    logger.info(f"      Level {level['depth']}: {level['count']} items")
                    for item in level['items'][:3]:
                        logger.info(f"         • {item['name']} ({Path(item['file']).name})")
                    if level['count'] > 3:
                        logger.info(f"         ... and {level['count'] - 3} more")
            
            # Show circular dependencies
            if dep_tree.get('has_circular'):
                logger.info(f"\n   ⚠️  CIRCULAR DEPENDENCIES:")
                for j, cycle in enumerate(dep_tree.get('circular_dependencies', [])[:2], 1):
                    logger.info(f"      {j}. {' → '.join(cycle)}")
            
            logger.info()
    
    def _format_flow_results(self, results: List[Dict]):
        """Format execution flow results."""
        for i, result in enumerate(results, 1):
            flow = result.get('execution_flow', {})
            
            if flow.get('error'):
                logger.info(f"\n{i}. ❌ {result.get('file_name', 'N/A')} - {result.get('name', 'N/A')}")
                logger.info(f"   Error: {flow['error']}")
                continue
            
            logger.info(f"\n{i}. 🔄 {result.get('file_name', 'N/A')} - {flow.get('function', 'N/A')}")
            logger.info(f"   📍 {Path(flow.get('file', '')).name}")
            logger.info(f"\n   Summary: {flow.get('summary', 'No summary')}")
            logger.info(f"   Total Paths: {flow.get('total_paths', 0)}")
            logger.info(f"   Max Depth: {flow.get('max_depth', 0)}")
            
            # Show common calls
            if flow.get('common_calls'):
                logger.info(f"\n   📊 COMMON CALLS:")
                for call in flow['common_calls'][:3]:
                    logger.info(f"      • {call['name']} - {call['frequency']} paths ({call['percentage']:.0f}%)")
            
            # Show sample execution paths
            if flow.get('paths'):
                logger.info(f"\n   🔀 SAMPLE EXECUTION PATHS:")
                for j, path in enumerate(flow['paths'][:2], 1):
                    logger.info(f"\n      Path {j} ({len(path)} steps):")
                    for k, step in enumerate(path[:5]):  # Show first 5 steps
                        indent = "      " + ("  " * k)
                        arrow = "└─>" if k == len(path) - 1 or k == 4 else "├─>"
                        logger.info(f"{indent}{arrow} {step['name']} ({step['file']})")
                    if len(path) > 5:
                        logger.info(f"         ... and {len(path) - 5} more steps")
            
            logger.info()
    
    def _format_code_results(self, results: List[Dict]):
        """Format code search results."""
        for i, hit in enumerate(results, 1):
            score = hit.get('llm_relevance_score', hit.get('combined_score', 0))
            logger.info(f"{i}. 🎯 {score:.0%} - {hit['file_name']}")
            logger.info(f"   📍 {hit.get('name', 'N/A')} (Lines {hit['start_line']}-{hit['end_line']})")
            
            # Show line highlights if available
            if hit.get('has_line_matches') and hit.get('line_matches'):
                logger.info(self.line_highlighter.format_highlighted_lines(hit['line_matches'], max_groups=2))
            else:
                logger.info(f"   💻 {hit['code'][:120]}...")
            
            logger.info()


def main():
    """Test the enhanced search agent."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("❌ Set OPENAI_API_KEY environment variable")
        return
    
    agent = EnhancedSearchAgent()
    
    # Test queries
    queries = [
        ("hardcoded secret keys", None),  # Should trigger security scan
        ("function that handles authentication", None),  # Should trigger code search
    ]
    
    for query, force_type in queries:
        results = agent.search(query, max_results=10, force_type=force_type)
        agent.format_results(results)
        logger.info("\n" + "="*80 + "\n")
        input("Press Enter for next query...")


if __name__ == "__main__":
    main()
