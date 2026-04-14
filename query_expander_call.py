"""
Query Expansion using LLM to generate query variations.

Expands user queries with:
- Synonyms
- Technical variations
- Related terms
- Common misspellings
"""

from typing import List, Dict
import json
from app.services.llm.utils import retry_and_log_llm_usage


class QueryExpander:
    """
    Expands search queries to improve recall.
    """
    
    def __init__(self, client=None, model: str = None):
        """Initialize query expander."""
        # Use llm_service client and model if not provided
        self.client = client 
        self.model = model 
    
    @retry_and_log_llm_usage(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def _call_llm_for_expansion(self, system_prompt: str, user_prompt: str) -> tuple:
        """
        Call LLM for query expansion with retry and token logging.
        
        Returns:
            Tuple containing (raw_response, processed_result)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if content.startswith('```json'):
            content = content.split('```json')[1].split('```')[0].strip()
        elif content.startswith('```'):
            content = content.split('```')[1].split('```')[0].strip()
        
        result = json.loads(content)
        return response, result
    
    def expand_query(self, query: str, max_variations: int = 3) -> Dict[str, any]:
        """
        Expand a query into multiple variations.
        
        Args:
            query: Original search query
            max_variations: Maximum number of variations to generate
            
        Returns:
            Dict with original query and variations
        """
        print(f"\n" + "="*80)
        print(f"🔍 Expanding query: '{query}'")
        print(f"📝 Using model: {self.model}")
        print("="*80 + "\n")
        
        system_prompt = """You are a code search query expansion expert. 
Given a search query, generate variations that would help find relevant code.

Generate variations by:
1. Using synonyms (e.g., "auth" → "authentication", "authorize", "login")
2. Adding technical terms (e.g., "validate email" → "email validation regex", "email format check")
3. Using common abbreviations (e.g., "database" → "db", "repository" → "repo")
4. Including related concepts (e.g., "user login" → "session management", "JWT tokens")

Keep variations concise and relevant to code search."""

        user_prompt = f"""Original query: "{query}"

Generate {max_variations} query variations that would help find relevant code.

Respond in JSON format:
{{
    "original": "original query",
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3"
    ],
    "search_terms": ["key", "terms", "to", "search"]
}}"""
        
        try:
            result = self._call_llm_for_expansion(system_prompt, user_prompt)
            
            # Ensure we have the required fields
            if 'variations' not in result:
                result['variations'] = []
            if 'search_terms' not in result:
                result['search_terms'] = query.split()
            if 'original' not in result:
                result['original'] = query
            
            print(f"✅ Generated {len(result['variations'])} variations")
            return result
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {e}")
            # Fallback: return original query
            return {
                "original": query,
                "variations": [],
                "search_terms": query.split()
            }
    
    def get_expanded_queries(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Get list of all queries (original + variations).
        
        Args:
            query: Original query
            max_variations: Max variations to generate
            
        Returns:
            List of queries to search
        """
        expansion = self.expand_query(query, max_variations)
        
        # Combine original + variations
        all_queries = [expansion.get('original', query)]
        all_queries.extend(expansion.get('variations', []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries
    
    def get_search_terms(self, query: str) -> List[str]:
        """
        Extract key search terms from query.
        
        Args:
            query: Search query
            
        Returns:
            List of important search terms
        """
        expansion = self.expand_query(query, max_variations=1)
        return expansion.get('search_terms', query.split())


# Create a default instance
# query_expander = QueryExpander()


if __name__ == "__main__":
    # Test query expansion
    expander = QueryExpander()
    
    test_queries = [
        "where is user authentication handled?",
        "JWT token validation",
        "database connection",
        "validate email address",
        "SQL injection vulnerabilities"
    ]
    
    print("\n" + "="*80)
    print("🔍 QUERY EXPANSION DEMO")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"Original: {query}")
        expansion = expander.expand_query(query)
        
        print(f"Variations:")
        for i, var in enumerate(expansion.get('variations', []), 1):
            print(f"  {i}. {var}")
        
        print(f"Key terms: {', '.join(expansion.get('search_terms', []))}")
        print()