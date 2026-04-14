from app.services.llm.utils import retry_and_log_llm_usage
from typing import Tuple, Any

@retry_and_log_llm_usage(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def detect_query_type(query: str) -> Tuple[Any, str]:
    """
    Use LLM to detect what type of search is needed.
    
    Returns:
        Tuple containing (raw_response, query_type) where query_type is:
        'security', 'standards', 'explain', 'document', 'dependencies', 'flow', or 'code'
    """
    system_prompt = """Analyze code search queries. Respond with ONE word:

security - security issues, vulnerabilities, secrets, credentials
standards - code quality, best practices, violations
explain - explanation, understanding, "what does this do"
document - generate documentation, JSDoc, comments
dependencies - dependencies, dependency tree, "what depends on"
flow - execution flow, trace, path, "how does X work"
code - specific functionality, "where is", "find code"

Examples:
"hardcoded API keys" → security
"console.log statements" → standards
"explain this function" → explain
"generate documentation" → document
"dependencies of this" → dependencies
"authentication flow" → flow
"where is login function" → code"""

    from app.services.llm import llm_service
    response = llm_service.client.chat.completions.create(
        model=llm_service.get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ],
        temperature=0.1,
        max_tokens=5  # Reduced since we only need one word
    )
    
    result = response.choices[0].message.content.strip().lower()
    
    # More efficient mapping
    query_map = {
        'security': 'security',
        'standards': 'standards', 
        'explain': 'explain',
        'document': 'document',
        'dependencies': 'dependencies',
        'dependency': 'dependencies',
        'flow': 'flow'
    }
    
    for key, value in query_map.items():
        if key in result:
            return response, value
    
    # Default to code search
    return response, 'code'