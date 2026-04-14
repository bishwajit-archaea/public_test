"""
Line Highlighter - Find and highlight exact matching lines within code chunks

Uses LLM to intelligently identify the specific lines within a code chunk 
that match the search query and highlights them in the output.
"""

import re
import os
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
import json
from app.services.llm.utils import retry_and_log_llm_usage
from app.core.logger import logger


class LineHighlighter:
    """
    Highlights specific lines within code chunks that match the query using LLM.
    """
    
    def __init__(self,
                client:OpenAI,
                model: str,):
        """
        Initialize Line Highlighter.
        
        Args:
            client: OpenAI client
            model: OpenAI model name to use
        """
        self.client = client
        self.model = model
    
    @retry_and_log_llm_usage(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def _call_llm_for_line_matching(self, system_prompt: str, user_prompt: str) -> Tuple[Any, Dict]:
        """
        Call LLM to find matching lines with retry and token logging.
        
        Returns:
            Tuple containing (raw_response, processed_response)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )
        
        llm_response = json.loads(response.choices[0].message.content)
        return response, llm_response
    
    def find_matching_lines(
        self,
        code: str,
        query: str,
        start_line: int = 1,
        context_lines: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find lines in code that match the query using LLM.
        
        Args:
            code: The code to search
            query: Search query
            start_line: Starting line number of the chunk
            context_lines: Number of context lines to show around matches
        
        Returns:
            List of matching line groups with context
        """
        # If no LLM client, return empty
        if not self.client:
            return []
        
        lines = code.split('\n')
        
        # First, do a quick keyword scan to find potentially relevant sections
        query_lower = query.lower()
        keywords = ['transaction', 'log', 'vista', 'query', 'select', 'runquery', 'sql', 'database', 'fetch', 'get']
        
        # Score each line by keyword matches
        line_scores = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for kw in keywords if kw in line_lower)
            if score > 0:
                line_scores.append((i, score, line))
        
        # Sort by score and take top sections
        line_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build focused code snippet with top matching sections
        if line_scores:
            # Take top 10 matching lines and their context
            focused_lines = set()
            for idx, score, _ in line_scores[:10]:
                # Add the line and 3 lines of context before/after
                for j in range(max(0, idx - 3), min(len(lines), idx + 4)):
                    focused_lines.add(j)
            
            # Build numbered code from focused sections
            focused_indices = sorted(focused_lines)
            numbered_code = '\n'.join([f"{start_line + i}: {lines[i]}" for i in focused_indices])
        else:
            # Fallback to first part of code
            numbered_code = '\n'.join([f"{start_line + i}: {line}" for i, line in enumerate(lines[:100])])
        
        # Ask LLM to identify relevant lines
        system_prompt = """You are a code analysis expert. Identify the specific line numbers in the code that are most relevant to answering the user's query.

Focus on IMPLEMENTATION, not setup:
- Lines that directly answer the query (e.g., SQL queries, API calls, function executions)
- Actual logic and operations, not imports or declarations
- Where the work happens, not where dependencies are declared

AVOID:
- Import statements (import, require, from)
- Type declarations
- Simple variable declarations without logic

Importance scoring (1-10):
- 10: The exact line that answers the query (e.g., the SQL query itself, the API call execution)
- 8-9: Critical supporting lines (e.g., function that executes the query, conditional logic)
- 5-7: Related implementation (e.g., data processing, variable assignments with logic)
- 1-4: Tangentially related or setup code

Respond in JSON format:
{
    "relevant_lines": [
        {
            "line_number": 123,
            "relevance": "why this line is relevant",
            "importance": 1-10
        }
    ]
}

Return the top 3-5 most relevant IMPLEMENTATION lines, ranked by importance."""
        
        user_prompt = f"""Query: "{query}"

Code:
{numbered_code[:5000]}

Which specific line numbers contain information relevant to this query? Focus on actual implementation, not just imports."""
        
        try:
            # Use the decorated LLM call method
            raw_response, llm_response = self._call_llm_for_line_matching(system_prompt, user_prompt)
            
            relevant_lines = llm_response.get('relevant_lines', [])
            
            # Filter out import/require statements (they're metadata, not answers)
            def is_import_line(line_num):
                if 0 <= line_num - start_line < len(lines):
                    line = lines[line_num - start_line].strip()
                    return (line.startswith('import ') or 
                           line.startswith('from ') or
                           line.startswith('require(') or
                           line.startswith('const ') and 'require(' in line or
                           line.startswith('import{') or
                           line.startswith('import {'))
                return False
            
            # Filter out imports
            relevant_lines = [line for line in relevant_lines if not is_import_line(line['line_number'])]
            
            # Sort by importance (highest first)
            relevant_lines.sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            # Build result groups with context
            result_groups = []
            used_indices = set()
            
            for line_info in relevant_lines[:3]:  # Top 3 by importance (excluding imports)
                line_num = line_info['line_number']
                idx = line_num - start_line
                
                # Skip if out of range or already used
                if idx < 0 or idx >= len(lines) or idx in used_indices:
                    continue
                
                # Get context
                start_idx = max(0, idx - context_lines)
                end_idx = min(len(lines), idx + context_lines + 1)
                
                context_lines_list = []
                for i in range(start_idx, end_idx):
                    context_lines_list.append({
                        'line_num': start_line + i,
                        'line': lines[i],
                        'is_match': i == idx,
                        'relevance': line_info.get('relevance', '') if i == idx else ''
                    })
                    used_indices.add(i)
                
                result_groups.append({
                    'match_line': line_num,
                    'relevance': line_info.get('relevance', ''),
                    'importance': line_info.get('importance', 5),
                    'context': context_lines_list
                })
            
            return result_groups
            
        except Exception as e:
            # Fallback: return empty on error
            return []
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract search terms from query."""
        # Remove common words but keep important ones
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'i', 'am', 'my', 'me'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words but keep words with 2+ chars
        terms = [w for w in words if w not in stop_words and len(w) >= 2]
        
        return terms
    
    def _score_line(self, line: str, query_terms: List[str]) -> float:
        """Score how well a line matches the query terms."""
        line_lower = line.lower()
        score = 0.0
        
        for term in query_terms:
            # Exact word match
            if re.search(rf'\b{re.escape(term)}\b', line_lower):
                score += 2.0
            # Partial match
            elif term in line_lower:
                score += 1.0
        
        # Bonus for multiple terms
        if score > 2:
            score *= 1.5
        
        return score
    
    def format_highlighted_lines(
        self,
        match_groups: List[Dict[str, Any]],
        max_groups: int = 3
    ) -> str:
        """
        Format highlighted lines for display.
        
        Args:
            match_groups: List of match groups from find_matching_lines
            max_groups: Maximum number of groups to show
        
        Returns:
            Formatted string
        """
        if not match_groups:
            return ""
        
        output = []
        
        for i, group in enumerate(match_groups[:max_groups], 1):
            relevance = group.get('relevance', '')
            importance = group.get('importance', 5)
            importance_emoji = '🔥' if importance >= 9 else '⭐' if importance >= 7 else '💡'
            output.append(f"\n   📍 Match {i} {importance_emoji} (Line {group['match_line']}, Importance: {importance}/10): {relevance}")
            
            for line_info in group['context']:
                line_num = line_info['line_num']
                line = line_info['line']
                is_match = line_info['is_match']
                
                # Truncate long lines
                if len(line) > 100:
                    line = line[:100] + "..."
                
                # Format with highlight
                if is_match:
                    output.append(f"   ➤ {line_num:4d} | {line}  ⭐")
                else:
                    output.append(f"     {line_num:4d} | {line}")
        
        return '\n'.join(output)
    
    def add_highlights_to_result(
        self,
        result: Dict[str, Any],
        query: str,
        context_lines: int = 2
    ) -> Dict[str, Any]:
        """
        Add line highlights to a search result.
        
        Args:
            result: Search result dict
            query: Search query
            context_lines: Number of context lines
        
        Returns:
            Result dict with highlights added
        """
        code = result.get('code', '')
        start_line = result.get('start_line', 1)
        
        if not code:
            return result
        
        # Find matching lines
        match_groups = self.find_matching_lines(
            code=code,
            query=query,
            start_line=start_line,
            context_lines=context_lines
        )
        
        # Add to result
        result['line_matches'] = match_groups
        result['has_line_matches'] = len(match_groups) > 0
        
        return result


def main():
    """Test the line highlighter."""
    # You'll need to initialize with actual client and model
    # highlighter = LineHighlighter(client=openai_client, model="gpt-4")
    
    # Test code
    test_code = """import prisma from "@core/helpers/prisma";
import { runQuery } from "@/helper/vistaDb.js";
import { calculateImpact } from "../dashboard/masters/sessions";

const transactionsSync = async function (db, dbConfig) {
  try {
    console.log("syncing transactions from vista");
    
    // Get transaction logs from Vista database
    const vistaTransactions = await runQuery(db, `
      SELECT * FROM transactions 
      WHERE date > DATEADD(day, -7, GETDATE())
    `);
    
    console.log(`Found ${vistaTransactions.length} transactions from Vista`);
    
    // Process transactions
    for (const transaction of vistaTransactions) {
      await prisma.transaction.upsert({
        where: { id: transaction.id },
        update: transaction,
        create: transaction
      });
    }
    
    return { success: true, count: vistaTransactions.length };
  } catch (error) {
    console.error("Error syncing transactions:", error);
    throw error;
  }
};"""
    
    query = "where am I getting the transaction logs from vista?"
    
    logger.info("🔍 Finding matching lines...\n")
    
    # This would work with proper initialization
    # matches = highlighter.find_matching_lines(
    #     code=test_code,
    #     query=query,
    #     start_line=1,
    #     context_lines=2
    # )
    
    # logger.info(f"Found {len(matches)} match groups\n")
    # logger.info(highlighter.format_highlighted_lines(matches))


if __name__ == "__main__":
    main()