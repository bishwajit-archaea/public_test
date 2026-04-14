@retry_and_log_llm_usage(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def _call_llm_for_line_matching_bishwajitself, system_prompt: str, user_prompt: str) -> Tuple[Any, Dict]:
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
