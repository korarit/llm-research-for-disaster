import os
import time
import json
import random
import ast
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Setup API keys and clients
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY
)

typhoon_client = OpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=TYPHOON_API_KEY
)

# Model configuration mapping
MODEL_MAPPING = {
    "gemma-4": {
        "client": openrouter_client,
        "model_id": "google/gemma-4-26b-a4b-it",
    },
    "deepseek-v4-flash": {
        "client": openrouter_client,
        "model_id": "deepseek/deepseek-v4-flash",
    },
    "typhoon-v2.5": {
        "client": typhoon_client,
        "model_id": "typhoon-v2.5-30b-a3b-instruct",
    }
}

def repair_json_string(s: str) -> str:
    """
    Repairs unescaped double quotes inside JSON string values.
    """
    in_string = False
    escaped = False
    result = []
    i = 0
    n = len(s)
    
    while i < n:
        char = s[i]
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
            elif char == '\\':
                result.append(char)
                escaped = True
            elif char == '"':
                # Look ahead to see if this is the closing quote.
                is_closing = False
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                if j >= n:
                    is_closing = True
                elif s[j] in ('}', ']', ':'):
                    is_closing = True
                elif s[j] == ',':
                    # If it's a comma, look ahead to the next non-whitespace character.
                    k = j + 1
                    while k < n and s[k].isspace():
                        k += 1
                    if k >= n or s[k] in ('"', '{', '['):
                        is_closing = True
                
                if is_closing:
                    result.append(char)
                    in_string = False
                else:
                    result.append('\\"')
            else:
                result.append(char)
        else:
            if char == '"':
                in_string = True
            result.append(char)
        i += 1
        
    return "".join(result)

def call_llm(model_name: str, system_prompt: str, user_prompt: str, schema: dict, function_name: str, temperature: float = 0.0, retries: int = 5) -> tuple[dict, int, int, float]:
    """
    Unified function call to openrouter/typhoon LLMs with tool choice for JSON schema parsing.
    Returns:
        tuple: (parsed_json_dict, prompt_tokens, completion_tokens, latency_seconds)
    """
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from: {list(MODEL_MAPPING.keys())}")
        
    cfg = MODEL_MAPPING[model_name]
    client = cfg["client"]
    model_id = cfg["model_id"]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": function_name,
                "description": f"Outputs JSON object fitting the schema for {function_name}",
                "parameters": schema
            }
        }
    ]
    tool_choice = {"type": "function", "function": {"name": function_name}}
    
    backoff = 1.0
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=4096,
                timeout=60.0
            )
            latency = time.time() - start_time
            
            prompt_tokens = response.usage.prompt_tokens if (response and response.usage) else 0
            completion_tokens = response.usage.completion_tokens if (response and response.usage) else 0
            
            if response is None:
                raise ValueError("API returned None response.")
                
            if not getattr(response, 'choices', None):
                # Try to output raw response dict or representation if possible
                try:
                    resp_str = str(response)
                except:
                    resp_str = "unrepresentable response object"
                raise ValueError(f"API response has no choices or choices is None. Response: {resp_str}")
                
            if len(response.choices) == 0:
                raise ValueError("API response choices list is empty.")
                
            choice = response.choices[0]
            if not choice:
                raise ValueError("First choice in response is None.")
                
            if not getattr(choice, 'message', None):
                raise ValueError("First choice in response has no message attribute.")
                
            if not getattr(choice.message, 'tool_calls', None):
                # Fallback: Try to parse JSON from choice.message.content
                content = choice.message.content
                if content:
                    cleaned_content = content.strip()
                    
                    # Handle <tool_call> tag
                    if "<tool_call>" in cleaned_content:
                        parts = cleaned_content.split("<tool_call>")
                        json_candidate = parts[-1].strip()
                        if "</tool_call>" in json_candidate:
                            json_candidate = json_candidate.split("</tool_call>")[0].strip()
                        cleaned_content = json_candidate
                        
                    if cleaned_content.startswith("```"):
                        lines = cleaned_content.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        cleaned_content = "\n".join(lines).strip()
                        
                    def extract_args_from_dict(d):
                        if "arguments" in d:
                            args = d["arguments"]
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    try:
                                        args = ast.literal_eval(args)
                                    except Exception:
                                        pass
                            if isinstance(args, dict):
                                return args
                        return d

                    repaired_content = repair_json_string(cleaned_content)
                    try:
                        parsed_json = json.loads(repaired_content)
                        if isinstance(parsed_json, dict):
                            return extract_args_from_dict(parsed_json), prompt_tokens, completion_tokens, latency
                    except Exception:
                        try:
                            parsed_json = ast.literal_eval(repaired_content)
                            if isinstance(parsed_json, dict):
                                return extract_args_from_dict(parsed_json), prompt_tokens, completion_tokens, latency
                        except Exception:
                            pass
                
                content_snippet = str(content)[:200] if content else "None"
                raise ValueError(f"No tool call returned by the model. Content returned: {content_snippet}")
                
            tool_call = choice.message.tool_calls[0]
            if not tool_call or not getattr(tool_call, 'function', None):
                raise ValueError("First tool call is None or has no function attribute.")
                
            arg_str = tool_call.function.arguments or ""
            repaired_arg_str = repair_json_string(arg_str)
            try:
                parsed_json = json.loads(repaired_arg_str)
            except Exception as first_err:
                try:
                    parsed_json = ast.literal_eval(repaired_arg_str)
                except Exception:
                    print(f"DEBUG: Failed to parse tool arguments: {arg_str!r}")
                    raise first_err
            return parsed_json, prompt_tokens, completion_tokens, latency
                
        except Exception as e:
            print(f"Error calling {model_name} (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                # On final fail, return empty dict and zero metrics
                return {}, 0, 0, 0.0
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2.0
            
    return {}, 0, 0, 0.0
