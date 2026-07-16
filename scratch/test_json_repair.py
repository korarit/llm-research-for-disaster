import json
import ast

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

# Test cases
test_cases = [
    # Case 1: Simple nested quotes
    '{"message_more_detail": "ผู้ประสบภัย "สมศักดิ์" ต้องการความช่วยเหลือ", "contact_victim": []}',
    # Case 2: Nested quotes with comma
    '{"message_more_detail": "ผู้ประสบภัย "สมศักดิ์", ต้องการความช่วยเหลือ", "contact_victim": []}',
    # Case 3: URL and normal JSON
    '{"google_map_url": "https://maps.app.goo.gl/abc", "lat": 13.75, "lng": 100.5}',
    # Case 4: Already escaped quotes
    '{"message_more_detail": "ผู้ประสบภัย \\"สมศักดิ์\\" ต้องการความช่วยเหลือ"}',
    # Case 5: Nested quotes and trailing quotes
    '{"message_more_detail": "น้ำท่วมสูงมากที่ \\"ม.อุบลฯ\\" และ \\"ต.วารินชำราบ\\""}'
]

for idx, tc in enumerate(test_cases):
    repaired = repair_json_string(tc)
    print(f"--- Test Case {idx+1} ---")
    print(f"Original: {tc}")
    print(f"Repaired: {repaired}")
    try:
        parsed = json.loads(repaired)
        print("Parsed successfully:", parsed)
    except Exception as e:
        print("Failed to parse repaired JSON:", e)
