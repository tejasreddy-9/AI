# if you wish to expose a REST endpoint:
from fastapi import Response, status
from typing import Dict

@app.post("/memory/add")
def add_to_memory(payload: Dict[str, str]):
    """
    Expected JSON: { "role": "user|assistant", "content": "Some text" }
    """
    role = payload.get("role")
    content = payload.get("content")
    if not role or not content:
        return Response(
            content="Missing 'role' or 'content' in payload",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    memory_tool.memory.add_message(role, content)
    return {"status": "stored"}
