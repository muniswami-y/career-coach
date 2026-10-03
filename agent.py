"""Agent loop: the model decides which tools to call, we execute them, repeat."""
import json
from dataclasses import dataclass

from rag import DocumentStore
from tools import TOOLS, execute_tool

SYSTEM_PROMPT = """You are Career Coach, an expert career and interview assistant.

Process:
1. Ground every claim about the user in their documents: call search_documents before judging their skills or experience.
2. Use web_search only for current external info (market trends, salaries, courses). Don't search for things you can answer from the documents.
3. If documents are missing, say so and ask the user to upload them rather than guessing.

Style:
- Be specific and actionable. Cite which document a fact came from (resume / job description).
- For skill-gap analysis, use sections: Strong matches, Gaps, Quick wins, 30-day learning plan.
- Keep answers concise and well structured with Markdown.
"""


@dataclass
class AgentContext:
    client: object
    model: str
    store: DocumentStore


def run_agent(ctx: AgentContext, messages: list, max_steps: int = 6):
    """Runs the tool-use loop. Mutates `messages`; returns (final_text, trace).

    `messages` holds the conversation WITHOUT the system prompt (added per call).
    """
    trace = []
    for _ in range(max_steps):
        resp = ctx.client.chat.completions.create(
            model=ctx.model,
            max_tokens=2000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", trace

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            output = execute_tool(tc.function.name, args, ctx)
            trace.append({"tool": tc.function.name, "input": args, "output": output[:600]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    return "I hit my step limit. Try a more specific question.", trace
