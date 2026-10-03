"""Tools the agent can call: schemas + executors."""
import json

from duckduckgo_search import DDGS

_RAW_TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Semantic search over the user's uploaded documents (resume, job description, notes). "
            "Use this whenever you need facts about the user's background or the target role. "
            "Optionally restrict to one source: 'resume' or 'job_description'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "source": {"type": "string", "description": "Optional: 'resume' or 'job_description'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current info: salary ranges, in-demand skills, courses, "
            "company details, industry trends."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "generate_interview_questions",
        "description": (
            "Generate tailored interview questions (with what a strong answer covers) for a role, "
            "grounded in the user's resume."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "focus_areas": {"type": "array", "items": {"type": "string"}},
                "count": {"type": "integer", "minimum": 3, "maximum": 10},
            },
            "required": ["role"],
        },
    },
]


# Groq uses the OpenAI-style function-calling format
TOOLS = [{"type": "function", "function": t} for t in _RAW_TOOLS]


def execute_tool(name: str, args: dict, ctx) -> str:
    try:
        if name == "search_documents":
            hits = ctx.store.query(args["query"], source=args.get("source"))
            if not hits:
                return "No documents indexed yet (or no matches). Ask the user to upload them."
            return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

        if name == "web_search":
            results = DDGS().text(args["query"], max_results=5)
            return json.dumps(
                [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]
            )

        if name == "generate_interview_questions":
            resume = "\n".join(h["text"] for h in ctx.store.query(args["role"], k=4, source="resume"))
            prompt = (
                f"Role: {args['role']}\nFocus areas: {args.get('focus_areas', 'general')}\n"
                f"Candidate resume excerpts:\n{resume or '(none provided)'}\n\n"
                f"Write {args.get('count', 6)} interview questions mixing technical, behavioral and "
                "resume-specific ones. For each, add a one-line 'strong answer covers' note. "
                "Return a numbered list."
            )
            resp = ctx.client.chat.completions.create(
                model=ctx.model,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": "You are a senior hiring manager writing realistic interview questions."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content

        return f"Unknown tool: {name}"
    except Exception as e:  # surface errors to the model so it can recover
        return f"Tool error: {e}"
