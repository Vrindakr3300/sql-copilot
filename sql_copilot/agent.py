"""
Query generation — takes a natural-language question plus retrieved
schema context, and asks Gemini to generate a SQL query.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash-lite"

_SYSTEM_PROMPT = """You are a SQL expert assistant. Given a database schema and a user's
question in plain English, generate a single, correct SQL query that answers it.

Rules:
- Only use tables and columns that appear in the provided schema context.
- Return ONLY the raw SQL query, no markdown formatting, no explanation, no backticks.
- If the question is ambiguous or cannot be answered with the given schema, return:
  -- CANNOT_ANSWER: <brief reason>
"""


class QueryGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.client = genai.Client(api_key=api_key)

    def generate_sql(self, user_question: str, schema_context: list[str]) -> str:
        """Generates a SQL query given a question and relevant schema chunks."""
        context_block = "\n".join(schema_context)

        prompt = f"""{_SYSTEM_PROMPT}

Schema context:
{context_block}

User question: {user_question}

SQL query:"""

        response = self.client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
        )

        sql = response.text.strip()
        # Strip markdown code fences if Gemini adds them despite instructions
        if sql.startswith("```"):
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()

        return sql