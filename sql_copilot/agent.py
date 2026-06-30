"""
Agentic query loop — Gemini iterates: generate SQL, run it, observe
the result, and either refine the query or produce a final answer.
"""

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

from sql_copilot.safety import classify_query, dry_run, execute_query, QuerySafety

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash-lite"
_MAX_ITERATIONS = 5

_SYSTEM_PROMPT = """You are a SQL expert agent with access to a live database.

You have two tools:
- run_query(sql): executes a SQL query against the database and returns the result or an error.
- final_answer(answer): call this when you have enough information to answer the user's question in plain English.

Rules:
- Only use tables/columns from the schema context provided.
- If a query errors, read the error and try a corrected query — don't repeat the same mistake.
- Once you have the data you need, call final_answer with a clear, concise explanation.
- Never call final_answer until you've actually run a query and seen real results.
- If a query would mutate data (INSERT/UPDATE/DELETE/etc.), still propose it via run_query —
  the system will handle safety confirmation; just report what happened.
"""

_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "run_query",
                "description": "Executes a SQL query against the connected database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SQL query to execute."}
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "final_answer",
                "description": "Provides the final answer to the user's question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final plain-English answer."}
                    },
                    "required": ["answer"],
                },
            },
        ]
    }
]


class SQLAgent:
    def __init__(self, engine: Engine, console=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.client = genai.Client(api_key=api_key)
        self.engine = engine
        self.console = console  # optional rich console for live logging

    def _log(self, msg: str):
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def _handle_run_query(self, sql: str) -> str:
        """Runs a query through the safety gate and returns a result string for the model."""
        check = classify_query(sql)
        self._log(f"[cyan]Agent wants to run:[/cyan] {sql}")
        self._log(f"[cyan]Classification:[/cyan] {check.safety.value}")

        if check.safety == QuerySafety.READ_ONLY:
            success, result = execute_query(self.engine, sql, check.safety)
            if success:
                return json.dumps({"success": True, "rows": result[:20]})  # cap to avoid huge payloads
            else:
                return json.dumps({"success": False, "error": result})

        else:
            ok, msg = dry_run(self.engine, sql)
            if not ok:
                return json.dumps({"success": False, "error": msg})

            confirm = self._confirm_mutation(sql)
            if not confirm:
                return json.dumps({"success": False, "error": "User declined to run this mutating query."})

            success, result = execute_query(self.engine, sql, check.safety)
            if success:
                return json.dumps({"success": True, "rows_affected": result})
            else:
                return json.dumps({"success": False, "error": result})

    def _confirm_mutation(self, sql: str) -> bool:
        """Override point — default uses input(); CLI can inject a richer confirm."""
        answer = input(f"\n⚠️  Agent wants to run a MUTATING query:\n{sql}\nAllow? [y/N]: ")
        return answer.strip().lower() == "y"

    def ask(self, user_question: str, schema_context: list[str], memory=None) -> str:
        """Runs the full agent loop and returns the final answer."""
        context_block = "\n".join(schema_context)
        history_block = memory.get_history_text() if memory else ""

        prompt_text = f"{_SYSTEM_PROMPT}\n\nSchema context:\n{context_block}"
        if history_block:
            prompt_text += f"\n\n{history_block}"
        prompt_text += f"\n\nUser question: {user_question}"

        contents = [
            types.Content(role="user", parts=[types.Part(text=prompt_text)])
        ]

        last_sql = ""
        last_success = False
        last_result_summary = ""

        for iteration in range(_MAX_ITERATIONS):
            response = self.client.models.generate_content(
                model=_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(tools=_TOOLS),
            )

            candidate = response.candidates[0]
            contents.append(candidate.content)

            if not candidate.content.parts:
                # Empty response from the model — retry once by continuing the loop
                continue

            function_call = None
            for part in candidate.content.parts:
                if part.function_call:
                    function_call = part.function_call
                    break

            if not function_call:
                answer = candidate.content.parts[0].text
                if memory:
                    memory.add_turn(user_question, answer)
                    memory.log_query(user_question, last_sql, last_success, last_result_summary)
                return answer

            if function_call.name == "final_answer":
                answer = function_call.args.get("answer", "No answer provided.")
                if memory:
                    memory.add_turn(user_question, answer)
                    memory.log_query(user_question, last_sql, last_success, last_result_summary)
                return answer

            if function_call.name == "run_query":
                sql = function_call.args.get("sql", "")
                last_sql = sql
                tool_result = self._handle_run_query(sql)
                result_data = json.loads(tool_result)
                last_success = result_data.get("success", False)
                last_result_summary = tool_result

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name="run_query",
                                    response={"result": tool_result},
                                )
                            )
                        ],
                    )
                )

        return "Reached max iterations without a final answer."