from __future__ import annotations

FINAL_ANSWER_TAG = "Final answer:"

MATH_SYSTEM_PROMPT = (
    "Solve the problem carefully. Show concise reasoning, then end with exactly one line "
    f"in the form `{FINAL_ANSWER_TAG} <answer>`."
)

CODE_SYSTEM_PROMPT = (
    "Write a correct Python solution. Return only one Python code block containing the requested "
    "function. Do not include tests or usage examples."
)


def math_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def code_messages(problem: str, tests: list[str] | None = None) -> list[dict[str, str]]:
    test_hint = ""
    if tests:
        test_hint = "\n\nThe function should satisfy examples such as:\n" + "\n".join(tests)
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip() + test_hint},
    ]


def gsm8k_reference(answer: str) -> str:
    reasoning, separator, final = answer.rpartition("####")
    if not separator:
        return answer.strip()
    return f"{reasoning.strip()}\n{FINAL_ANSWER_TAG} {final.strip()}"


def fenced_python(code: str) -> str:
    return f"```python\n{code.strip()}\n```"
