from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    passed: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def run_python_tests(code: str, tests: list[str], timeout_seconds: float = 5.0) -> ExecutionResult:
    """Run generated code in a subprocess with a timeout.

    This is a basic local safeguard, not a hardened security boundary. Run untrusted model code
    inside a container or dedicated sandbox in production experiments.
    """
    program = code.rstrip() + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="qwen3-code-eval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=directory,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                passed=False,
                returncode=None,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                timed_out=True,
            )
    return ExecutionResult(
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
