import dataclasses

import inspect_ai.solver
import inspect_ai.util


@dataclasses.dataclass
class ExecCall:
    """Represents an exec sandbox call."""

    cmd: list[str]
    input: str | None = None
    cwd: str | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    user: str | None = None
    timeout: int | None = None
    timeout_retry: bool = True


@dataclasses.dataclass
class WriteFileCall:
    """Represents a write_file sandbox call."""

    file: str
    contents: str | bytes


SandboxCall = ExecCall | WriteFileCall


@inspect_ai.solver.solver
def ez_solver(
    sandbox_calls: list[SandboxCall],
    answer: str,
) -> inspect_ai.solver.Solver:
    """
    A solver that executes a series of sandbox calls and then sets the completion.

    Args:
        sandbox_calls: List of sandbox calls to execute (exec or write_file)
        answer: The answer to set as the completion
    """

    async def solve(
        state: inspect_ai.solver.TaskState,
        generate: inspect_ai.solver.Generate,  # pyright: ignore[reportUnusedParameter]
    ) -> inspect_ai.solver.TaskState:
        sandbox = inspect_ai.util.sandbox()
        for call in sandbox_calls:
            if isinstance(call, ExecCall):
                _ = await sandbox.exec(
                    call.cmd,
                    input=call.input,
                    cwd=call.cwd,
                    env=call.env,
                    user=call.user,
                    timeout=call.timeout,
                    timeout_retry=call.timeout_retry,
                )
            else:
                await sandbox.write_file(file=call.file, contents=call.contents)

        state.output.completion = answer
        state.completed = True

        return state

    return solve
