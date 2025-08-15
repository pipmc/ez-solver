from typing import Any, Literal, TypedDict

import inspect_ai.solver
import inspect_ai.util


class SandboxCall(TypedDict):
    type: Literal["exec", "write_file"]
    args: dict[str, Any]  # pyright: ignore[reportExplicitAny]


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
            if call["type"] == "exec":
                _ = await sandbox.exec(**call["args"])  # pyright: ignore[reportAny]
            else:
                await sandbox.write_file(**call["args"])  # pyright: ignore[reportAny]

        state.output.completion = answer
        state.completed = True

        return state

    return solve
