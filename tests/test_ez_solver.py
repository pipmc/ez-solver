from typing import TypedDict

import inspect_ai
import inspect_ai.dataset
import inspect_ai.scorer
import inspect_ai.solver
import inspect_ai.util
import pytest

import ez_solver


class FileCheckResult(TypedDict):
    correct: bool
    error: str | None
    contents: str | bytes | None


@inspect_ai.scorer.scorer(metrics=[inspect_ai.scorer.accuracy()])
def expected_files_scorer(
    expected_files: dict[str, str | bytes],
) -> inspect_ai.scorer.Scorer:
    async def score(
        state: inspect_ai.solver.TaskState,  # pyright: ignore[reportUnusedParameter]
        target: inspect_ai.scorer.Target,  # pyright: ignore[reportUnusedParameter]
    ) -> inspect_ai.scorer.Score:
        sandbox = inspect_ai.util.sandbox()

        actual: dict[str, FileCheckResult] = {}
        for filename, content in expected_files.items():
            result: FileCheckResult = {"correct": False, "error": None, "contents": None}
            try:
                actual_content = await sandbox.read_file(
                    filename, text=isinstance(content, str)
                )
                result["contents"] = actual_content
                result["correct"] = actual_content == content
            except Exception as e:
                result["correct"] = False
                result["error"] = f"{type(e).__name__}: {str(e)}"
            actual[filename] = result

        score = sum(result["correct"] for result in actual.values()) / len(
            expected_files
        )

        return inspect_ai.scorer.Score(
            value=score,
            metadata={"actual": actual},
        )

    return score


@pytest.mark.asyncio
async def test_sandbox_calls_execution() -> None:
    """Test that sandbox calls are properly configured."""
    solver = ez_solver.ez_solver(
        sandbox_calls=[
            ez_solver.WriteFileCall(
                file="/tmp/test.txt",
                contents="test content",
            ),
            ez_solver.ExecCall(
                cmd=["sh", "-c", "echo -n normal > test2.txt"],
                cwd="/root",
            ),
            ez_solver.ExecCall(
                cmd=["sh", "-c", "echo -n $OUTCOME > test3.txt"],
                cwd="/tmp",
                env={"OUTCOME": "stall"},
            ),
            ez_solver.WriteFileCall(
                file="/tmp/test4.bin",
                contents=b"\x03\x0a\x22",
            ),
        ],
        answer="File operations completed",
    )

    task = inspect_ai.Task(
        dataset=[inspect_ai.dataset.Sample(input="test")],
        solver=solver,
        sandbox="docker",
        scorer=expected_files_scorer(
            expected_files={
                "/tmp/test.txt": "test content",
                "/root/test2.txt": "normal",
                "/tmp/test3.txt": "stall",
                "/tmp/test4.bin": b"\x03\x0a\x22",
            },
        ),
    )
    result = await inspect_ai.eval_async(task)

    assert len(result) == 1
    assert (samples := result[0].samples) and len(samples) == 1

    sample = samples[0]
    assert (scores := sample.scores) and len(scores) == 1

    assert "expected_files_scorer" in scores
    assert scores["expected_files_scorer"].value == 1.0
