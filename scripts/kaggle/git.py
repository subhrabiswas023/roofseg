from dataclasses import dataclass, field
from typing import Self
from .executor import run_command


def get_branch() -> str:
    return run_command("git", "rev-parse", "--abbrev-ref", "HEAD")


def get_commit() -> str:
    return run_command("git", "rev-parse", "HEAD")


def get_description() -> str:
    return run_command("git", "describe", "--tags", "--always")


@dataclass(frozen=True, slots=True)
class GitStamp:
    branch: str = field(default_factory=get_branch)
    commit: str = field(default_factory=get_commit)
    description: str = field(default_factory=get_description)
    
    @classmethod
    def capture(cls) -> Self:
        return cls(
            branch=get_branch(),
            commit=get_commit(),
            description=get_description(),
        )


def get_status() -> str:
    return run_command("git", "status", "--porcelain")


def assert_clean_repo() -> None:
    status = get_status()

    if status:
        raise RuntimeError(
            f"Repository contains uncommitted or untracked changes:\n{status}"
        )
