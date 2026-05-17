from dataclasses import dataclass
from typing import Self

from .executor import run_command

@dataclass(frozen=True)
class GitStamp:
    branch: str
    commit: str
    description: str

    @classmethod
    def create(cls) -> Self:
        return cls(
            branch=get_branch(), commit=get_commit(), description=get_description()
        )


def get_branch() -> str:
    return run_command("git", "rev-parse", "--abbrev-ref", "HEAD")


def get_commit() -> str:
    return run_command("git", "rev-parse", "HEAD")


def get_description() -> str:
    return run_command("git", "describe", "--tags", "--always")


def get_status() -> str:
    return run_command("git", "status", "--porcelain")


def assert_clean_repo() -> None:
    status = get_status()

    if status:
        raise RuntimeError(
            f"Repository contains uncommitted or untracked changes:\n{status}"
        )


if __name__ == "__main__":
    assert_clean_repo()

    stamp = GitStamp.create()

    print(f"branch      : {stamp.branch}")
    print(f"commit      : {stamp.commit}")
    print(f"description : {stamp.description}")
