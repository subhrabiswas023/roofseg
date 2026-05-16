from dataclasses import dataclass
from subprocess import run, CalledProcessError

@dataclass(frozen=True)
class GitStamp:
    branch: str
    commit: str
    description: str
    
def get_branch() -> str:
    return run_git("rev-parse", "--abbrev-ref", "HEAD")
    
def get_commit() -> str:
    return run_git("rev-parse", "HEAD")
    
def get_description() -> str:
    return run_git("describe", "--tags", "--always")
    
def get_stamp() -> GitStamp:
    return GitStamp(
        branch=get_branch(),
        commit=get_commit(),
        description=get_description()
    )
    
def run_git(*args: str) -> str:
    try:
        result = run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout.strip()
    except CalledProcessError as e:
        raise RuntimeError(
            f"Git command failed: git {" ".join(args)}\n"
            f"{e.stderr.strip()}"
        ) from e
     
def get_status() -> str:
    return run_git("status", "--porcelain")
        
def assert_clean_repo() -> None:
    status = get_status()
    
    if status:
        raise RuntimeError(
            "Repository contains uncommitted or untracked changes:\n"
            f"{status}"
        )
        
if __name__ == "__main__":
    assert_clean_repo()
    
    stamp = get_stamp()

    print(f"branch      : {stamp.branch}")
    print(f"commit      : {stamp.commit}")
    print(f"description : {stamp.description}")