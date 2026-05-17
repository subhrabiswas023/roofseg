from subprocess import run, CalledProcessError


def run_command(*args: str) -> str:
    try:
        result = run(args, check=True, capture_output=True, text=True)
        return result.stdout.strip() or result.stderr.strip()
    
    except CalledProcessError as e:
        error_message = (
            e.stderr.strip()
            or e.stdout.strip()
            or "Unknown command error"
        )
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{error_message}") from e
