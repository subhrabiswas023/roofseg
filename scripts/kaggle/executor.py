from subprocess import run, CalledProcessError


def run_command(*args: str) -> str:
    try:
        result = run(args, check=True, capture_output=True, text=True)
        return result.stdout.strip() or result.stderr.strip()
    
    except CalledProcessError as e:
        error_message = (
            e.stderr
            or e.stdout
            or "Unknown command error"
        ).strip()
        
        failed_command = " ".join(map(str, e.cmd))
        
        raise RuntimeError(f"Command failed: {failed_command}\n{error_message}") from e
