# =====================================================================
# CRITICAL DEPLOYMENT ARCHITECTURE REQUIREMENT
# =====================================================================
# The procedural environment bootstrap logic below MUST execute in the 
# global scope before any local package imports are evaluated.
# =====================================================================

from pathlib import Path
import subprocess
import sys

IS_KAGGLE = Path("/kaggle/working").exists()

if IS_KAGGLE:
    print("Kaggle environment detected. Syncing the remote environment...")
    
    extra_dependencies = [
        "segmentation-models-pytorch==0.3.3",
        "kornia==0.7.0"
    ]
    
    print("Installing requirements...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *extra_dependencies]
    )
    
    print("Remote environment sync complete.")
    


# =====================================================================
# LINTER & FORMATTER EXCEPTIONS
# =====================================================================
# DO NOT move the main package import to the top of this file. Doing so will
# cause stickytape to cluster third-party dependencies above the pip 
# installer block, resulting in a remote boot-time ModuleNotFoundError.
# =====================================================================
# If using any import sorter add exceptions to this line as well

# isort: skip
# fmt: skip
from roofseg.pipeline import run_training  # noqa: E402, I001
        
run_training()