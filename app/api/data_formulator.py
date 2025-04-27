"""
Data Formulator integration API routes.
"""
from fastapi import APIRouter
import subprocess
import sys
import platform
import tempfile

router = APIRouter(prefix="/data-formulator", tags=["data_formulator"])

@router.get("/check")
async def check_data_formulator():
    """
    Check if Data Formulator is installed
    """
    try:
        # Try to import data_formulator
        result = subprocess.run(
            [sys.executable, "-c", "import data_formulator; print('installed')"],
            capture_output=True,
            text=True,
            check=False
        )
        
        is_installed = "installed" in result.stdout.strip()
        return {"installed": is_installed}
    except Exception:
        return {"installed": False}

@router.post("/launch")
async def launch_data_formulator():
    """
    Launch Data Formulator in a separate process
    """
    try:
        # Check if data_formulator is installed
        check_result = await check_data_formulator()
        if not check_result["installed"]:
            return {"success": False, "message": "Data Formulator is not installed. Please install it first."}
        
        # Create a temporary launcher script
        launcher_code = """
import subprocess
import sys
import platform

try:
    # Import data_formulator and run it
    from data_formulator import main
    
    # Launch data formulator
    main()
except Exception as e:
    print(f"Error launching Data Formulator: {str(e)}")
    input("Press Enter to exit...")
"""
        
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(launcher_code)
            temp_path = f.name
        
        # Launch the script in a new window
        if platform.system() == "Windows":
            subprocess.Popen(
                ["start", "python", temp_path],
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # macOS or Linux
            subprocess.Popen(
                ["gnome-terminal", "--", "python3", temp_path],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
        
        return {
            "success": True,
            "message": "Data Formulator launched successfully in a new window."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error launching Data Formulator: {str(e)}"
        }
