import runpy
from pathlib import Path

# Entry point wrapper to allow running 'streamlit run app.py' directly from the root
frontend_app = Path(__file__).parent / "frontend" / "app.py"
runpy.run_path(str(frontend_app), run_name="__main__")
