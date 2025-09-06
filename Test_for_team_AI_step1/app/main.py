import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.demo import app

if __name__ == "__main__":
    app()
