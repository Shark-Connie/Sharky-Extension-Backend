# Sharky Extension Backend

This is the **Python API Backend** for the Sharky Extension ecosystem. It is designed to run locally and provide computationally intensive AI services and automated processing functions that cannot be natively executed within a browser extension environment.

## 🔗 Architecture & Frontend

This server is built exclusively to support the Sharky browser extension. It must be running for the extension to have full access to its AI features.

👉 **[Sharky-Extension (Frontend) Repository](https://github.com/Shark-Connie/Sharky-Extension)**

### Key Technologies
*   **FastAPI:** High-performance web framework for handling API requests from the extension.
*   **Uvicorn:** Lightning-fast ASGI server.
*   **Python Virtual Environments:** Clean and isolated dependency management.

## Installation & Usage

1.  Ensure you have **Python 3.10+** installed on your system.
2.  Clone or download this repository.
3.  Simply double-click the **`Start.bat`** file (on Windows) or run `npm run start` if you have Node.js configured.

The startup script will automatically:
- Detect or create a Python virtual environment (`venv`).
- Install all necessary dependencies from `requirements.txt`.
- Start the Uvicorn server on `http://127.0.0.1:8000`.

Keep the terminal window open while using the Sharky Extension in your browser. The extension will automatically detect that the server is online.

## Manual Setup (If `Start.bat` is not used)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment (Windows)
.\venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
