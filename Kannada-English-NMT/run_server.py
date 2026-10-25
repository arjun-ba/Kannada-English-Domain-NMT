"""
Single Entrypoint Runner: Kannada-English Domain NMT Server
Launches the FastAPI Gateway & Interactive Studio.
"""

import sys
import os
import webbrowser
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 70)
    print(" 🌾 IndicTrans: Kannada-English Domain NMT Microservices Studio")
    print("=" * 70)
    print(f" [*] Python Runtime : {sys.version.split()[0]}")
    print(f" [*] Server Address : {url}")
    print(f" [*] Interactive UI : {url}")
    print(f" [*] API Docs / Swagger : {url}/docs")
    print(f" [*] Domain Scope   : Agriculture & Rural Science")
    print("=" * 70)
    print(" Starting Uvicorn Gateway Server...\n")

    # Launch browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("api_gateway.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
