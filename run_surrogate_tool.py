import threading
import time
import webbrowser


def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:5001")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    from app import create_app
    app = create_app()
    app.run(debug=False, port=5001)
