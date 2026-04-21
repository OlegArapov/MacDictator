"""MacDictator menu bar tray icon (runs as separate process)."""
import os
import sys
import signal
import rumps


class MacDictatorTray(rumps.App):
    def __init__(self, parent_pid):
        super().__init__("MacDictator", quit_button=None, title="\U0001f3a4")
        self.parent_pid = parent_pid

    @rumps.clicked("Show / Hide")
    def toggle(self, _):
        try:
            os.kill(self.parent_pid, signal.SIGUSR1)
        except ProcessLookupError:
            rumps.quit_application()

    @rumps.clicked("Quit MacDictator")
    def quit_app(self, _):
        try:
            os.kill(self.parent_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        rumps.quit_application()


def _watch_parent(parent_pid):
    """Exit if parent process dies."""
    import threading
    import time

    def _check():
        while True:
            try:
                os.kill(parent_pid, 0)  # check if alive
            except OSError:
                rumps.quit_application()
                return
            time.sleep(2)

    t = threading.Thread(target=_check, daemon=True)
    t.start()


if __name__ == "__main__":
    pid = int(sys.argv[1])
    _watch_parent(pid)
    MacDictatorTray(pid).run()
