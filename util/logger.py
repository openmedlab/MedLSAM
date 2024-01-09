import time
import os

class Logger(object):
    def __init__(self, log_path=None):
        import sys
        if not log_path:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_path = os.path.join(
                'runs', current_time + '_' + socket.gethostname()+'/default.log')
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=64, encoding="utf-8")
 
    def print(self, *message):
        message = ",".join([str(it) for it in message])
        self.terminal.write(str(message) + "\n")
        self.log.write(str(message) + "\n")
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
 
    def close(self):
        self.log.close()