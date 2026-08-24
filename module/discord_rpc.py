import struct
import json
import time
import threading
import builtins
import os

PIPE_DIR = "\\\\.\\pipe\\"
IPC_PREFIX = "discord-ipc-"

class DiscordRPC:
    def __init__(self, client_id):
        self.client_id = str(client_id)
        self.pipe = None
        self.connected = False
        self._lock = threading.Lock()

        self._next_attempt_time = 0.0
        self._retry_delay = 10.0
        self._no_discord_retry_delay = 60.0
        self._last_activity = None

    @staticmethod
    def _ipc_available():
        """DiscordのIPCパイプが存在するか(=Discordが起動中か)を調べる。
        判定できない場合は試行を許可するため True を返す。"""
        try:
            names = os.listdir(PIPE_DIR)
        except OSError:
            return True
        return any(name.startswith(IPC_PREFIX) for name in names)

    def _schedule_next_attempt(self, delay):
        self._next_attempt_time = time.time() + delay

    def _read_response(self):
        try:
            header = self.pipe.read(8)
            if not header or len(header) != 8:
                return None
            op, length = struct.unpack("<II", header)
            
            data = b""
            while len(data) < length:
                chunk = self.pipe.read(length - len(data))
                if not chunk:
                    break
                data += chunk
                
            return json.loads(data.decode('utf-8'))
        except Exception:
            return None

    def _connect_task(self):
        with self._lock:
            if self.connected:
                return

        if not self._ipc_available():
            # Discordが起動していない(オフライン環境など)場合は
            # パイプを開く試行自体を行わず、再確認を長い間隔で行う
            self._schedule_next_attempt(self._no_discord_retry_delay)
            return

        for i in range(10):
            pipe_path = f"\\\\.\\pipe\\discord-ipc-{i}"
            try:
                pipe = builtins.open(pipe_path, "r+b", buffering=0)
                
                payload = json.dumps({"v": 1, "client_id": self.client_id}).encode('utf-8')
                header = struct.pack("<II", 0, len(payload))
                pipe.write(header + payload)
                
                self.pipe = pipe
                
                resp = self._read_response()
                if resp and resp.get("evt") == "READY":
                    with self._lock:
                        self.connected = True
                        self._last_activity = None  # Force resend on reconnect
                    return
                    
                pipe.close()
                self.pipe = None
            except Exception:
                pass

    def connect_async(self):
        if self.connected:
            return

        if time.time() < self._next_attempt_time:
            return

        # 接続試行の連鎖を防ぐため、まず標準間隔で次回を予約する
        # (成功時は connected になるため以降の試行は行われない)
        self._schedule_next_attempt(self._retry_delay)
        t = threading.Thread(target=self._connect_task, daemon=True)
        t.start()

    def _send_payload(self, op, payload_dict):
        with self._lock:
            if not self.connected or not self.pipe:
                return False
            pipe = self.pipe
            
        try:
            payload = json.dumps(payload_dict).encode('utf-8')
            header = struct.pack("<II", op, len(payload))
            pipe.write(header + payload)
            
            def _read_discard():
                self._read_response()
            threading.Thread(target=_read_discard, daemon=True).start()
            
            return True
        except Exception:
            with self._lock:
                self.connected = False
                if self.pipe:
                    try:
                        self.pipe.close()
                    except:
                        pass
                    self.pipe = None
            return False

    def _update_task(self, details, state, large_image, large_text, start_time):
        activity = {}
        if details:
            activity["details"] = details
        if state:
            activity["state"] = state
        
        if start_time is not None:
            activity["timestamps"] = {"start": int(start_time)}

        assets = {}
        if large_image:
            assets["large_image"] = large_image
        if large_text:
            assets["large_text"] = large_text
            
        if assets:
            activity["assets"] = assets
            
        with self._lock:
            if self._last_activity == activity:
                return
            self._last_activity = activity.copy()

        payload_dict = {
            "cmd": "SET_ACTIVITY",
            "args": {
                "pid": os.getpid(),
                "activity": activity
            },
            "nonce": str(time.time())
        }
        
        self._send_payload(1, payload_dict)

    def update(self, details=None, state=None, large_image=None, large_text=None, start_time=None):
        if not self.connected:
            self.connect_async()
            return
            
        t = threading.Thread(
            target=self._update_task, 
            args=(details, state, large_image, large_text, start_time), 
            daemon=True
        )
        t.start()

    def close(self):
        payload_dict = {
            "cmd": "SET_ACTIVITY",
            "args": {
                "pid": os.getpid(),
                "activity": None
            },
            "nonce": str(time.time())
        }
        
        def _close_task():
            self._send_payload(1, payload_dict)
            with self._lock:
                if self.pipe:
                    try:
                        self.pipe.close()
                    except:
                        pass
                    self.pipe = None
                self.connected = False
                self._next_attempt_time = 0.0
                
        threading.Thread(target=_close_task, daemon=True).start()
