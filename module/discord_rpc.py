import struct
import json
import time
import threading
import builtins
import os

class DiscordRPC:
    def __init__(self, client_id):
        self.client_id = str(client_id)
        self.pipe = None
        self.connected = False
        self._lock = threading.Lock()
        
        self._last_connect_attempt = 0
        self._connect_cooldown = 10.0
        self._last_activity = None

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
            
        now = time.time()
        if now - self._last_connect_attempt < self._connect_cooldown:
            return
            
        self._last_connect_attempt = now
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
        self._send_payload(1, payload_dict)
        
        with self._lock:
            if self.pipe:
                try:
                    self.pipe.close()
                except:
                    pass
                self.pipe = None
            self.connected = False
