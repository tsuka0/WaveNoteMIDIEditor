import sys
import threading
import rtmidi


def _clean_port_name(port, index):
    """Windows環境でrtmidiが付加する末尾のインデックス番号 (' 0', ' 1'...) を除去する。"""
    if sys.platform.startswith("win") and port.endswith(f" {index}"):
        return port[:-len(f" {index}")]
    return port


def list_ports():
    """利用可能なMIDI出力ポート名の一覧を返す。"""
    try:
        out = rtmidi.MidiOut()
        raw_ports = list(out.get_ports())
        clean_ports = [_clean_port_name(p, i) for i, p in enumerate(raw_ports)]

        # 重複がなければ綺麗な名前を使い、同名デバイスがある場合のみ区別番号を付加
        if len(clean_ports) == len(set(clean_ports)):
            ports = clean_ports
        else:
            seen = {}
            for name in clean_ports:
                seen[name] = seen.get(name, 0) + 1
            ports = [
                name if seen[name] == 1 else f"{name} ({i})"
                for i, name in enumerate(clean_ports)
            ]

        if not sys.platform.startswith("win"):
            ports.append("WaveNote Virtual Out")
        return ports
    except Exception:
        return []


class MidiOutDevice:
    def __init__(self):
        self._rtmidi = None
        self._active = set()
        self._port_name = None

    def open(self, name):
        if self._rtmidi is not None and self._rtmidi.is_port_open():
            return True

        try:
            self._rtmidi = rtmidi.MidiOut()

            # Linux / macOS の仮想ポート
            if not sys.platform.startswith("win") and name in ("WaveNote Virtual Out", "WaveNote Out"):
                self._rtmidi.open_virtual_port("WaveNote Out")
                self._port_name = name
                return True

            ports = self._rtmidi.get_ports()

            # 1. 完全一致（生名またはクリーン名）
            for i, port in enumerate(ports):
                clean = _clean_port_name(port, i)
                if port == name or clean == name:
                    self._rtmidi.open_port(i)
                    self._port_name = port
                    return True

            # 2. 前方一致・部分一致（互換用）
            for i, port in enumerate(ports):
                clean = _clean_port_name(port, i)
                if (
                    port.startswith(name)
                    or name.startswith(port)
                    or clean.startswith(name)
                    or name.startswith(clean)
                ):
                    self._rtmidi.open_port(i)
                    self._port_name = port
                    return True
        except Exception:
            self._rtmidi = None

        return False

    def _send(self, status, data1, data2):
        if self._rtmidi is not None and self._rtmidi.is_port_open():
            try:
                self._rtmidi.send_message([
                    int(status) & 0xFF,
                    max(0, min(127, int(data1))),
                    max(0, min(127, int(data2))),
                ])
            except Exception:
                pass

    def note_on(self, pitch, velocity=100, channel=0):
        ch = int(channel) & 0x0F
        p = max(0, min(127, int(pitch)))
        v = max(0, min(127, int(velocity)))
        self._send(0x90 | ch, p, v)
        self._active.add((p, ch))

    def note_off(self, pitch, channel=0):
        ch = int(channel) & 0x0F
        p = max(0, min(127, int(pitch)))
        self._send(0x80 | ch, p, 0)
        self._active.discard((p, ch))

    def control_change(self, control, value, channel=0):
        ch = int(channel) & 0x0F
        c = max(0, min(127, int(control)))
        v = max(0, min(127, int(value)))
        self._send(0xB0 | ch, c, v)

    def all_notes_off(self):
        for pitch, channel in list(self._active):
            self._send(0x80 | channel, pitch, 0)
        self._active.clear()

        for channel in range(16):
            self._send(0xB0 | channel, 123, 0)  # All Notes Off
            self._send(0xB0 | channel, 120, 0)  # All Sound Off
            self._send(0xB0 | channel, 64, 0)   # Sustain Off

    def close(self):
        try:
            self.all_notes_off()
        except Exception:
            pass

        if self._rtmidi is not None:
            try:
                if self._rtmidi.is_port_open():
                    self._rtmidi.close_port()
            except Exception:
                pass
            self._rtmidi = None

        self._active.clear()
        self._port_name = None


class MidiOutManager:
    """複数のAudioDataインスタンスで1つのMIDI出力デバイスを共有する。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._devices = {}

    def acquire(self, name):
        with self._lock:
            entry = self._devices.get(name)
            if entry is None:
                device = MidiOutDevice()
                if not device.open(name):
                    return None
                self._devices[name] = [device, 1]
                return device
            entry[1] += 1
            return entry[0]

    def release(self, name, device):
        with self._lock:
            entry = self._devices.get(name)
            if entry is None or entry[0] is not device:
                return
            entry[1] -= 1
            if entry[1] <= 0:
                try:
                    entry[0].close()
                except Exception:
                    pass
                del self._devices[name]

    def close_all(self):
        with self._lock:
            devices = list(self._devices.values())
            self._devices.clear()
        for entry in devices:
            try:
                entry[0].close()
            except Exception:
                pass


shared_manager = MidiOutManager()
