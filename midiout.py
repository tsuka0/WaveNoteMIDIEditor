import ctypes
import time
from ctypes import wintypes

MMSYSERR_NOERROR = 0
CALLBACK_NULL = 0

winmm = ctypes.WinDLL("winmm")

winmm.midiOutGetNumDevs.restype = ctypes.c_uint

winmm.midiOutOpen.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
]
winmm.midiOutOpen.restype = ctypes.c_uint

winmm.midiOutShortMsg.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
]
winmm.midiOutShortMsg.restype = ctypes.c_uint

winmm.midiOutReset.argtypes = [
    ctypes.c_void_p,
]
winmm.midiOutReset.restype = ctypes.c_uint

winmm.midiOutClose.argtypes = [
    ctypes.c_void_p,
]
winmm.midiOutClose.restype = ctypes.c_uint


class _MidiOutCaps(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", ctypes.c_wchar * 32),
        ("wTechnology", wintypes.WORD),
        ("wVoices", wintypes.WORD),
        ("wNotes", wintypes.WORD),
        ("wChannelMask", wintypes.WORD),
        ("dwSupport", wintypes.DWORD),
    ]


def list_ports():
    ports = []

    count = int(winmm.midiOutGetNumDevs())

    for i in range(count):
        caps = _MidiOutCaps()

        if (
            winmm.midiOutGetDevCapsW(
                i,
                ctypes.byref(caps),
                ctypes.sizeof(_MidiOutCaps)
            ) == MMSYSERR_NOERROR
        ):
            ports.append(caps.szPname)

    return ports


def _find_device_id(name):
    count = int(winmm.midiOutGetNumDevs())

    for i in range(count):
        caps = _MidiOutCaps()

        if (
            winmm.midiOutGetDevCapsW(
                i,
                ctypes.byref(caps),
                ctypes.sizeof(_MidiOutCaps)
            ) == MMSYSERR_NOERROR
        ):
            if caps.szPname == name:
                return i

    return None


class MidiOutDevice:
    def __init__(self):
        self._handle = None
        self._active = set()

    def open(self, name):
        if self._handle is not None:
            return True

        device_id = _find_device_id(name)

        if device_id is None:
            return False

        handle = ctypes.c_void_p()

        if (
            winmm.midiOutOpen(
                ctypes.byref(handle),
                device_id,
                None,
                None,
                CALLBACK_NULL
            ) != MMSYSERR_NOERROR
        ):
            return False

        self._handle = handle.value

        return True

    def _send(self, status, data1, data2):
        if self._handle is None:
            return

        message = (
            status |
            ((data1 & 0x7F) << 8) |
            ((data2 & 0x7F) << 16)
        )

        winmm.midiOutShortMsg(
            self._handle,
            message
        )

    def note_on(self, pitch, velocity=100, channel=0):
        if self._handle is None:
            return

        self._send(
            0x90 | (channel & 0x0F),
            max(0, min(127, pitch)),
            max(0, min(127, velocity))
        )

        self._active.add((int(pitch), channel & 0x0F))

    def note_off(self, pitch, channel=0):
        if self._handle is None:
            return

        self._send(
            0x80 | (channel & 0x0F),
            max(0, min(127, pitch)),
            0
        )

        self._active.discard((int(pitch), channel & 0x0F))

    def control_change(self, control, value, channel=0):
        if self._handle is None:
            return

        self._send(
            0xB0 | (channel & 0x0F),
            max(0, min(127, control)),
            max(0, min(127, value))
        )

    def all_notes_off(self):
        if self._handle is None:
            return

        for pitch, channel in list(self._active):
            self._send(
                0x80 | channel,
                max(0, min(127, pitch)),
                0
            )

        self._active.clear()

        for channel in range(16):
            self._send(0xB0 | channel, 123, 0)
            self._send(0xB0 | channel, 120, 0)
            self._send(0xB0 | channel, 64, 0)

    def close(self):
        if self._handle is None:
            return

        self.all_notes_off()

        winmm.midiOutReset(self._handle)

        time.sleep(0.05)

        winmm.midiOutClose(self._handle)

        self._handle = None
        self._active.clear()
