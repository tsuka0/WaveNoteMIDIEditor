import winreg

REG_KEY = r"Software\WaveNoteMIDIEditor"


def load_value(name, default=None):
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            REG_KEY,
            0,
            winreg.KEY_READ
        )

        value, _ = winreg.QueryValueEx(
            key,
            name
        )

        winreg.CloseKey(key)

        return value
    except OSError:
        return default


def save_value(name, value):
    key = winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        REG_KEY,
        0,
        winreg.KEY_SET_VALUE
    )

    winreg.SetValueEx(
        key,
        name,
        0,
        winreg.REG_SZ,
        str(value)
    )

    winreg.CloseKey(key)
