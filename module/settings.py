import os

from PySide6.QtCore import QSettings

try:
    import winreg
except ImportError:
    winreg = None

ORGANIZATION = "WaveNoteMIDIEditor"
APPLICATION = "WaveNoteMIDIEditor"

LEGACY_REG_KEY = r"Software\WaveNoteMIDIEditor"

_settings = None


def _qsettings():
    global _settings

    if _settings is None:
        _settings = QSettings(
            ORGANIZATION,
            APPLICATION
        )

        _migrate_legacy_registry(_settings)

    return _settings


def _migrate_legacy_registry(qs):
    """旧レジストリ(HKCU\\Software\\WaveNoteMIDIEditor)の値を
    QSettings へ一度だけコピーして移行する。"""
    if qs.value("migrated_from_registry", "0") == "1":
        return

    if winreg is None:
        return

    values = []

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            LEGACY_REG_KEY,
            0,
            winreg.KEY_READ
        )

        index = 0

        while True:
            try:
                name, val, _ = winreg.EnumValue(
                    key,
                    index
                )
            except OSError:
                break

            values.append((name, val))

            index += 1

        winreg.CloseKey(key)
    except OSError:
        values = []

    if values:
        for name, val in values:
            if not qs.contains(name):
                qs.setValue(
                    name,
                    str(val)
                )

    qs.setValue(
        "migrated_from_registry",
        "1"
    )

    try:
        winreg.DeleteKey(
            winreg.HKEY_CURRENT_USER,
            LEGACY_REG_KEY
        )
    except OSError:
        pass


def load_value(name, default=None):
    val = _qsettings().value(name)

    if val is None:
        return default

    if isinstance(val, bytes):
        val = val.decode(
            "utf-8",
            errors="replace"
        )

    return val if isinstance(val, str) else str(val)


def save_value(name, value):
    _qsettings().setValue(
        name,
        str(value)
    )


def delete_value(name):
    _qsettings().remove(name)


def load_last_dir(default="", key="last_open_dir"):
    """最後にファイルを開いた/保存したフォルダを返す。
    無効な場合は default を返す。"""
    path = load_value(
        key,
        ""
    )

    if path and os.path.isdir(str(path)):
        return str(path)

    return default


def save_last_dir_from_path(path, key="last_open_dir"):
    directory = os.path.dirname(
        os.path.abspath(str(path))
    )

    if directory:
        save_value(
            key,
            directory
        )
