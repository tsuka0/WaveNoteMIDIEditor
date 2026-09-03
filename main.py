import os
import time
import sys
import ctypes
import gzip
import threading
import json
from pathlib import Path

def get_resource_path(relative_path):
    import sys, os
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

GZIP_MAGIC = b"\x1f\x8b"

def read_project_json(path):
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)

    return json.loads(raw.decode("utf-8"))

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QLabel,
    QCheckBox,
    QProgressDialog,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QComboBox,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout, 
    QListWidget,
    QListWidgetItem,
    QKeySequenceEdit,
    QSizePolicy,
    QWidget
)
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtCore import QTimer, Qt, QEvent, QObject
from module.audio import AudioData
from module.spectrum import SpectrumData
from module.midi import MidiData, Note, PedalEvent
from module.piano_roll import PianoRoll
from module import midiout
from module.midiout import list_ports
from module.settings import load_value, save_value, delete_value, load_last_dir, save_last_dir_from_path
from module.i18n import tr, get_language, set_language, LANGUAGES
from module.discord_rpc import DiscordRPC
from module.features import ENABLE_LYRICS, ENABLE_SVP

DISCORD_CLIENT_ID = "1539710543751942214"

DEFAULT_SHORTCUTS = {
    "action_new_project": "Ctrl+N",
    "action_open_project": "Ctrl+O",
    "action_save_project": "Ctrl+S",
    "action_save_midi": "Shift+S",
    "action_undo": "Ctrl+Z",
    "action_redo": "Ctrl+Y",
    "action_play": "Space",
    "action_split": "S",
    "action_select_all": "Ctrl+A",
    "action_copy": "Ctrl+C",
    "action_paste": "Ctrl+V"
}

if ENABLE_LYRICS:
    DEFAULT_SHORTCUTS["action_lyric_mode"] = "L"

def get_shortcut(key):
    val = load_value(f"shortcut_{key}")
    if val:
        return str(val)
    return DEFAULT_SHORTCUTS.get(key, "")

def apply_toggle_style(btn, checked_color):
    """トグルボタンのスタイル。ONのとき分かりやすいように指定色で塗りつぶし、
    OFFのときはOSのテーマ(ダーク/ライト)に従うようにスタイルシートをリセットする。"""
    def update_style(checked):
        if checked:
            btn.setStyleSheet(f"background-color: {checked_color}; color: white; font-weight: bold; border: none; padding: 4px 10px; border-radius: 4px;")
        else:
            btn.setStyleSheet("")
    
    btn.toggled.connect(update_style)
    # 遅延適用 (初期化直後は状態が反映されない場合があるため)
    QTimer.singleShot(0, lambda: update_style(btn.isChecked()))


def make_help_badge(tooltip):
    badge = QLabel("?")
    badge.setToolTip(tooltip)
    badge.setAlignment(Qt.AlignCenter)
    badge.setFixedSize(14, 14)
    badge.setStyleSheet(
        "QLabel { color: #999; border: 1px solid #888; border-radius: 7px;"
        " font-size: 9px; font-weight: bold; }"
        "QLabel:hover { color: #fff; border-color: #fff; }"
    )
    return badge


def label_with_help(text, tooltip, tail=""):
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    layout.addWidget(QLabel(text))
    layout.addWidget(make_help_badge(tooltip))
    if tail:
        layout.addWidget(QLabel(tail))
    return container

AUDIO_PRESETS_KEY = "audio_presets"

CHANNEL_MODE_LABELS = [
    tr("ステレオ", "Stereo"),
    tr("L+R (モノラル)", "L+R (Mono)"),
    tr("L-R (ボーカルキャンセル)", "L-R (Vocal Cancel)"),
    tr("Lのみ", "L only"),
    tr("Rのみ", "R only")
]

def default_audio_presets():
    return [
        {"name": "フラット", "channel": 0, "low": 100, "mid": 100, "high": 100, "builtin": True},
        {"name": "メロディを聞きやすく", "channel": 1, "low": 60, "mid": 145, "high": 115, "builtin": True},
        {"name": "ベースを聞きやすく", "channel": 1, "low": 170, "mid": 80, "high": 55, "builtin": True},
        {"name": "リズムを聞きやすく", "channel": 1, "low": 155, "mid": 65, "high": 150, "builtin": True},
        {"name": "ボーカルを消す", "channel": 2, "low": 100, "mid": 100, "high": 100, "builtin": True}
    ]


BUILTIN_PRESET_NAME_EN = {
    "フラット": "Flat",
    "メロディを聞きやすく": "Bring out melody",
    "ベースを聞きやすく": "Bring out bass",
    "リズムを聞きやすく": "Bring out rhythm",
    "ボーカルを消す": "Remove vocals"
}


def normalize_audio_preset(data):
    preset = {
        "name": str(data.get("name", tr("プリセット", "Preset"))),
        "channel": int(data.get("channel", 0)),
        "low": int(data.get("low", 100)),
        "mid": int(data.get("mid", 100)),
        "high": int(data.get("high", 100)),
        "builtin": bool(data.get("builtin", False))
    }
    preset["name"] = preset["name"] or tr("プリセット", "Preset")
    preset["channel"] = max(0, min(len(CHANNEL_MODE_LABELS) - 1, preset["channel"]))
    for key in ("low", "mid", "high"):
        preset[key] = max(0, min(200, preset[key]))
    return preset


def load_audio_presets():
    raw = load_value(AUDIO_PRESETS_KEY)

    if raw:
        try:
            data = json.loads(str(raw))
            if isinstance(data, list):
                presets = []

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    preset = normalize_audio_preset(item)

                    if (
                        "builtin" not in item and
                        preset["name"] in BUILTIN_PRESET_NAME_EN
                    ):
                        preset["builtin"] = True

                    presets.append(preset)

                return presets
        except (ValueError, TypeError):
            pass

    presets = [dict(p) for p in default_audio_presets()]
    save_value(
        AUDIO_PRESETS_KEY,
        json.dumps(presets, ensure_ascii=False)
    )
    return presets


def preset_display_name(preset):
    name = str(preset.get("name", ""))

    if preset.get("builtin") and get_language() == "en":
        return BUILTIN_PRESET_NAME_EN.get(name, name)

    return name


class ShortcutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("カスタムショートカット", "Custom Shortcuts"))
        
        self.edits = {}
        form = QFormLayout()
        
        labels = {
            "action_new_project": tr("プロジェクトを新規作成", "New Project"),
            "action_open_project": tr("プロジェクトを開く", "Open Project"),
            "action_save_project": tr("プロジェクト保存", "Save Project"),
            "action_save_midi": tr("MIDI保存", "Save MIDI"),
            "action_undo": tr("元に戻す", "Undo"),
            "action_redo": tr("やり直し", "Redo"),
            "action_play": tr("再生 / 停止", "Play / Stop"),
            "action_split": tr("ノーツを分割", "Split Notes"),
            "action_select_all": tr("すべて選択", "Select All"),
            "action_copy": tr("コピー", "Copy"),
            "action_paste": tr("ペースト", "Paste")
        }

        if ENABLE_LYRICS:
            labels["action_lyric_mode"] = tr("歌詞入力モード", "Lyric Input Mode")
        
        for key, label in labels.items():
            edit = QKeySequenceEdit()
            edit.setKeySequence(QKeySequence(get_shortcut(key)))
            self.edits[key] = edit
            form.addRow(label, edit)
            
        reset_btn = QPushButton(tr("デフォルトにリセット", "Reset to Defaults"))
        reset_btn.clicked.connect(self.reset_to_defaults)
        form.addRow("", reset_btn)
            
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)
        
    def reset_to_defaults(self):
        for key, edit in self.edits.items():
            edit.setKeySequence(QKeySequence(DEFAULT_SHORTCUTS.get(key, "")))
            
    def apply_shortcuts(self):
        for key, edit in self.edits.items():
            val = edit.keySequence().toString()
            if val:
                save_value(f"shortcut_{key}", val)
            else:
                delete_value(f"shortcut_{key}")

class SettingsDialog(QDialog):
    def __init__(self, parent=None, current="internal"):
        super().__init__(parent)

        self.setWindowTitle(tr("設定", "Settings"))

        self._device_combo = QComboBox()

        self._device_combo.addItem(
            tr("内蔵音源", "Internal Synth"),
            "internal"
        )

        selected_index = 0

        for port_name in list_ports():
            self._device_combo.addItem(
                port_name,
                port_name
            )

            if (
                port_name == current and
                current != "internal"
            ):
                selected_index = (
                    self._device_combo.count() - 1
                )

        self._device_combo.setCurrentIndex(
            selected_index
        )

        refresh_button = QPushButton(tr("デバイスを更新", "Refresh Devices"))
        refresh_button.clicked.connect(self._refresh_devices)

        self.backup_cb = QCheckBox(tr("定期バックアップを有効にする", "Enable periodic backup"))
        self.backup_cb.setChecked(load_value("auto_backup_enabled", "0") == "1")

        self.backup_spin = QSpinBox()
        self.backup_spin.setRange(1, 120)
        self.backup_spin.setSuffix(tr(" 分", " min"))
        self.backup_spin.setValue(int(load_value("auto_backup_interval", "5")))

        self.discord_cb = QCheckBox(tr("DiscordのRich Presenceを有効にする", "Enable Discord Rich Presence"))
        self.discord_cb.setChecked(load_value("discord_rpc_enabled", "1") == "1")

        self.shortcut_btn = QPushButton(tr("カスタムショートカットの設定...", "Configure Shortcuts..."))
        self.shortcut_btn.clicked.connect(self.open_shortcuts)

        self.lang_combo = QComboBox()

        for code, label in LANGUAGES:
            self.lang_combo.addItem(label, code)

        self._initial_language = get_language()
        lang_index = self.lang_combo.findData(get_language())

        if lang_index >= 0:
            self.lang_combo.setCurrentIndex(lang_index)

        self.grid_combo = QComboBox()
        self.grid_combo.addItem(tr("4分音符", "Quarter note"), "1.0")
        self.grid_combo.addItem(tr("8分音符", "Eighth note"), "0.5")
        self.grid_combo.addItem(tr("16分音符", "16th note"), "0.25")
        
        current_grid = load_value("grid_fineness", "1.0")
        if current_grid == "auto":
            current_grid = "1.0"
            
        grid_idx = self.grid_combo.findData(current_grid)
        if grid_idx >= 0:
            self.grid_combo.setCurrentIndex(grid_idx)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem(tr("PCに合わせる", "Match System"), "auto")
        self.theme_combo.addItem(tr("白 (Light)", "Light"), "light")
        self.theme_combo.addItem(tr("黒 (Dark)", "Dark"), "dark")
        
        current_theme = load_value("theme", "auto")
        theme_idx = self.theme_combo.findData(current_theme)
        if theme_idx >= 0:
            self.theme_combo.setCurrentIndex(theme_idx)

        form = QFormLayout()
        form.addRow(tr("テーマ", "Theme"), self.theme_combo)
        form.addRow(tr("言語", "Language"), self.lang_combo)
        form.addRow(tr("MIDI出力", "MIDI Output"), self._device_combo)
        form.addRow("", refresh_button)
        form.addRow(tr("バックアップ", "Backup"), self.backup_cb)
        form.addRow(tr("間隔", "Interval"), self.backup_spin)
        form.addRow("Discord", self.discord_cb)
        form.addRow(tr("グリッドの細かさ", "Grid Fineness"), self.grid_combo)
        form.addRow(tr("ショートカット", "Shortcuts"), self.shortcut_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(
            self.accept
        )
        buttons.rejected.connect(
            self.reject
        )

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def open_shortcuts(self):
        dlg = ShortcutDialog(self)
        if dlg.exec() == QDialog.Accepted:
            dlg.apply_shortcuts()
            main = self.parent()
            if main is not None and hasattr(main, "update_shortcuts"):
                main.update_shortcuts()

    def accept(self):
        save_value(
            "language",
            self.lang_combo.currentData() or "ja"
        )
        set_language(self.lang_combo.currentData() or "ja")
        save_value("auto_backup_enabled", "1" if self.backup_cb.isChecked() else "0")
        save_value("auto_backup_interval", str(self.backup_spin.value()))
        save_value("discord_rpc_enabled", "1" if self.discord_cb.isChecked() else "0")
        save_value("grid_fineness", self.grid_combo.currentData())
        save_value("theme", self.theme_combo.currentData())
        super().accept()

    def language_changed(self):
        return (
            (self.lang_combo.currentData() or "ja") !=
            self._initial_language
        )

    def _refresh_devices(self):
        current = self._device_combo.currentData()

        self._device_combo.blockSignals(True)

        self._device_combo.clear()

        self._device_combo.addItem(
            tr("内蔵音源", "Internal Synth"),
            "internal"
        )

        selected_index = 0

        for port_name in list_ports():
            self._device_combo.addItem(
                port_name,
                port_name
            )

            if port_name == current:
                selected_index = (
                    self._device_combo.count() - 1
                )

        self._device_combo.setCurrentIndex(
            selected_index
        )

        self._device_combo.blockSignals(False)

    def output_device(self):
        return self._device_combo.currentData()

class TrackComboFilter(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self.main_window.rename_track()
            return True
        return super().eventFilter(obj, event)

class AudioPresetDialog(QDialog):
    def __init__(self, parent=None, presets=None):
        super().__init__(parent)
        self.setWindowTitle(tr("プリセットの管理", "Manage Presets"))
        self.setMinimumWidth(420)

        self.presets = [dict(p) for p in (presets or [])]
        self._editing_row = -1

        self.list_widget = QListWidget(self)

        self.name_edit = QLineEdit(self)
        self.channel_combo = QComboBox(self)
        self.channel_combo.addItems(CHANNEL_MODE_LABELS)

        self.eq_spins = {}
        form = QFormLayout()
        form.addRow(tr("名前", "Name"), self.name_edit)
        form.addRow(tr("チャンネル", "Channel"), self.channel_combo)

        for key, label in (
            ("low", tr("低域", "Low")),
            ("mid", tr("中域", "Mid")),
            ("high", tr("高域", "High"))
        ):
            spin = QSpinBox(self)
            spin.setRange(0, 200)
            spin.setSuffix(" %")
            self.eq_spins[key] = spin
            form.addRow(label, spin)

        add_btn = QPushButton(tr("新規作成", "New"), self)
        add_btn.clicked.connect(self.add_preset)
        delete_btn = QPushButton(tr("削除", "Delete"), self)
        delete_btn.clicked.connect(self.delete_preset)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            self
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(add_btn)
        left_layout.addWidget(delete_btn)

        right_layout = QVBoxLayout()
        right_layout.addLayout(form)
        right_layout.addStretch()
        right_layout.addWidget(btn_box)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.list_widget.currentRowChanged.connect(
            self.on_row_changed
        )
        self.name_edit.textChanged.connect(
            self.on_name_edited
        )

        for index in range(len(self.presets)):
            self.list_widget.addItem(
                preset_display_name(self.presets[index])
            )

        if self.presets:
            self.list_widget.setCurrentRow(0)
        else:
            self.set_form_enabled(False)

    def set_form_enabled(self, enabled):
        self.name_edit.setEnabled(enabled)
        self.channel_combo.setEnabled(enabled)
        for spin in self.eq_spins.values():
            spin.setEnabled(enabled)

    def commit_form(self):
        row = self._editing_row
        if row < 0 or row >= len(self.presets):
            return
        preset = self.presets[row]
        name = self.name_edit.text().strip()

        if name:
            if preset.get("builtin"):
                if (
                    name != preset["name"] and
                    name != preset_display_name(preset)
                ):
                    preset["builtin"] = False
                    preset["name"] = name
            else:
                preset["name"] = name

        preset["channel"] = self.channel_combo.currentIndex()
        for key, spin in self.eq_spins.items():
            preset[key] = spin.value()

        self.list_widget.item(row).setText(preset_display_name(preset))

    def on_row_changed(self, row):
        self.commit_form()
        self._editing_row = row

        if row < 0 or row >= len(self.presets):
            self.set_form_enabled(False)
            return

        self.set_form_enabled(True)
        preset = self.presets[row]
        self.name_edit.setText(preset_display_name(preset))
        self.channel_combo.setCurrentIndex(preset["channel"])
        for key, spin in self.eq_spins.items():
            spin.setValue(preset[key])

    def add_preset(self):
        name = tr("新規プリセット", "New Preset")
        names = [p["name"] for p in self.presets]

        if name in names:
            index = 2
            while f"{name}{index}" in names:
                index += 1
            name = f"{name}{index}"

        base = dict(self.presets[-1]) if self.presets else {
            "name": name,
            "channel": 0,
            "low": 100,
            "mid": 100,
            "high": 100
        }
        base["name"] = name
        base["builtin"] = False

        self.presets.append(base)
        self.list_widget.addItem(name)
        self.list_widget.setCurrentRow(len(self.presets) - 1)
        self.name_edit.selectAll()
        self.name_edit.setFocus()

    def delete_preset(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.presets):
            return

        del self.presets[row]
        self.list_widget.takeItem(row)

        if not self.presets:
            self._editing_row = -1
            self.set_form_enabled(False)

    def on_name_edited(self, text):
        row = self._editing_row
        if row < 0 or row >= len(self.presets):
            return

        item = self.list_widget.item(row)
        if item is not None:
            item.setText(text)

    def accept(self):
        self.commit_form()
        super().accept()

class MainWindow(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()

        icon_path = get_resource_path(os.path.join("Assets", "icon.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("WaveNoteMIDIEditor")
        self.setMinimumSize(1280, 720)
        self.showMaximized()
        self.setAcceptDrops(True)
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        self.menuBar().setContextMenuPolicy(Qt.PreventContextMenu)

        self.audio = AudioData()
        self._global_audio = self.audio
        self.track_audio = {}
        self.spectrum = SpectrumData()
        self.midi = MidiData()

        self.audio_presets = load_audio_presets()

        self.audio.set_output_device(
            load_value(
                "midi_out_device",
                "internal"
            )
        )

        self.editor = PianoRoll(
            self.audio,
            self.spectrum,
            self.midi
        )
        self.editor.track_switch_requested.connect(
            self.switch_track_by_delta
        )
        self.editor.tap_tempo_applied.connect(
            self.apply_tapped_tempo
        )

        saved_return = load_value(
            "return_to_start_on_stop",
            "1"
        )
        self.editor.return_to_start_on_stop = (
            str(saved_return).lower() in ("1", "true", "yes", "on")
        )

        saved_mute_midi = load_value(
            "mute_midi",
            "0"
        )
        self.audio.midi_muted = (
            str(saved_mute_midi).lower() in ("1", "true", "yes", "on")
        )

        saved_play_all = load_value(
            "play_all_tracks",
            "0"
        )
        self.midi.play_all_tracks = (
            str(saved_play_all).lower() in ("1", "true", "yes", "on")
        )

        saved_threshold = load_value(
            "spectrum_threshold",
            None
        )
        if saved_threshold is not None:
            try:
                thresh_val = int(float(saved_threshold))
                thresh_val = max(0, min(40, thresh_val))
                self.editor.spectrum_threshold = (
                    thresh_val / 100.0
                )
            except (ValueError, TypeError):
                pass

        saved_sensitivity = load_value(
            "spectrum_sensitivity",
            None
        )
        if saved_sensitivity is not None:
            try:
                sens_val = float(saved_sensitivity)
                sens_val = max(10.0, min(100.0, sens_val))
                self.editor.spectrum_db_range = sens_val
            except (ValueError, TypeError):
                pass

        self.setCentralWidget(self.editor)

        self.editor.note_length = 0.25
        self.editor.placement_beats = 0.25

        self.editor.marker_edited.connect(
            self.after_edit
        )

        self.actions = {}
        self.create_menu()
        self.create_toolbar()
        self.create_status_bar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(
            self.update_editor
        )
        self.timer.setTimerType(
            Qt.TimerType.PreciseTimer
        )
        self.timer.start(16)

        self._analysis_ready = False
        self._analysis_error = None
        self._analysis_token = 0
        self._pending_audio_duration = None
        self._pending_tempo_analysis = None
        self._pending_tap_onsets = None
        self._tempo_analyzed = False
        self._project_path = None
        self._last_midi_path = None
        self._last_midi_is_svp = False
        self._saved_project_state = None

        self._mark_project_saved()

        self.auto_backup_timer = QTimer(self)
        self.auto_backup_timer.timeout.connect(self.auto_backup)
        self.update_auto_backup_timer()

        self.discord_rpc = DiscordRPC(DISCORD_CLIENT_ID)
        self.discord_rpc_timer = QTimer(self)
        self.discord_rpc_timer.timeout.connect(self.update_discord_rpc)
        self.discord_rpc_timer.start(5000)
        self._discord_start_time = time.time()
        self.update_discord_rpc()

        if initial_file:
            self.open_file_by_path(initial_file)

        self.update_shortcuts()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def update_discord_rpc(self):
        rpc = getattr(self, "discord_rpc", None)
        if rpc is None:
            return

        enabled = load_value("discord_rpc_enabled", "1") == "1"
        if not enabled:
            if rpc.connected:
                rpc.close()
            return

        rpc.update(
            details="WaveNoteMIDIEditor",
            start_time=self._discord_start_time
        )

    def update_auto_backup_timer(self):
        enabled = load_value("auto_backup_enabled", "0") == "1"
        interval = int(load_value("auto_backup_interval", "5"))
        if enabled:
            self.auto_backup_timer.start(interval * 60 * 1000)
        else:
            self.auto_backup_timer.stop()

    def auto_backup(self):
        if self._project_path:
            self.save_project()
            current_time = time.strftime("%Y/%m/%d %H:%M:%S")
            self.status_backup_label.setText(
                tr(
                    f"バックアップ保存: {current_time}",
                    f"Backup saved: {current_time}"
                )
            )

    def update_shortcuts(self):
        if not hasattr(self, "actions"):
            return
        for key, action in self.actions.items():
            shortcut_str = get_shortcut(key)
            if shortcut_str:
                action.setShortcut(QKeySequence(shortcut_str))
            else:
                action.setShortcut(QKeySequence())

    def create_status_bar(self):
        status = self.statusBar()
        status.setSizeGripEnabled(False)

        self.status_position_label = QLabel(" 00:00.0")
        self.status_position_label.setMinimumWidth(80)
        self.status_position_label.setToolTip(tr("現在の再生位置", "Current playback position"))

        self.status_notes_label = QLabel(tr("ノーツ 0", "Notes 0"))
        self.status_notes_label.setMinimumWidth(90)
        self.status_notes_label.setToolTip(
            tr(
                "ノーツ数(トラック選択中はそのトラックの数)",
                "Note count (per-track count when a track is selected)"
            )
        )

        self.status_bpm_label = QLabel("BPM ---")
        self.status_bpm_label.setMinimumWidth(90)
        self.status_bpm_label.setToolTip(tr("再生位置のテンポ", "Tempo at playback position"))

        status.addWidget(self.status_position_label)
        status.addWidget(self.status_notes_label)
        status.addWidget(self.status_bpm_label)

        self.status_backup_label = QLabel("")
        self.status_backup_label.setToolTip(tr("自動バックアップの記録", "Auto backup history"))

        status.addPermanentWidget(self.status_backup_label)

    def format_status_time(self, seconds):
        seconds = max(
            0.0,
            float(seconds)
        )

        minutes = int(seconds // 60)

        rest = seconds - minutes * 60

        return f"{minutes:02d}:{rest:04.1f}"

    def update_status_labels(self):
        position_text = " " + self.format_status_time(
            self.editor.play_position
        )
        self.status_position_label.setText(position_text)

        if (
            self.midi.filter_track is not None and
            0 <= self.midi.filter_track < len(self.midi.tracks)
        ):
            notes_count = len(
                self.midi.tracks[self.midi.filter_track].notes
            )
        else:
            notes_count = sum(
                len(track.notes)
                for track in self.midi.tracks
            )

        notes_text = tr(f"ノーツ {notes_count}", f"Notes {notes_count}")

        if self.status_notes_label.text() != notes_text:
            self.status_notes_label.setText(notes_text)

        bpm = self.midi.tempo_at(
            self.editor.play_position
        )

        bpm_text = f"BPM {bpm:.0f}"

        if self.status_bpm_label.text() != bpm_text:
            self.status_bpm_label.setText(bpm_text)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                if self.project_is_modified():
                    answer = QMessageBox.question(
                        self,
                        tr("プロジェクトを保存", "Save Project"),
                        tr(
                            "現在のプロジェクトに未保存の変更があります。保存しますか？",
                            "The current project has unsaved changes. Save them?"
                        ),
                        QMessageBox.Yes |
                        QMessageBox.No |
                        QMessageBox.Cancel,
                        QMessageBox.Yes
                    )
                    if answer == QMessageBox.Cancel:
                        return
                    if answer == QMessageBox.Yes:
                        if not self.save_project():
                            return
                self.open_file_by_path(path)

    def open_file_by_path(self, path):
        if not path:
            return
        path = str(path).strip().strip('"').strip("'")
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(f"ファイルが見つかりません:\n{path}", f"File not found:\n{path}")
            )
            return

        ext = Path(path).suffix.lower()
        if ext == ".wnp":
            self.load_project(path)
        elif ext in (".mid", ".midi") or (
            ENABLE_SVP and ext == ".svp"
        ):
            self.load_midi_file(path)
        elif ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            self.load_audio_file(path)
        else:
            try:
                data = read_project_json(path)
                if isinstance(data, dict) and ("midi_tracks" in data or "audio_file" in data):
                    self.load_project(path)
                    return
            except Exception:
                pass
            QMessageBox.warning(
                self,
                tr("未対応の形式", "Unsupported Format"),
                tr(
                    f"サポートされていないファイル形式です:\n{path}",
                    f"Unsupported file format:\n{path}"
                )
            )

    def closeEvent(self, event):
        if self.project_is_modified():
            answer = QMessageBox.question(
                self,
                tr("プロジェクトを保存", "Save Project"),
                tr(
                    "プロジェクトに未保存の変更があります。保存しますか？",
                    "The project has unsaved changes. Save them?"
                ),
                QMessageBox.Yes |
                QMessageBox.No |
                QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if answer == QMessageBox.Cancel:
                event.ignore()
                return

            if answer == QMessageBox.Yes:
                if not self.save_project():
                    event.ignore()
                    return

        self.discord_rpc_timer.stop()
        self.discord_rpc.close()
        self.audio.close()
        self._global_audio.close()
        for ta in self.track_audio.values():
            ta.close()
        midiout.shared_manager.close_all()
        super().closeEvent(event)

    def create_menu(self):
        file_menu = self.menuBar().addMenu(
            tr("ファイル", "File")
        )

        new_project_action = QAction(tr("プロジェクトを新規作成", "New Project"), self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)

        load_project_action = QAction(tr("プロジェクトを開く", "Open Project"), self)
        load_project_action.setShortcut(QKeySequence.StandardKey.Open)
        load_project_action.triggered.connect(self.open_project)

        save_project_action = QAction(tr("プロジェクトを保存", "Save Project"), self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)

        if ENABLE_SVP:
            open_midi_label = tr("MIDI / SVPを開く", "Open MIDI / SVP")
        else:
            open_midi_label = tr("MIDIを開く", "Open MIDI")

        open_midi_action = QAction(open_midi_label, self)
        open_midi_action.triggered.connect(self.open_midi)

        open_audio_action = QAction(tr("オーディオを開く", "Open Audio"), self)
        open_audio_action.triggered.connect(self.open_audio)

        save_action = QAction(tr("MIDI上書き保存", "Overwrite-save MIDI"), self)
        save_action.triggered.connect(self.save_midi)
        
        save_as_action = QAction(tr("MIDI名前を付けて保存", "Save MIDI As..."), self)
        save_as_action.triggered.connect(self.save_midi_as)

        exit_action = QAction(tr("終了", "Exit"), self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(new_project_action)
        file_menu.addAction(load_project_action)
        file_menu.addAction(save_project_action)
        file_menu.addAction(open_midi_action)
        file_menu.addAction(open_audio_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu(
            tr("編集", "Edit")
        )

        undo_action = QAction(
            tr("元に戻す", "Undo"),
            self
        )
        undo_action.setShortcut(
            QKeySequence.StandardKey.Undo
        )
        undo_action.triggered.connect(
            self.undo
        )
        edit_menu.addAction(
            undo_action
        )

        redo_action = QAction(
            tr("やり直し", "Redo"),
            self
        )
        redo_action.setShortcut(
            QKeySequence.StandardKey.Redo
        )
        redo_action.triggered.connect(
            self.redo
        )
        edit_menu.addAction(
            redo_action
        )

        copy_action = QAction(
            tr("コピー", "Copy"),
            self
        )
        copy_action.setShortcut(
            QKeySequence.StandardKey.Copy
        )
        copy_action.triggered.connect(
            self.editor.copy_selected
        )
        edit_menu.addAction(copy_action)

        paste_action = QAction(
            tr("ペースト", "Paste"),
            self
        )
        paste_action.setShortcut(
            QKeySequence.StandardKey.Paste
        )
        paste_action.triggered.connect(
            self.editor.paste_notes
        )
        edit_menu.addAction(paste_action)

        playback_menu = self.menuBar().addMenu(
            tr("再生", "Playback")
        )

        play_action = QAction(
            tr("再生 / 停止", "Play / Stop"),
            self
        )
        play_action.setShortcut(
            QKeySequence(
                Qt.Key_Space
            )
        )
        play_action.triggered.connect(
            self.editor.toggle_play
        )
        playback_menu.addAction(
            play_action
        )

        split_action = QAction(
            tr("分割", "Split"),
            self
        )
        split_action.triggered.connect(
            self.editor.split_notes_at_play_position
        )
        edit_menu.addAction(split_action)

        self.actions["action_new_project"] = new_project_action
        self.actions["action_open_project"] = load_project_action
        self.actions["action_save_project"] = save_project_action
        self.actions["action_save_midi"] = save_action
        self.actions["action_undo"] = undo_action
        self.actions["action_redo"] = redo_action
        self.actions["action_copy"] = copy_action
        self.actions["action_paste"] = paste_action
        self.actions["action_play"] = play_action
        self.actions["action_split"] = split_action

        select_all_action = QAction(
            tr("すべて選択", "Select All"),
            self
        )
        select_all_action.triggered.connect(
            self.editor.select_all
        )
        edit_menu.addAction(select_all_action)
        self.actions["action_select_all"] = select_all_action

        if ENABLE_LYRICS:
            lyric_mode_action = QAction(
                tr("歌詞入力モード", "Lyric Input Mode"),
                self
            )
            lyric_mode_action.setCheckable(True)
            lyric_mode_action.setToolTip(
                tr(
                    "ノーツをクリックして歌詞を入力します。"
                    "右ドラッグやCtrl+Aで選択したノーツには時系列順に連続入力できます(L)",
                    "Click notes to enter lyrics. "
                    "Notes selected via right-drag or Ctrl+A can be filled in order (L)"
                )
            )
            lyric_mode_action.triggered.connect(
                self.toggle_lyric_mode
            )
            edit_menu.addAction(lyric_mode_action)
            self.actions["action_lyric_mode"] = lyric_mode_action

        stop_action = QAction(
            tr("停止", "Stop"),
            self
        )
        stop_action.triggered.connect(
            self.editor.stop
        )
        playback_menu.addAction(
            stop_action
        )

        settings_menu = self.menuBar().addMenu(
            tr("設定", "Settings")
        )

        settings_action = QAction(
            tr("設定...", "Settings..."),
            self
        )
        settings_action.triggered.connect(
            self.open_settings
        )
        settings_menu.addAction(
            settings_action
        )

    def create_toolbar(self):
        toolbar = self.addToolBar(
            "MIDI"
        )

        toolbar.setMovable(False)
        toolbar.setContextMenuPolicy(Qt.PreventContextMenu)

        toolbar.addWidget(
            label_with_help(
                "  " + tr("トラック", "Track"),
                tr(
                    "表示・編集するトラックを選択します。\n"
                    "「すべてのトラック」を選ぶと全トラックのノーツを表示します。\n"
                    "リスト上で右クリックするとトラック名を変更できます。",
                    "Selects the track to view and edit.\n"
                    "\"All Tracks\" shows notes from every track.\n"
                    "Right-click an entry to rename the track."
                ),
                tail=" "
            )
        )

        self.track_combo = QComboBox()
        self.track_filter = TrackComboFilter(self)
        self.track_combo.installEventFilter(self.track_filter)

        self.refresh_track_combo()

        self.track_combo.currentIndexChanged.connect(
            self.change_track
        )

        toolbar.addWidget(
            self.track_combo
        )

        add_track_button = QPushButton(
            "＋"
        )
        add_track_button.setFixedSize(
            28,
            28
        )
        add_track_button.setToolTip(
            tr("トラックを追加", "Add Track")
        )
        add_track_button.clicked.connect(
            self.add_track
        )

        toolbar.addWidget(
            add_track_button
        )
        
        spacer = QLabel("   ")
        toolbar.addWidget(spacer)
        toolbar.addSeparator()
        spacer2 = QLabel("   ")
        toolbar.addWidget(spacer2)

        insert_tempo_button = QPushButton(
            tr("テンポ追加", "Add Tempo")
        )
        insert_tempo_button.setToolTip(
            tr("再生位置にテンポを挿入", "Insert a tempo at the play position")
        )
        insert_tempo_button.clicked.connect(
            self.insert_tempo
        )
        toolbar.addWidget(
            insert_tempo_button
        )

        insert_timesig_button = QPushButton(
            tr("拍子追加", "Add Time Signature")
        )
        insert_timesig_button.setToolTip(
            tr("再生位置に拍子を挿入", "Insert a time signature at the play position")
        )
        insert_timesig_button.clicked.connect(
            self.insert_timesig
        )
        toolbar.addWidget(
            insert_timesig_button
        )
        toolbar.addWidget(
            label_with_help(
                "  " + tr("ノート長", "Note Length"),
                tr(
                    "配置や分割をするノーツの長さです。\n"
                    "4分音符(1拍)が基準で、3連音符や付点の長さも選べます。",
                    "The length used when placing or splitting notes.\n"
                    "A quarter note (1 beat) is the base; triplets and dotted lengths are also available."
                ),
                tail=" "
            )
        )

        self.length_combo = QComboBox()

        for name, beats in self.editor.note_lengths:
            self.length_combo.addItem(
                name,
                beats
            )

        # 起動時の実際のノート長(初期値は0.25=16分音符)に選択表示を合わせる
        self.length_combo.setCurrentIndex(
            max(
                0,
                self.length_combo.findData(
                    self.editor.note_length
                )
            )
        )

        self.length_combo.currentIndexChanged.connect(
            self.change_note_length
        )

        toolbar.addWidget(
            self.length_combo
        )

        self.return_to_start_checkbox = QCheckBox(
            tr("停止時に開始位置へ戻る", "Return to start position on stop")
        )
        spacer1 = QWidget()
        spacer1.setFixedWidth(8)
        toolbar.addWidget(spacer1)
        self.return_to_start_checkbox.setChecked(
            self.editor.return_to_start_on_stop
        )
        self.return_to_start_checkbox.toggled.connect(
            self.change_return_to_start
        )

        toolbar.addWidget(
            self.return_to_start_checkbox
        )

        self.mute_midi_button = QPushButton(
            tr("MIDIをミュート", "Mute MIDI")
        )
        self.mute_midi_button.setCheckable(True)
        spacer2 = QWidget()
        spacer2.setFixedWidth(8)
        toolbar.addWidget(spacer2)
        apply_toggle_style(self.mute_midi_button, "#c0392b")
        self.mute_midi_button.setToolTip(
            tr("再生時にMIDI音を鳴らさず、波形(オーディオ)のみ再生します", "Play only the waveform (audio) without MIDI sounds during playback")
        )
        self.mute_midi_button.setChecked(
            self.audio.midi_muted
        )
        self.mute_midi_button.toggled.connect(
            self.change_mute_midi
        )

        toolbar.addWidget(
            self.mute_midi_button
        )

        self.play_all_tracks_button = QPushButton(
            tr("MIDI全体を再生", "Play All MIDI Tracks")
        )
        self.play_all_tracks_button.setCheckable(True)
        spacer3 = QWidget()
        spacer3.setFixedWidth(8)
        toolbar.addWidget(spacer3)
        apply_toggle_style(self.play_all_tracks_button, "#2e7d32")
        self.play_all_tracks_button.setToolTip(
            tr("単一トラック選択中でも、全トラックのMIDIを鳴らして再生します", "Play MIDI from all tracks even when a single track is selected")
        )
        self.play_all_tracks_button.setChecked(
            self.midi.play_all_tracks
        )
        self.play_all_tracks_button.toggled.connect(
            self.change_play_from_start
        )

        toolbar.addWidget(
            self.play_all_tracks_button
        )

        # --- Audio DSP Toolbar (New Row) ---
        self.addToolBarBreak()
        audio_toolbar = self.addToolBar("Audio DSP")
        audio_toolbar.setMovable(False)
        audio_toolbar.setContextMenuPolicy(Qt.PreventContextMenu)

        # Audio Presets
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("プリセット", "Preset"),
                tr(
                    "チャンネルとEQの組み合わせにワンタッチで切り替えられます。\n"
                    "「現在の設定をプリセット保存...」で追加、"
                    "「プリセットの管理...」で編集・削除できます。",
                    "Switches between saved combinations of channel mode and EQ.\n"
                    "Use \"Save Current Settings as Preset...\" to add, and\n"
                    "\"Manage Presets...\" to edit or remove them."
                ),
                tail=": "
            )
        )

        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.preset_combo.setMaximumWidth(160)
        self.preset_combo.currentIndexChanged.connect(
            self.change_audio_preset
        )
        audio_toolbar.addWidget(self.preset_combo)
        self.rebuild_preset_combo()

        # Channel Mode
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("解析/音声", "Analysis/Audio"),
                tr(
                    "音声の再生方法と解析の入力を選択します。\n"
                    "L-R (ボーカルキャンセル): 左右の差を取って中央の音(ボーカル等)を小さくします。\n"
                    "選択した内容はスペクトラム解析にも使われます。",
                    "Chooses how the audio is played and analyzed.\n"
                    "L-R (vocal cancel): subtracts the channels to reduce center-panned sounds such as vocals.\n"
                    "The selection also feeds the spectrum analysis."
                ),
                tail=": "
            )
        )
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(CHANNEL_MODE_LABELS)
        self.channel_combo.currentIndexChanged.connect(self.change_channel_mode)
        audio_toolbar.addWidget(self.channel_combo)
        
        # Simple EQ
        audio_toolbar.addWidget(
            label_with_help(
                "  EQ",
                tr(
                    "EQ(イコライザー)とは、音を低域・中域・高域の3つの音域に分けて、\n"
                    "それぞれの大きさを調整する機能です。\n"
                    "例: 低域を上げるとベースが、中域を上げるとメロディが聞き取りやすくなります。",
                    "EQ (equalizer) splits the sound into three bands (low / mid / high)\n"
                    "and adjusts the volume of each.\n"
                    "e.g. raising the low band makes the bass stand out, raising the mid makes the melody clearer."
                ),
                tail=": "
            )
        )
        
        # Low EQ
        self.eq_low_label = QLabel(tr(" 低域", " Low"))
        audio_toolbar.addWidget(self.eq_low_label)
        self.eq_low_slider = QSlider(Qt.Horizontal)
        self.eq_low_slider.setRange(0, 200) # 0.0x to 2.0x
        self.eq_low_slider.setValue(100)
        self.eq_low_slider.setMaximumWidth(80)
        self.eq_low_slider.setMinimumWidth(30)
        self.eq_low_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.eq_low_slider.valueChanged.connect(self.change_eq)
        audio_toolbar.addWidget(self.eq_low_slider)
        
        # Mid EQ
        self.eq_mid_label = QLabel(tr(" 中域", " Mid"))
        audio_toolbar.addWidget(self.eq_mid_label)
        self.eq_mid_slider = QSlider(Qt.Horizontal)
        self.eq_mid_slider.setRange(0, 200)
        self.eq_mid_slider.setValue(100)
        self.eq_mid_slider.setMaximumWidth(80)
        self.eq_mid_slider.setMinimumWidth(30)
        self.eq_mid_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.eq_mid_slider.valueChanged.connect(self.change_eq)
        audio_toolbar.addWidget(self.eq_mid_slider)
        
        # High EQ
        self.eq_high_label = QLabel(tr(" 高域", " High"))
        audio_toolbar.addWidget(self.eq_high_label)
        self.eq_high_slider = QSlider(Qt.Horizontal)
        self.eq_high_slider.setRange(0, 200)
        self.eq_high_slider.setValue(100)
        self.eq_high_slider.setMaximumWidth(80)
        self.eq_high_slider.setMinimumWidth(30)
        self.eq_high_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.eq_high_slider.valueChanged.connect(self.change_eq)
        audio_toolbar.addWidget(self.eq_high_slider)

        # EQ Reset Button
        self.eq_reset_btn = QPushButton(tr("リセット", "Reset"))
        self.eq_reset_btn.setToolTip(tr("EQをフラット(初期値)に戻します", "Reset EQ to flat (default)"))
        self.eq_reset_btn.clicked.connect(self.reset_eq)
        audio_toolbar.addWidget(self.eq_reset_btn)

        # Audio Offset
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("音声オフセット", "Audio Offset"),
                tr(
                    "MIDIと音声の時刻のずれ(音ズレ)を秒単位で調整します。\n"
                    "波形とノーツの位置が合わないときに調整してください。",
                    "Adjusts the timing gap between the audio and MIDI in seconds.\n"
                    "Useful when the waveform does not line up with the notes."
                ),
                tail=" "
            )
        )

        self.offset_box = QDoubleSpinBox()
        self.offset_box.setRange(-60.0, 60.0)
        self.offset_box.setDecimals(3)
        self.offset_box.setSingleStep(0.005)
        self.offset_box.setSuffix(" s")
        self.offset_box.valueChanged.connect(self.change_offset)
        audio_toolbar.addWidget(self.offset_box)

        # Audio Mute
        self.mute_audio_button = QPushButton(tr("音声ミュート", "Mute Audio"))
        self.mute_audio_button.setCheckable(True)
        spacer4 = QWidget()
        spacer4.setFixedWidth(8)
        audio_toolbar.addWidget(spacer4)
        apply_toggle_style(self.mute_audio_button, "#c0392b")
        self.mute_audio_button.setToolTip(
            tr("音声ファイルの再生をミュートします(MIDI音源は鳴り続けます)", "Mutes audio file playback (MIDI keeps playing)")
        )
        self.mute_audio_button.setChecked(
            str(load_value("mute_audio", "0")).lower() in ("1", "true", "yes", "on")
        )
        self.audio.audio_muted = self.mute_audio_button.isChecked()
        self.mute_audio_button.toggled.connect(self.change_mute_audio)
        audio_toolbar.addWidget(self.mute_audio_button)

        # Spectrum Threshold
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("閾値", "Threshold"),
                tr(
                    "スペクトラムを表示する音の強さの下限です。\n"
                    "大きくするほど弱い音が非表示になり、表示がすっきりします。",
                    "The level below which weak spectrum parts are hidden.\n"
                    "Higher values hide more quiet content for a cleaner display."
                ),
                tail=" "
            )
        )

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 40)
        self.threshold_slider.setValue(int(self.editor.spectrum_threshold * 100))
        self.threshold_slider.setMaximumWidth(110)
        self.threshold_slider.setMinimumWidth(50)
        self.threshold_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.threshold_slider.setToolTip(tr("スペクトラムの弱い部分を非表示にする閾値", "Threshold that hides weak spectrum parts"))
        self.threshold_slider.valueChanged.connect(self.change_threshold)
        audio_toolbar.addWidget(self.threshold_slider)

        # Spectrum Sensitivity
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("感度", "Sensitivity"),
                tr(
                    "スペクトラムの見え方の範囲です。\n"
                    "低いほど強い音だけが鮮明に表示され、高いほど小さい音まで広く表示されます。",
                    "Controls the display range of the spectrum.\n"
                    "Lower shows only strong sounds sharply; higher also reveals quiet ones."
                ),
                tail=" "
            )
        )

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)
        self.sensitivity_slider.setValue(int(self.editor.spectrum_db_range))
        self.sensitivity_slider.setMaximumWidth(110)
        self.sensitivity_slider.setMinimumWidth(50)
        self.sensitivity_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.sensitivity_slider.setToolTip(tr("スペクトラム感度：低いほど鮮明、高いほど広範囲表示", "Spectrum sensitivity: lower is sharper, higher shows a wider range"))
        self.sensitivity_slider.valueChanged.connect(self.change_sensitivity)
        audio_toolbar.addWidget(self.sensitivity_slider)

        # Audio File Volume
        audio_toolbar.addWidget(
            label_with_help(
                "  " + tr("音量", "Volume"),
                tr(
                    "読み込んだ音声ファイルの再生音量です。\n"
                    "MIDI音源の音量には影響しません。",
                    "Playback volume of the loaded audio file.\n"
                    "Does not affect MIDI playback volume."
                ),
                tail=" "
            )
        )

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 50)
        self.volume_slider.setValue(int(self.audio.volume * 100))
        self.volume_slider.setMaximumWidth(110)
        self.volume_slider.setMinimumWidth(50)
        self.volume_slider.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.volume_slider.setToolTip(tr("音声ファイルの音量", "Audio file volume"))
        self.volume_slider.valueChanged.connect(self.change_volume)
        audio_toolbar.addWidget(self.volume_slider)

        self.sync_preset_combo()


    def open_settings(self):
        dialog = SettingsDialog(
            self,
            self.audio.output_device
        )

        if dialog.exec() != QDialog.Accepted:
            return

        device = dialog.output_device()

        self.audio.set_output_device(
            device
        )

        if (
            device != "internal" and
            self.audio.output_device == "internal"
        ):
            QMessageBox.warning(
                self,
                tr("設定", "Settings"),
                tr(
                    "選択したデバイスを開けませんでした。\n内蔵音源に戻しました。",
                    "Could not open the selected device.\nReverted to the internal synth."
                )
            )

        save_value(
            "midi_out_device",
            self.audio.output_device
        )
        self.update_auto_backup_timer()

        if dialog.language_changed():
            QMessageBox.information(
                self,
                tr("言語", "Language"),
                tr(
                    "言語の変更は再起動後に完全に適用されます。",
                    "The language change will be fully applied after a restart."
                )
            )

        self.update_discord_rpc()
        self.editor.grid_fineness = load_value("grid_fineness", "1.0")
        if self.editor.grid_fineness == "auto":
            self.editor.grid_fineness = "1.0"
        self.editor.update()

        theme = load_value("theme", "auto")
        if theme == "light":
            QApplication.instance().styleHints().setColorScheme(Qt.ColorScheme.Light)
        elif theme == "dark":
            QApplication.instance().styleHints().setColorScheme(Qt.ColorScheme.Dark)
        else:
            QApplication.instance().styleHints().setColorScheme(Qt.ColorScheme.Unknown)

    def refresh_track_combo(self):
        self.track_combo.blockSignals(True)

        self.track_combo.clear()

        self.track_combo.addItem(
            tr("すべてのトラック", "All Tracks"),
            None
        )

        for index, track in enumerate(
            self.midi.tracks
        ):
            self.track_combo.addItem(
                track.name,
                index
            )

        if self.midi.filter_track is None:
            self.track_combo.setCurrentIndex(0)
        else:
            self.track_combo.setCurrentIndex(
                min(
                    self.midi.filter_track + 1,
                    self.track_combo.count() - 1
                )
            )

        self.track_combo.blockSignals(False)

    def switch_track_by_delta(self, delta):
        idx = self.track_combo.currentIndex() + delta
        idx = max(0, min(self.track_combo.count() - 1, idx))
        self.track_combo.setCurrentIndex(idx)

    def change_track(self, index):
        self._persist_active_audio_params()
        self.editor.set_track_filter(
            self.track_combo.currentData()
        )
        self._apply_audio_context()

    def _current_audio(self):
        """現在のトラックコンテキストで参照すべき音声を返す。

        単一トラック選択時はそのトラック専用の音声(あれば)を、
        それ以外は全体トラック用の音声を返す。
        """
        index = self.midi.filter_track
        if index is not None and 0 <= index < len(self.midi.tracks):
            track = self.midi.tracks[index]
            if track.audio_file:
                return self._ensure_track_audio(index)
        return self._global_audio

    def _active_track_index(self):
        """現在アクティブな音声が単一トラック専用の場合そのindexを返す。"""
        for index, ta in self.track_audio.items():
            if ta is self.audio:
                return index
        return None

    def _ensure_track_audio(self, track_index):
        ta = self.track_audio.get(track_index)
        if ta is not None:
            return ta

        track = self.midi.tracks[track_index]
        ta = AudioData()

        ta.spectrum = SpectrumData()

        global_audio = self._global_audio

        ta.set_output_device(
            global_audio.output_device
        )
        ta.midi_muted = global_audio.midi_muted
        ta.set_midi(self.midi)

        if track.audio_params:
            self._apply_params_to_audio(
                ta,
                track.audio_params
            )
        else:
            ta.volume = global_audio.volume
            ta.offset = global_audio.offset
            ta.channel_mode = global_audio.channel_mode
            ta.eq_low = global_audio.eq_low
            ta.eq_mid = global_audio.eq_mid
            ta.eq_high = global_audio.eq_high
            ta.audio_muted = global_audio.audio_muted
            ta.a4_freq = global_audio.a4_freq

        self.track_audio[track_index] = ta

        return ta

    def _collect_audio_params(self, audio):
        return {
            "volume": audio.volume,
            "offset": audio.offset,
            "channel": audio.channel_mode,
            "eq_low": audio.eq_low,
            "eq_mid": audio.eq_mid,
            "eq_high": audio.eq_high,
            "muted": audio.audio_muted,
            "a4": audio.a4_freq,
        }

    def _apply_params_to_audio(self, audio, params):
        audio.volume = float(
            params.get("volume", audio.volume)
        )
        audio.offset = float(
            params.get("offset", audio.offset)
        )
        audio.channel_mode = int(
            params.get("channel", audio.channel_mode)
        )
        audio.eq_low = float(
            params.get("eq_low", audio.eq_low)
        )
        audio.eq_mid = float(
            params.get("eq_mid", audio.eq_mid)
        )
        audio.eq_high = float(
            params.get("eq_high", audio.eq_high)
        )
        audio.audio_muted = bool(
            params.get("muted", audio.audio_muted)
        )
        audio.a4_freq = float(
            params.get("a4", audio.a4_freq)
        )

    def _persist_active_audio_params(self):
        for index, ta in self.track_audio.items():
            if ta is self.audio:
                if 0 <= index < len(self.midi.tracks):
                    track = self.midi.tracks[index]

                    track.audio_params = self._collect_audio_params(
                        ta
                    )
                return

    def _sync_audio_ui(self, audio=None):
        if not hasattr(self, "offset_box"):
            return

        audio = audio or self.audio

        self.offset_box.blockSignals(True)
        self.offset_box.setValue(
            audio.offset if audio.offset is not None else 0.0
        )
        self.offset_box.blockSignals(False)

        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(
            int((audio.volume if audio.volume is not None else 0.5) * 100)
        )
        self.volume_slider.blockSignals(False)

        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(
            audio.channel_mode if audio.channel_mode is not None else 0
        )
        self.channel_combo.blockSignals(False)

        self.eq_low_slider.blockSignals(True)
        self.eq_low_slider.setValue(
            int((audio.eq_low if audio.eq_low is not None else 1.0) * 100)
        )
        self.eq_low_slider.blockSignals(False)

        self.eq_mid_slider.blockSignals(True)
        self.eq_mid_slider.setValue(
            int((audio.eq_mid if audio.eq_mid is not None else 1.0) * 100)
        )
        self.eq_mid_slider.blockSignals(False)

        self.eq_high_slider.blockSignals(True)
        self.eq_high_slider.setValue(
            int((audio.eq_high if audio.eq_high is not None else 1.0) * 100)
        )
        self.eq_high_slider.blockSignals(False)

        self.mute_audio_button.blockSignals(True)
        self.mute_audio_button.setChecked(
            bool(audio.audio_muted)
        )
        self.mute_audio_button.blockSignals(False)

        self.sync_preset_combo()

    def _apply_audio_context(self):
        self._persist_active_audio_params()

        target = self._current_audio()

        if target is not self.audio:
            if self.audio.playing:
                self.audio.stop()

            self.audio = target

            self.editor.set_audio(target)

        self._maybe_load_active_audio()

        self._sync_audio_ui()
        self.update_title()

    def _resolve_audio_path(self, audio_file):
        if not audio_file:
            return None

        if os.path.exists(audio_file):
            return os.path.abspath(audio_file)

        proj_dir = None

        if self._project_path:
            proj_dir = Path(self._project_path).resolve().parent

        if proj_dir:
            cand1 = proj_dir / audio_file
            if cand1.exists():
                return str(cand1.resolve())

            cand2 = proj_dir / Path(audio_file).name
            if cand2.exists():
                return str(cand2.resolve())

        return None

    def _load_audio_into(
        self,
        audio,
        resolved_audio_file,
        window_title,
        restore_params=None
    ):
        self._analysis_token += 1
        token = self._analysis_token
        self._analysis_ready = False
        self._analysis_error = None
        self._pending_audio_duration = None
        self._pending_tempo_analysis = None
        self._pending_tap_onsets = None

        if getattr(audio, "spectrum", None) is None:
            audio.spectrum = SpectrumData()

        # 作曲テンポの基準はMIDIファイル(または最初の音声から計測したテンポ)。
        # 最初の音声ロード時のみテンポを解析し、以降の音声では上書きしない。
        analyze_tempo = (
            not self.midi.has_file and
            not self._tempo_analyzed
        )

        is_active = self.audio is audio

        audio.clear()

        if is_active:
            # 読み込み対象が現在表示中の音声の場合のみエディタ表示を
            # リセットする。バックグラウンド対象への読み込みでは
            # 現在表示中の音声(全体トラック等)の解析結果を保持する。
            self.editor.clear_audio()
            self.editor._spectrum_image = None
            self.editor._spectrum_key = None

        def worker():
            try:
                duration = audio.load(resolved_audio_file)
                if token != self._analysis_token:
                    return

                self._pending_audio_duration = duration

                audio.spectrum.analyze(
                    audio.y_mono,
                    audio.sr,
                    self.editor.min_pitch,
                    self.editor.max_pitch,
                    audio.a4_freq
                )

                if analyze_tempo:
                    try:
                        tempo_result = audio.spectrum.analyze_tempo(
                            audio.y_mono,
                            audio.sr
                        )
                    except Exception:
                        tempo_result = None

                    if token == self._analysis_token:
                        self._pending_tempo_analysis = tempo_result

                if token != self._analysis_token:
                    return

                try:
                    from module.taptempo import compute_onset_envelope
                    onset_times, onset_strengths = compute_onset_envelope(
                        audio.y_mono,
                        audio.sr
                    )
                except Exception:
                    onset_times, onset_strengths = [], []

                if token == self._analysis_token:
                    self._pending_tap_onsets = (onset_times, onset_strengths)
                if token == self._analysis_token:
                    self._analysis_ready = True
            except Exception as e:
                if token == self._analysis_token:
                    self._analysis_error = e
                    self._analysis_ready = True

        progress = QProgressDialog(
            tr("オーディオとスペクトラムを解析中...", "Analyzing audio and spectrum..."),
            tr("キャンセル", "Cancel"), 0, 0, self
        )
        progress.setWindowTitle(window_title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()

        thread = threading.Thread(
            target=worker,
            daemon=True
        )
        thread.start()

        while not self._analysis_ready and thread.is_alive():
            QApplication.processEvents()
            time.sleep(0.01)

        progress.close()

        if restore_params:
            self._apply_params_to_audio(
                audio,
                restore_params
            )

        self._refresh_timeline_reference()

    def _refresh_timeline_reference(self):
        ends = [0.0]

        def candidate(a):
            if a.file_path is None:
                return 0.0
            return a.duration() + a.offset

        ends.append(candidate(self._global_audio))

        for ta in self.track_audio.values():
            ends.append(candidate(ta))

        self.editor.set_timeline_reference_end(
            max(ends)
        )

    def _maybe_load_active_audio(self):
        index = self.midi.filter_track

        if index is None or not (0 <= index < len(self.midi.tracks)):
            return

        track = self.midi.tracks[index]

        if not track.audio_file:
            return

        ta = self.track_audio.get(index)

        if ta is None:
            ta = self._ensure_track_audio(index)

        if ta.file_path is not None:
            return

        resolved = self._resolve_audio_path(
            track.audio_file
        )

        if not resolved:
            return

        keep_position = self.editor.play_position

        self._load_audio_into(
            ta,
            resolved,
            tr("オーディオを開く", "Open Audio"),
            restore_params=track.audio_params or None
        )

        self.update_editor()

        self.editor.audio.position = keep_position

        self.editor.set_play_position(keep_position)

        self.editor.follow_play_position(
            force=True
        )

        self.editor.update()

    def add_track(self):
        self.midi.push_undo()

        self.midi.add_track()

        self.refresh_track_combo()

        self.track_combo.setCurrentIndex(
            self.track_combo.count() - 1
        )

        self.change_track(
            self.track_combo.currentIndex()
        )

    def rename_track(self):
        index = self.track_combo.currentData()
        if index is None:
            return

        track = self.midi.tracks[index]
        new_name, ok = QInputDialog.getText(
            self,
            tr("トラック名の変更", "Rename Track"),
            tr("新しいトラック名:", "New track name:"),
            text=track.name
        )

        if ok and new_name:
            self.midi.push_undo()
            track.name = new_name
            self.refresh_track_combo()

    def change_return_to_start(self, checked):
        self.editor.return_to_start_on_stop = checked
        save_value(
            "return_to_start_on_stop",
            "1" if checked else "0"
        )

    def change_mute_midi(self, checked):
        self.audio.midi_muted = checked
        self.audio.apply_midi_mute()
        save_value(
            "mute_midi",
            "1" if checked else "0"
        )

    def change_mute_audio(self, checked):
        self.audio.audio_muted = checked
        self._persist_active_audio_params()
        save_value(
            "mute_audio",
            "1" if checked else "0"
        )

    def change_play_from_start(self, checked):
        self.midi.play_all_tracks = checked
        self.audio.invalidate_midi_cache()
        save_value(
            "play_all_tracks",
            "1" if checked else "0"
        )

    def change_offset(self, value):
        self.audio.offset = value
        self._persist_active_audio_params()
        self._refresh_timeline_reference()
        self.editor.update()

    def change_threshold(self, value):
        self.editor.spectrum_threshold = (
            value /
            100.0
        )

        self.editor.update()

        save_value(
            "spectrum_threshold",
            str(value)
        )

    def change_volume(self, value):
        self.audio.volume = (
            value /
            100.0
        )
        self._persist_active_audio_params()

    def change_channel_mode(self, index):
        self.audio.channel_mode = index
        self.audio.apply_dsp()
        self.update_spectrum()
        self.sync_preset_combo()
        self._persist_active_audio_params()

    def change_eq(self):
        self.audio.eq_low = self.eq_low_slider.value() / 100.0
        self.audio.eq_mid = self.eq_mid_slider.value() / 100.0
        self.audio.eq_high = self.eq_high_slider.value() / 100.0
        self.sync_preset_combo()
        self._persist_active_audio_params()

    def reset_eq(self):
        self.eq_low_slider.setValue(100)
        self.eq_mid_slider.setValue(100)
        self.eq_high_slider.setValue(100)
        self.change_eq()

    def current_audio_state(self):
        return {
            "channel": self.channel_combo.currentIndex(),
            "low": self.eq_low_slider.value(),
            "mid": self.eq_mid_slider.value(),
            "high": self.eq_high_slider.value()
        }

    def match_audio_preset_index(self):
        state = self.current_audio_state()

        for index, preset in enumerate(self.audio_presets):
            if (
                preset["channel"] == state["channel"] and
                preset["low"] == state["low"] and
                preset["mid"] == state["mid"] and
                preset["high"] == state["high"]
            ):
                return index + 1

        return 0

    def rebuild_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(tr("カスタム", "Custom"))

        for preset in self.audio_presets:
            self.preset_combo.addItem(preset_display_name(preset), preset)

        if self.audio_presets:
            self.preset_combo.insertSeparator(self.preset_combo.count())
        self.preset_combo.addItem(tr("現在の設定をプリセット保存...", "Save Current Settings as Preset..."), "__save__")
        self.preset_combo.addItem(tr("プリセットの管理...", "Manage Presets..."), "__manage__")
        self.preset_combo.blockSignals(False)

    def sync_preset_combo(self):
        if not hasattr(self, "channel_combo"):
            return

        index = self.match_audio_preset_index()

        if self.preset_combo.currentIndex() == index:
            return

        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)

    def change_audio_preset(self, index):
        data = self.preset_combo.itemData(index)

        if data is None:
            return

        if data == "__save__":
            self.save_current_as_preset()
            return

        if data == "__manage__":
            self.manage_presets()
            return

        if isinstance(data, dict):
            self.apply_audio_preset(data)

    def apply_audio_preset(self, preset):
        self.channel_combo.setCurrentIndex(preset["channel"])

        self.eq_low_slider.blockSignals(True)
        self.eq_mid_slider.blockSignals(True)
        self.eq_high_slider.blockSignals(True)
        self.eq_low_slider.setValue(preset["low"])
        self.eq_mid_slider.setValue(preset["mid"])
        self.eq_high_slider.setValue(preset["high"])
        self.eq_low_slider.blockSignals(False)
        self.eq_mid_slider.blockSignals(False)
        self.eq_high_slider.blockSignals(False)

        self.change_eq()

    def save_current_as_preset(self):
        state = self.current_audio_state()

        default_name = tr(
            f"プリセット{len(self.audio_presets) + 1}",
            f"Preset {len(self.audio_presets) + 1}"
        )
        name, ok = QInputDialog.getText(
            self,
            tr("プリセット保存", "Save Preset"),
            tr("プリセット名:", "Preset name:"),
            text=default_name
        )

        name = name.strip() if ok else ""

        if not name:
            self.sync_preset_combo()
            return

        preset = {"name": name, **state}
        self.audio_presets.append(normalize_audio_preset(preset))
        self.save_audio_presets()
        self.rebuild_preset_combo()
        self.select_preset_by_name(name)

    def manage_presets(self):
        dialog = AudioPresetDialog(
            self,
            self.audio_presets
        )

        if dialog.exec() != QDialog.Accepted:
            self.sync_preset_combo()
            return

        self.audio_presets = [
            normalize_audio_preset(p)
            for p in dialog.presets
        ]
        self.save_audio_presets()
        self.rebuild_preset_combo()
        self.sync_preset_combo()

    def select_preset_by_name(self, name):
        for index in range(1, self.preset_combo.count()):
            if self.preset_combo.itemText(index) == name:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(index)
                self.preset_combo.blockSignals(False)
                return

    def save_audio_presets(self):
        save_value(
            AUDIO_PRESETS_KEY,
            json.dumps(self.audio_presets, ensure_ascii=False)
        )

    def update_spectrum(self):
        if getattr(self.audio, "y_mono", None) is None:
            return

        if getattr(self.audio, "spectrum", None) is None:
            self.audio.spectrum = SpectrumData()

        if self.editor.spectrum is not self.audio.spectrum:
            self.editor.spectrum = self.audio.spectrum

        # Show progress or just block briefly
        # Since analyze is fast enough, we just run it directly. If it takes too long, 
        # it should be put in a thread. But it's usually acceptable for CQT cache warmups.
        self.audio.spectrum.analyze(
            self.audio.y_mono,
            self.audio.sr,
            self.editor.min_pitch,
            self.editor.max_pitch,
            self.audio.a4_freq
        )
        self.editor.warm_spectrum_cache()
        self.editor.update()


    def change_sensitivity(self, value):
        self.editor.spectrum_db_range = float(value)
        self.editor._spectrum_key = None
        self.editor._spectrum_image = None
        self.editor.update()

        save_value(
            "spectrum_sensitivity",
            str(value)
        )

    def _project_data(self):
        project = {
            "midi_tracks": [],
            "midi_filter_track": self.midi.filter_track,
            "midi_tempos": self.midi.tempos,
            "midi_timesigs": self.midi.time_signatures,
            "midi_beat_phase": getattr(self.midi, "beat_phase", 0.0),
            "audio_offset": self._global_audio.offset,
            "audio_volume": self._global_audio.volume,
            "audio_file": self._global_audio.file_path,
            "audio_a4_freq": getattr(self._global_audio, "a4_freq", 440.0),
            "audio_channel": getattr(self._global_audio, "channel_mode", 0),
            "audio_eq_low": getattr(self._global_audio, "eq_low", 1.0),
            "audio_eq_mid": getattr(self._global_audio, "eq_mid", 1.0),
            "audio_eq_high": getattr(self._global_audio, "eq_high", 1.0),
            "audio_muted": getattr(self._global_audio, "audio_muted", False),
        }

        for index, track in enumerate(self.midi.tracks):
            def note_dict(note):
                data = {
                    "start": note.start,
                    "duration": note.duration,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "channel": getattr(note, "channel", track.channel),
                }

                if ENABLE_LYRICS:
                    data["lyric"] = getattr(note, "lyric", "")

                return data

            track_audio = self.track_audio.get(index)

            if (
                track_audio is not None and
                track_audio.file_path
            ):
                track.audio_file = track_audio.file_path

                track.audio_params = self._collect_audio_params(
                    track_audio
                )

            project["midi_tracks"].append({
                "name": track.name,
                "channel": track.channel,
                "notes": [
                    note_dict(note)
                    for note in track.notes
                ],
                "pedals": [
                    {
                        "time": pedal.time,
                        "down": pedal.down,
                    }
                    for pedal in track.pedals
                ],
                "audio_file": getattr(track, "audio_file", "") or "",
                "audio_params": dict(
                    getattr(track, "audio_params", {})
                ),
            })

        return project

    def _project_state(self):
        counts = (
            tuple(
                len(track.notes)
                for track in self.midi.tracks
            ),
            tuple(
                len(track.pedals)
                for track in self.midi.tracks
            ),
            tuple(self.midi.tempos),
            len(self.midi.time_signatures),
            float(getattr(self.midi, "beat_phase", 0.0)),
            self._global_audio.offset,
            self._global_audio.volume,
            getattr(self._global_audio, "a4_freq", 440.0),
            getattr(self._global_audio, "channel_mode", 0),
            getattr(self._global_audio, "eq_low", 1.0),
            getattr(self._global_audio, "eq_mid", 1.0),
            getattr(self._global_audio, "eq_high", 1.0),
            getattr(self._global_audio, "audio_muted", False),
            self.midi.filter_track,
            self._track_audio_state(),
        )
        return counts

    def _track_audio_state(self):
        state = []

        for index, track in enumerate(self.midi.tracks):
            audio_file = (
                getattr(track, "audio_file", "") or ""
            )

            params = dict(
                getattr(track, "audio_params", {}) or {}
            )

            ta = self.track_audio.get(index)

            if ta is not None and ta.file_path:
                audio_file = ta.file_path

                params = self._collect_audio_params(ta)

            state.append(
                (
                    audio_file,
                    tuple(
                        sorted(params.items())
                    )
                )
            )

        return tuple(state)

    def _mark_project_saved(self):
        self._saved_project_state = self._project_state()
        self.update_title()

    def project_is_modified(self):
        return (
            self._saved_project_state is not None and
            self._project_state() != self._saved_project_state
        )

    def update_title(self):
        if self._project_path:
            name = Path(self._project_path).name
            base = f"WaveNoteMIDIEditor - {name}"
        elif self.audio.file_path:
            name = Path(self.audio.file_path).name
            base = f"WaveNoteMIDIEditor - {name}"
        else:
            base = "WaveNoteMIDIEditor"

        if self.project_is_modified():
            title = f"{base} *"
        else:
            title = base

        if self.windowTitle() != title:
            self.setWindowTitle(title)

    def new_project(self):
        if self.project_is_modified():
            answer = QMessageBox.question(
                self,
                tr("未保存の変更", "Unsaved Changes"),
                tr(
                    "現在のプロジェクトに変更があります。\n保存しますか？",
                    "The current project has been modified.\nSave changes?"
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if answer == QMessageBox.Cancel:
                return

            if answer == QMessageBox.Yes:
                if not self.save_project():
                    return

        self.audio.stop()
        self.midi = MidiData()
        self.midi.play_all_tracks = self.play_all_tracks_button.isChecked()

        for ta in self.track_audio.values():
            ta.clear()
            ta.close()
        self.track_audio.clear()
        self.track_audio = {}

        self.audio = self._global_audio
        self.editor.set_audio(self._global_audio)

        self.audio.clear()
        self.editor.set_midi(self.midi)
        self.editor.clear_audio()
        self.refresh_track_combo()
        self._project_path = None
        self._mark_project_saved()
        self._tempo_analyzed = False
        self._refresh_timeline_reference()
        self.editor.update_timeline()
        self.editor.update()

    def _save_project_to_path(self, path):
        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(
                    self._project_data(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                str(e)
            )
            return False

        self._project_path = path
        save_last_dir_from_path(path, key="last_project_dir")
        self._mark_project_saved()
        return True

    def save_project(self):
        if self._project_path:
            return self._save_project_to_path(
                self._project_path
            )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "プロジェクトを保存",
            load_last_dir(key="last_project_dir"),
            "WaveNote Project (*.wnp);;All Files (*)"
        )

        if not path:
            return False

        if not path.lower().endswith(".wnp"):
            path += ".wnp"

        return self._save_project_to_path(path)

    def load_project(self, path):
        path = str(path).strip().strip('"').strip("'")
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(
                    f"プロジェクトファイルが見つかりません:\n{path}",
                    f"Project file not found:\n{path}"
                )
            )
            return

        save_last_dir_from_path(path, key="last_project_dir")

        try:
            project = read_project_json(path)
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(
                    f"プロジェクトの読み込みに失敗しました:\n{e}",
                    f"Failed to load project:\n{e}"
                )
            )
            return

        # MIDIトラックの再構築
        self.midi = MidiData()
        self.midi.play_all_tracks = self.play_all_tracks_button.isChecked()
        self.midi.tracks.clear()
        raw_tempos = project.get("midi_tempos", [(0.0, 120.0)])
        self.midi.tempos = [
            (float(t[0]), float(t[1])) for t in raw_tempos
        ] if raw_tempos else [(0.0, 120.0)]
        self.midi.bpm = float(self.midi.tempos[0][1])

        raw_timesigs = project.get("midi_timesigs", [(0.0, 4, 4)])
        self.midi.time_signatures = [
            (float(ts[0]), int(ts[1]), int(ts[2])) for ts in raw_timesigs
        ] if raw_timesigs else [(0.0, 4, 4)]

        # グリッド原点(タップ計測などで設定される)を復元しないと
        # 開き直した時にMIDIに対してグリッドがずれる
        self.midi.beat_phase = float(
            project.get("midi_beat_phase", 0.0)
        )

        self.midi.has_file = bool(project.get("midi_tracks"))

        # トラックの再構築
        for ta in self.track_audio.values():
            ta.clear()
            ta.close()
        self.track_audio.clear()
        self.track_audio = {}

        self._project_path = path
        self._tempo_analyzed = False

        for track_data in project.get("midi_tracks", []):
            track = self.midi.add_track(name=track_data.get("name"))
            track.channel = track_data.get("channel", len(self.midi.tracks) - 1)
            track.audio_file = track_data.get("audio_file", "") or ""
            track.audio_params = dict(
                track_data.get("audio_params", {}) or {}
            )
            for note_data in track_data.get("notes", []):
                note_kwargs = {
                    "start": note_data["start"],
                    "duration": note_data["duration"],
                    "pitch": note_data["pitch"],
                    "velocity": note_data.get("velocity", 100),
                    "channel": note_data.get("channel", track.channel),
                }

                if ENABLE_LYRICS:
                    note_kwargs["lyric"] = note_data.get("lyric", "")

                track.notes.append(
                    Note(**note_kwargs)
                )
            for pedal_data in track_data.get("pedals", []):
                track.pedals.append(
                    PedalEvent(
                        time=pedal_data["time"],
                        down=pedal_data["down"]
                    )
                )
            track.pedals.sort(key=lambda e: e.time)

        # トラック毎に固有のMIDIチャンネルを割り当て、
        # CC64(サステイン)がトラック間で共有されないようにする
        self.midi.ensure_unique_channels()

        # フィルタトラックの復元
        filter_track = project.get("midi_filter_track")
        if filter_track is not None and 0 <= filter_track < len(self.midi.tracks):
            self.midi.set_filter_track(filter_track)

        self.midi._bump()

        # 全体トラック用の音声設定の復元
        project_audio_offset = project.get("audio_offset", 0.0)
        project_audio_volume = project.get("audio_volume", 0.5)
        project_audio_a4 = project.get("audio_a4_freq", 440.0)
        project_audio_channel = project.get("audio_channel", 0)
        project_audio_eq_low = project.get("audio_eq_low", 1.0)
        project_audio_eq_mid = project.get("audio_eq_mid", 1.0)
        project_audio_eq_high = project.get("audio_eq_high", 1.0)
        project_audio_muted = bool(project.get("audio_muted", False))
        self._global_audio.clear()
        self._global_audio.offset = project_audio_offset
        self._global_audio.volume = project_audio_volume
        self._global_audio.a4_freq = project_audio_a4
        self._global_audio.channel_mode = int(project_audio_channel)
        self._global_audio.eq_low = float(project_audio_eq_low)
        self._global_audio.eq_mid = float(project_audio_eq_mid)
        self._global_audio.eq_high = float(project_audio_eq_high)
        self._global_audio.audio_muted = project_audio_muted

        # エディタにMIDIを設定
        self.editor.set_midi(self.midi)
        self._global_audio.set_midi(self.midi)

        # 復元したトラック選択に応じたアクティブな音声へ切り替える。
        # (トラックに専用音声の登録がある場合はそのトラック専用の
        #  AudioDataが_ensure_track_audio経由で生成され、パラメータも復元される)
        target_audio = self._current_audio()
        self.audio = target_audio
        self.editor.set_audio(target_audio)

        # 音声ファイルの解決とスペクトラム解析（非同期）
        if target_audio is self._global_audio:
            audio_file = project.get("audio_file")
            restore_params = {
                "volume": project_audio_volume,
                "offset": project_audio_offset,
                "a4": project_audio_a4,
                "channel": project_audio_channel,
                "eq_low": project_audio_eq_low,
                "eq_mid": project_audio_eq_mid,
                "eq_high": project_audio_eq_high,
                "muted": project_audio_muted,
            }
        else:
            active_index = self.midi.filter_track
            audio_file = self.midi.tracks[active_index].audio_file
            restore_params = (
                self.midi.tracks[active_index].audio_params or None
            )

        resolved_audio_file = self._resolve_audio_path(
            audio_file
        ) if audio_file else None

        if resolved_audio_file:
            # 全体トラック用の音声を先に読み込む。
            # 専用音声のないトラックは全体トラック用の音声を参照するため、
            # プロジェクトを開いた直後から再生できるようにする。
            # 作曲テンポの基準も全体トラック用の音声にする。
            if target_audio is not self._global_audio:
                global_file = project.get("audio_file")
                global_resolved = (
                    self._resolve_audio_path(global_file)
                    if global_file else None
                )
                if (
                    global_resolved and
                    os.path.abspath(global_resolved) !=
                    os.path.abspath(resolved_audio_file)
                ):
                    global_analyzed_tempo = (
                        not self.midi.has_file and
                        not self._tempo_analyzed
                    )
                    self._load_audio_into(
                        self._global_audio,
                        global_resolved,
                        tr("プロジェクトを開く", "Open Project"),
                        restore_params={
                            "volume": project_audio_volume,
                            "offset": project_audio_offset,
                            "a4": project_audio_a4,
                            "channel": project_audio_channel,
                            "eq_low": project_audio_eq_low,
                            "eq_mid": project_audio_eq_mid,
                            "eq_high": project_audio_eq_high,
                            "muted": project_audio_muted,
                        }
                    )
                    if global_analyzed_tempo:
                        # 全体トラック用の音声がテンポの基準になったので、
                        # 続いて読み込むトラック専用の音声では上書きしない。
                        self._tempo_analyzed = True

            self._load_audio_into(
                target_audio,
                resolved_audio_file,
                tr("プロジェクトを開く", "Open Project"),
                restore_params=restore_params
            )
            self.update_editor()
        else:
            target_audio.clear()
            self.editor.clear_audio()
            self.editor.update_timeline()
            if audio_file:
                QMessageBox.warning(
                    self,
                    tr("音声ファイルが見つかりません", "Audio File Not Found"),
                    tr(
                        f"プロジェクトに登録されている音声ファイルが見つかりませんでした:\n{audio_file}",
                        f"The audio file registered in the project was not found:\n{audio_file}"
                    )
                )

        self.refresh_track_combo()
        self.editor.set_track_filter(self.midi.filter_track)
        self.editor.set_play_position(0.0)
        self.editor.scroll_x = 0.0
        self._refresh_timeline_reference()
        self._sync_audio_ui()
        self.editor.update()

        self._project_path = path
        self._mark_project_saved()



    def undo(self):
        if not self.midi.undo():
            return
            
        if "audio_offset" in getattr(self.midi, "extra_state", {}):
            self.audio.offset = self.midi.extra_state["audio_offset"]

        self.after_edit()

    def redo(self):
        if not self.midi.redo():
            return
            
        if "audio_offset" in getattr(self.midi, "extra_state", {}):
            self.audio.offset = self.midi.extra_state["audio_offset"]

        self.after_edit()

    def after_edit(self):
        self.refresh_track_combo()
        self.editor.bpm = self.midi.bpm
        self.editor.selected_notes = []

        self._refresh_timeline_reference()

        self.editor.update_timeline()

        self.editor.update()

    def change_note_length(self, index):
        beats = (
            self.length_combo.currentData()
        )

        self.editor.note_length = beats
        self.editor.placement_beats = beats

        self.editor.update()

    def toggle_lyric_mode(self):
        if not ENABLE_LYRICS:
            return

        has_notes = any(
            track.notes
            for track in self.midi.tracks
        )

        if not self.editor.lyric_mode and not has_notes:
            QMessageBox.information(
                self,
                tr("歌詞入力モード", "Lyric Input Mode"),
                tr(
                    "ノーツがありません。\n"
                    "先にノーツを追加してください。",
                    "There are no notes.\n"
                    "Add notes first."
                )
            )
            action = self.actions.get("action_lyric_mode")
            if action is not None:
                action.setChecked(False)
            return

        enabled = self.editor.toggle_lyric_mode()

        action = self.actions.get("action_lyric_mode")
        if action is not None and action.isChecked() != enabled:
            action.setChecked(enabled)

        if enabled:
            self.statusBar().showMessage(
                tr(
                    "歌詞入力モード: ノーツをクリックして歌詞を入力"
                    "(選択中のノーツは時系列順に連続入力 / "
                    "右ドラッグで範囲選択 / Escで終了)",
                    "Lyric input mode: click notes to enter lyrics"
                    "(selected notes are filled in chronological order / "
                    "right-drag to select a range / Esc to exit)"
                ),
                8000
            )
        else:
            self.statusBar().showMessage(
                tr("歌詞入力モードを終了しました", "Lyric input mode exited"),
                3000
            )

    def apply_tapped_tempo(self, bpm, phi_time, start_time):
        try:
            self.midi.push_undo()
            self.midi.apply_tempo_fit(start_time, bpm, phi_time)
        except Exception:
            return

        self.editor.bpm = self.midi.bpm
        self.after_edit()
        self.statusBar().showMessage(
            tr(
                f"計測したテンポを適用しました: {bpm:.2f} BPM "
                f"(開始位置 {max(0.0, float(start_time)):.2f}s)",
                f"Applied measured tempo: {bpm:.2f} BPM "
                f"(start at {max(0.0, float(start_time)):.2f}s)"
            ),
            5000
        )

    def insert_tempo(self):
        current_tempo = self.midi.tempo_at(self.editor.play_position)
        
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("テンポの追加", "Add Tempo"))
        layout = QVBoxLayout(dialog)
        
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel(tr("新しいテンポ (BPM):", "New tempo (BPM):")))
        tempo_spin = QDoubleSpinBox()
        tempo_spin.setRange(20.0, 999.0)
        tempo_spin.setDecimals(1)
        tempo_spin.setValue(current_tempo)
        hlayout.addWidget(tempo_spin)
        layout.addLayout(hlayout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        
        # プレビュー用のテンポ変更をリアルタイムに反映するための一時退避
        self.midi.push_undo()
        
        def preview_tempo(val):
            # 現在のUndoスタックをロールバックしてから再度追加する
            self.midi.undo()
            self.midi.push_undo()
            self.midi.add_tempo(self.editor.play_position, val)
            self.editor.bpm = self.midi.bpm
            self.editor.update_timeline()
            self.editor.update()
            
        tempo_spin.valueChanged.connect(preview_tempo)
        
        # 初期状態でもプレビューを適用する
        preview_tempo(current_tempo)
        
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        if dialog.exec() == QDialog.Accepted:
            self.after_edit()
        else:
            self.midi.undo()
            self.editor.update_timeline()
            self.editor.update()

    def insert_timesig(self):
        current_num, current_den = self.midi.time_sig_at(self.editor.play_position)
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("拍子の追加", "Add Time Signature"))
        layout = QVBoxLayout(dialog)
        
        hlayout = QHBoxLayout()
        num_spin = QSpinBox()
        num_spin.setRange(1, 32)
        num_spin.setValue(current_num)
        den_spin = QSpinBox()
        den_spin.setRange(1, 32)
        den_spin.setValue(current_den)
        
        hlayout.addWidget(num_spin)
        hlayout.addWidget(QLabel("/"))
        hlayout.addWidget(den_spin)
        layout.addLayout(hlayout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            self.midi.push_undo()
            self.midi.add_time_signature(
                self.editor.play_position,
                num_spin.value(),
                den_spin.value()
            )
            self.after_edit()

    def clear_audio(self):
        self._analysis_token += 1
        self._analysis_ready = False
        self._analysis_error = None
        self._pending_audio_duration = None
        self._pending_tempo_analysis = None
        self._pending_tap_onsets = None
        self._tempo_analyzed = False

        active_index = self._active_track_index()

        if active_index is not None:
            if (
                0 <= active_index <
                len(self.midi.tracks)
            ):
                track = self.midi.tracks[active_index]
                track.audio_file = ""
                track.audio_params = {}

            ta = self.track_audio.pop(active_index, None)

            if ta is not None:
                ta.clear()
                ta.close()

            self.audio = self._global_audio

            self.editor.set_audio(self._global_audio)
        else:
            self.audio.clear()

            self.editor.clear_audio()

        self._refresh_timeline_reference()

        self._sync_audio_ui()

        self.update_title()

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("オーディオを開く", "Open Audio"),
            load_last_dir(),
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )

        if path:
            self.load_audio_file(path)

    def load_audio_file(self, path):
        path = str(path).strip().strip('"').strip("'")
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(f"音声ファイルが見つかりません:\n{path}", f"Audio file not found:\n{path}")
            )
            return

        save_last_dir_from_path(path)

        val, ok = QInputDialog.getDouble(
            self,
            tr("基準周波数の設定", "Reference Pitch"),
            tr("A=440Hz以外のチューニングを使用する場合は変更してください：", "Change this if the tuning is not A=440Hz:"),
            440.0,
            400.0,
            500.0,
            1
        )
        if not ok:
            return
        
        a4_freq = val

        try:
            # 「すべてのトラック」選択時は全体トラック用の音声へ、
            # 単一トラック選択時はそのトラック専用の音声へ読み込む。
            track_index = self.midi.filter_track
            is_track_audio = (
                track_index is not None and
                0 <= track_index < len(self.midi.tracks)
            )

            if is_track_audio:
                audio = self._ensure_track_audio(track_index)
            else:
                audio = self._global_audio

            audio.a4_freq = a4_freq

            self._load_audio_into(
                audio,
                path,
                tr("オーディオを開く", "Open Audio")
            )

            if is_track_audio:
                track = self.midi.tracks[track_index]
                track.audio_file = path
                track.audio_params = self._collect_audio_params(
                    audio
                )

            self._apply_audio_context()
            self.update_editor()
            self.editor.set_play_position(0.0)
            self.editor.scroll_x = 0.0
            self.editor.update_timeline()
            self.editor.update()

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                str(e)
            )

        self.update_title()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("プロジェクトを開く", "Open Project"),
            load_last_dir(key="last_project_dir"),
            "WaveNote Project (*.wnp);;All Files (*)"
        )

        if path:
            self.load_project(path)

    def open_midi(self):
        if ENABLE_SVP:
            title = tr("MIDI / SVPを開く", "Open MIDI / SVP")
            file_filter = (
                "MIDI / Synthesizer V Project (*.mid *.midi *.svp);;"
                "MIDI Files (*.mid *.midi);;"
                "Synthesizer V Project (*.svp)"
            )
        else:
            title = tr("MIDIを開く", "Open MIDI")
            file_filter = "MIDI Files (*.mid *.midi)"

        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            load_last_dir(),
            file_filter
        )

        if path:
            self.load_midi_file(path)

    def load_midi_file(self, path):
        path = str(path).strip().strip('"').strip("'")
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(f"ファイルが見つかりません:\n{path}", f"File not found:\n{path}")
            )
            return

        save_last_dir_from_path(path)

        try:
            if Path(path).suffix.lower() == ".svp" and ENABLE_SVP:
                self.midi.load_svp(path)
            elif Path(path).suffix.lower() == ".svp":
                raise ValueError(
                    tr(
                        "このバージョンではSVPファイルを読み込めません",
                        "SVP files are not supported in this version"
                    )
                )
            else:
                self.midi.load(path)

            self._last_midi_path = path
            self._last_midi_is_svp = (
                ENABLE_SVP and
                Path(path).suffix.lower() == ".svp"
            )

            self.midi.clear_history()

            for ta in self.track_audio.values():
                ta.clear()
                ta.close()
            self.track_audio.clear()
            self.track_audio = {}

            self.audio = self._global_audio
            self.editor.set_audio(self._global_audio)

            self.editor.set_track_filter(
                0
            )

            self.refresh_track_combo()
            self._tempo_analyzed = False

            self.editor.set_play_position(0.0)
            self.audio.position = 0.0
            self.editor.scroll_x = 0.0
            self._refresh_timeline_reference()
            self.editor.update_timeline()
            self.editor.update()
            self._project_path = None
            self.update_title()

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                str(e)
            )

    def save_midi(self):
        if self._last_midi_path:
            try:
                if self._last_midi_is_svp and ENABLE_SVP:
                    self.midi.save_svp(self._last_midi_path)
                    QMessageBox.information(
                        self,
                        tr("完了", "Done"),
                        tr("SVPファイルを上書き保存しました。", "SVP file saved (overwrite).")
                    )
                else:
                    self.midi.save(self._last_midi_path)
                    QMessageBox.information(
                        self,
                        tr("完了", "Done"),
                        tr("MIDIファイルを上書き保存しました。", "MIDI file saved (overwrite).")
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    tr("エラー", "Error"),
                    tr(
                        f"保存中にエラーが発生しました:\n{e}",
                        f"An error occurred while saving:\n{e}"
                    )
                )
        else:
            self.save_midi_as()

    def save_midi_as(self):
        if ENABLE_SVP:
            file_filter = (
                "MIDI Files (*.mid *.midi);;"
                "WAV Files (*.wav);;"
                "Synthesizer V Project (*.svp)"
            )
        else:
            file_filter = "MIDI Files (*.mid *.midi);;WAV Files (*.wav)"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("書き出しを保存", "Export As"),
            load_last_dir(),
            file_filter
        )

        if not path:
            return

        save_last_dir_from_path(path)

        try:
            if selected_filter == "WAV Files (*.wav)" or path.lower().endswith(".wav"):
                if not path.lower().endswith(".wav"):
                    path += ".wav"
                    
                # プログレスダイアログを表示してWAVエクスポート
                progress = QProgressDialog(
                    tr("WAVファイルを出力中...", "Exporting WAV file..."),
                    tr("キャンセル", "Cancel"), 0, 0, self
                )
                progress.setWindowTitle(tr("書き出し", "Export"))
                progress.setWindowModality(Qt.WindowModal)
                progress.setCancelButton(None)
                progress.show()
                QApplication.processEvents()
                
                self.audio.export_wav(path)
                
                progress.close()
                QMessageBox.information(
                    self,
                    tr("完了", "Done"),
                    tr("WAVファイルの書き出しが完了しました。", "WAV export completed.")
                )
            elif (
                ENABLE_SVP and
                (
                    selected_filter == "Synthesizer V Project (*.svp)" or
                    path.lower().endswith(".svp")
                )
            ):
                if not path.lower().endswith(".svp"):
                    path += ".svp"

                self.midi.save_svp(path)
                self._last_midi_path = path
                self._last_midi_is_svp = True
                QMessageBox.information(
                    self,
                    tr("完了", "Done"),
                    tr("SVPファイルの保存が完了しました。", "SVP file saved.")
                )
            else:
                if not path.lower().endswith((".mid", ".midi")):
                    path += ".mid"
                self.midi.save(path)
                self._last_midi_path = path
                self._last_midi_is_svp = False
                QMessageBox.information(
                    self,
                    tr("完了", "Done"),
                    tr("MIDIファイルの保存が完了しました。", "MIDI file saved.")
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("エラー", "Error"),
                tr(
                    f"保存中にエラーが発生しました:\n{e}",
                    f"An error occurred while saving:\n{e}"
                )
            )

    def update_editor(self):
        self.update_title()
        self.audio.update_position()

        if self._pending_audio_duration is not None:
            self.editor.set_audio_duration(self._pending_audio_duration)
            self._pending_audio_duration = None
            self.editor.update_timeline()

        if self._pending_tap_onsets is not None:
            onset_times, onset_strengths = self._pending_tap_onsets
            self._pending_tap_onsets = None
            self.editor.set_tap_onsets(onset_times, onset_strengths)

        if self._analysis_ready:
            self._analysis_ready = False

            if self._analysis_error is not None:
                error = self._analysis_error
                self._analysis_error = None
                QMessageBox.critical(
                    self,
                    tr("エラー", "Error"),
                    str(error)
                )
            else:
                if self._pending_tempo_analysis is not None:
                    bpm, _beat_origin = self._pending_tempo_analysis
                    self._pending_tempo_analysis = None

                    self.midi.set_base_tempo(bpm)

                    self._tempo_analyzed = True

                    # グリッドの拍1が必ず開始位置(0秒)に来るようにする。
                    # 開始位置より前に拍が読めるグリッドを作らない。
                    # beat_to_time(1) = (1 - beat_phase) * (60/bpm) = 0
                    self.midi.set_beat_phase(
                        1.0
                    )
                    self.editor.bpm = bpm



                self.editor.update_timeline()
                self.editor.warm_spectrum_cache()
                self.editor.update()

        self.editor.update_timeline()

        previous_position = (
            self.editor.play_position
        )

        if self.audio.playing:
            self.editor.set_play_position(
                self.audio.position
            )
        else:
            if getattr(self.audio, "auto_stopped", False):
                self.audio.auto_stopped = False
                if getattr(self.editor, "return_to_start_on_stop", True):
                    self.editor.stop()

        if (
            self.offset_box.value() !=
            self.audio.offset
        ):
            self.offset_box.blockSignals(True)
            self.offset_box.setValue(
                self.audio.offset
            )
            self.offset_box.blockSignals(False)

        if (
            self.editor.play_position !=
            previous_position
        ):
            self.editor.update()

        self.update_status_labels()

        if hasattr(self.editor, "auto_scroll"):
            self.editor.auto_scroll()

if __name__ == "__main__":
    try:
        import ctypes
        myappid = "wavenote.midi.editor.v2"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    from PySide6.QtCore import qInstallMessageHandler, QtMsgType
    def qt_message_handler(mode, context, message):
        if mode == QtMsgType.QtWarningMsg and "QThreadStorage" in message:
            return
        
        # 開発中の他の重要なエラー等は見落とさないように、それ以外は標準出力へ
        if mode == QtMsgType.QtWarningMsg:
            print(f"Warning: {message}")
        elif mode == QtMsgType.QtCriticalMsg:
            print(f"Critical: {message}")
        elif mode == QtMsgType.QtFatalMsg:
            print(f"Fatal: {message}")

    qInstallMessageHandler(qt_message_handler)

    app = QApplication(
        sys.argv
    )

    icon_path = get_resource_path(os.path.join("Assets", "icon.ico"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    app.setStyle(
        "Fusion"
    )

    theme = load_value("theme", "auto")
    if theme == "light":
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    elif theme == "dark":
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    else:
        app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)

    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(initial_file=initial_file)
    window.show()

    ret = app.exec()
    import os
    os._exit(ret)
