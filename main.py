import sys
import threading
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QComboBox,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import QTimer, Qt
from audio import AudioData
from spectrum import SpectrumData
from midi import MidiData, Note, PedalEvent
from piano_roll import PianoRoll
from midiout import list_ports
from settings import load_value, save_value

class SettingsDialog(QDialog):
    def __init__(self, parent=None, current="internal"):
        super().__init__(parent)

        self.setWindowTitle("設定")

        self._device_combo = QComboBox()

        self._device_combo.addItem(
            "内蔵音源",
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

        refresh_button = QPushButton(
            "デバイスを更新"
        )
        refresh_button.clicked.connect(
            self._refresh_devices
        )

        form = QFormLayout()
        form.addRow(
            "MIDI出力",
            self._device_combo
        )
        form.addRow(
            "",
            refresh_button
        )

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

    def _refresh_devices(self):
        current = self._device_combo.currentData()

        self._device_combo.blockSignals(True)

        self._device_combo.clear()

        self._device_combo.addItem(
            "内蔵音源",
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WaveNoteMIDIEditor")
        self.showMaximized()

        self.audio = AudioData()
        self.spectrum = SpectrumData()
        self.midi = MidiData()

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

        self.setCentralWidget(self.editor)

        self.editor.note_length = 0.25
        self.editor.placement_beats = 0.25

        self.editor.marker_edited.connect(
            self.after_edit
        )

        self.create_menu()
        self.create_toolbar()

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
        self._project_path = None
        self._saved_project_state = None

        self._mark_project_saved()

    def closeEvent(self, event):
        if self.project_is_modified():
            answer = QMessageBox.question(
                self,
                "プロジェクトを保存",
                "プロジェクトに未保存の変更があります。保存しますか？",
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

        self.audio.stop()
        self.audio._close_midi_out()
        super().closeEvent(event)

    def create_menu(self):
        file_menu = self.menuBar().addMenu(
            "ファイル"
        )

        open_audio_action = QAction(
            "オーディオを開く",
            self
        )
        open_audio_action.triggered.connect(
            self.open_audio
        )

        open_midi_action = QAction(
            "MIDIを開く",
            self
        )
        open_midi_action.triggered.connect(
            self.open_midi
        )

        new_midi_action = QAction(
            "MIDIを新規作成",
            self
        )
        new_midi_action.setShortcut(
            QKeySequence.StandardKey.New
        )
        new_midi_action.triggered.connect(
            self.new_midi
        )

        save_project_action = QAction(
            "プロジェクトを保存",
            self
        )
        save_project_action.setShortcut(
            QKeySequence("Shift+S")
        )
        save_project_action.triggered.connect(
            self.save_project
        )

        load_project_action = QAction(
            "プロジェクトを開く",
            self
        )
        load_project_action.triggered.connect(
            self.open_project
        )

        clear_audio_action = QAction(
            "音楽を空にする",
            self
        )
        clear_audio_action.triggered.connect(
            self.clear_audio
        )

        save_action = QAction(
            "MIDIを保存",
            self
        )
        save_action.triggered.connect(
            self.save_midi
        )
        # Keep common project and source-file actions in workflow order.
        file_menu.addAction(
            load_project_action
        )
        file_menu.addAction(
            save_project_action
        )
        file_menu.addAction(
            open_midi_action
        )
        file_menu.addAction(
            new_midi_action
        )
        file_menu.addAction(
            open_audio_action
        )
        file_menu.addAction(
            clear_audio_action
        )
        file_menu.addAction(
            save_action
        )

        exit_action = QAction(
            "終了",
            self
        )
        exit_action.triggered.connect(
            self.close
        )
        file_menu.addAction(
            exit_action
        )

        edit_menu = self.menuBar().addMenu(
            "編集"
        )

        undo_action = QAction(
            "元に戻す",
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
            "やり直し",
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

        playback_menu = self.menuBar().addMenu(
            "再生"
        )

        play_action = QAction(
            "再生 / 停止",
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

        stop_action = QAction(
            "停止",
            self
        )
        stop_action.triggered.connect(
            self.editor.stop
        )
        playback_menu.addAction(
            stop_action
        )

        settings_menu = self.menuBar().addMenu(
            "設定"
        )

        settings_action = QAction(
            "設定...",
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

        track_label = QLabel(
            "  トラック "
        )

        toolbar.addWidget(
            track_label
        )

        self.track_combo = QComboBox()

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
            "トラックを追加"
        )
        add_track_button.clicked.connect(
            self.add_track
        )

        toolbar.addWidget(
            add_track_button
        )

        tempo_label = QLabel(
            "  テンポ "
        )

        toolbar.addWidget(
            tempo_label
        )

        self.tempo_box = QSpinBox()

        self.tempo_box.setRange(
            20,
            999
        )

        self.tempo_box.setValue(
            self.midi.bpm
        )

        self.tempo_box.setSuffix(
            " BPM"
        )

        self.tempo_box.valueChanged.connect(
            self.change_tempo
        )

        toolbar.addWidget(
            self.tempo_box
        )

        insert_tempo_button = QPushButton(
            "テンポ追加"
        )
        insert_tempo_button.setToolTip(
            "再生位置にテンポを挿入"
        )
        insert_tempo_button.clicked.connect(
            self.insert_tempo
        )

        toolbar.addWidget(
            insert_tempo_button
        )

        timesig_label = QLabel(
            "  拍子 "
        )

        toolbar.addWidget(
            timesig_label
        )

        self.num_box = QSpinBox()

        self.num_box.setRange(
            1,
            32
        )

        self.num_box.setValue(4)

        toolbar.addWidget(
            self.num_box
        )

        slash_label = QLabel(
            "/"
        )

        toolbar.addWidget(
            slash_label
        )

        self.den_box = QSpinBox()

        self.den_box.setRange(
            1,
            32
        )

        self.den_box.setValue(4)

        toolbar.addWidget(
            self.den_box
        )

        insert_timesig_button = QPushButton(
            "拍子設定"
        )
        insert_timesig_button.setToolTip(
            "再生位置で拍子を変更"
        )
        insert_timesig_button.clicked.connect(
            self.insert_timesig
        )

        toolbar.addWidget(
            insert_timesig_button
        )

        length_label = QLabel(
            "  ノート長 "
        )

        toolbar.addWidget(
            length_label
        )

        self.length_combo = QComboBox()

        for name, beats in self.editor.note_lengths:
            self.length_combo.addItem(
                name,
                beats
            )

        self.length_combo.setCurrentIndex(4)

        self.length_combo.currentIndexChanged.connect(
            self.change_note_length
        )

        toolbar.addWidget(
            self.length_combo
        )

        offset_label = QLabel(
            "  音声オフセット "
        )

        toolbar.addWidget(
            offset_label
        )

        self.offset_box = QDoubleSpinBox()

        self.offset_box.setRange(
            -60.0,
            60.0
        )

        self.offset_box.setDecimals(
            3
        )

        self.offset_box.setSingleStep(
            0.005
        )

        self.offset_box.setSuffix(
            " s"
        )

        self.offset_box.valueChanged.connect(
            self.change_offset
        )

        toolbar.addWidget(
            self.offset_box
        )

        threshold_label = QLabel(
            "  スペクトラム閾値 "
        )

        toolbar.addWidget(
            threshold_label
        )

        self.threshold_slider = QSlider(
            Qt.Horizontal
        )

        self.threshold_slider.setRange(
            0,
            40
        )

        self.threshold_slider.setValue(
            int(
                self.editor.spectrum_threshold *
                100
            )
        )

        self.threshold_slider.setFixedWidth(
            150
        )

        self.threshold_slider.setToolTip(
            "弱い部分を非表示にする閾値"
        )

        self.threshold_slider.valueChanged.connect(
            self.change_threshold
        )

        toolbar.addWidget(
            self.threshold_slider
        )

        sensitivity_label = QLabel(
            "  スペクトラム感度 "
        )

        toolbar.addWidget(
            sensitivity_label
        )

        self.sensitivity_slider = QSlider(
            Qt.Horizontal
        )

        self.sensitivity_slider.setRange(
            10,
            100
        )

        self.sensitivity_slider.setValue(
            int(
                self.editor.spectrum_db_range
            )
        )

        self.sensitivity_slider.setFixedWidth(
            150
        )

        self.sensitivity_slider.setToolTip(
            "感度：低いほど鮮明、高いほど広範囲表示"
        )

        self.sensitivity_slider.valueChanged.connect(
            self.change_sensitivity
        )

        toolbar.addWidget(
            self.sensitivity_slider
        )

        volume_label = QLabel(
            "  音声ファイル音量 "
        )

        toolbar.addWidget(
            volume_label
        )

        self.volume_slider = QSlider(
            Qt.Horizontal
        )

        self.volume_slider.setRange(
            0,
            50
        )

        self.volume_slider.setValue(
            int(
                self.audio.volume *
                100
            )
        )

        self.volume_slider.setFixedWidth(
            150
        )

        self.volume_slider.setToolTip(
            "音声ファイルの音量"
        )

        self.volume_slider.valueChanged.connect(
            self.change_volume
        )

        toolbar.addWidget(
            self.volume_slider
        )

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
                "設定",
                "選択したデバイスを開けませんでした。\n内蔵音源に戻しました。"
            )

        save_value(
            "midi_out_device",
            self.audio.output_device
        )

    def refresh_track_combo(self):
        self.track_combo.blockSignals(True)

        self.track_combo.clear()

        self.track_combo.addItem(
            "すべてのトラック",
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

    def change_track(self, index):
        self.editor.set_track_filter(
            self.track_combo.currentData()
        )

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

    def change_offset(self, value):
        self.audio.offset = value
        self.editor.update()

    def change_threshold(self, value):
        self.editor.spectrum_threshold = (
            value /
            100.0
        )

        self.editor.update()

    def change_volume(self, value):
        self.audio.volume = (
            value /
            100.0
        )

    def change_sensitivity(self, value):
        self.editor.spectrum_db_range = float(value)
        self.editor._spectrum_key = None
        self.editor._spectrum_image = None
        self.editor.update()

    def _project_data(self):
        import json
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "プロジェクトを保存",
            "",
            "WaveNote Project (*.wnp);;All Files (*)"
        )
        if not path:
            return

        project = {
            "midi_tracks": [],
            "midi_filter_track": self.midi.filter_track,
            "midi_tempos": self.midi.tempos,
            "midi_timesigs": self.midi.time_signatures,
            "audio_offset": self.audio.offset,
            "audio_volume": self.audio.volume,
            "audio_file": self.audio.file_path,
        }

        for track in self.midi.tracks:
            track_data = []
            for note in track.notes:
                track_data.append({
                    "start": note.start,
                    "duration": note.duration,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "channel": getattr(note, 'channel', track.channel)
                })
            project["midi_tracks"].append({
                "name": track.name,
                "channel": track.channel,
                "notes": track_data,
                "pedals": [
                    {
                        "time": pedal.time,
                        "down": pedal.down
                    }
                    for pedal in track.pedals
                ]
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

    def _project_data(self):
        project = {
            "midi_tracks": [],
            "midi_filter_track": self.midi.filter_track,
            "midi_tempos": self.midi.tempos,
            "midi_timesigs": self.midi.time_signatures,
            "audio_offset": self.audio.offset,
            "audio_volume": self.audio.volume,
            "audio_file": self.audio.file_path,
        }

        for track in self.midi.tracks:
            project["midi_tracks"].append({
                "name": track.name,
                "channel": track.channel,
                "notes": [
                    {
                        "start": note.start,
                        "duration": note.duration,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "channel": getattr(note, "channel", track.channel),
                    }
                    for note in track.notes
                ],
                "pedals": [
                    {
                        "time": pedal.time,
                        "down": pedal.down,
                    }
                    for pedal in track.pedals
                ],
            })

        return project

    def _project_state(self):
        return json.dumps(
            self._project_data(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        )

    def _mark_project_saved(self):
        self._saved_project_state = self._project_state()

    def project_is_modified(self):
        return (
            self._saved_project_state is not None and
            self._project_state() != self._saved_project_state
        )

    def _save_project_to_path(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self._project_data(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                str(e)
            )
            return False

        self._project_path = path
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
            "",
            "WaveNote Project (*.wnp);;All Files (*)"
        )

        if not path:
            return False

        if not path.lower().endswith(".wnp"):
            path += ".wnp"

        return self._save_project_to_path(path)

    def load_project(self, path):
        import json
        import os

        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            project = json.load(f)

        # MIDIトラックの再構築
        self.midi = MidiData()
        self.midi.tracks.clear()
        self.midi.tempos = project.get("midi_tempos", [(0.0, 120.0)])
        self.midi.time_signatures = project.get("midi_timesigs", [(0.0, 4, 4)])

        # トラックの再構築
        for track_data in project.get("midi_tracks", []):
            track = self.midi.add_track(name=track_data.get("name"))
            track.channel = track_data.get("channel", len(self.midi.tracks) - 1)
            for note_data in track_data.get("notes", []):
                track.notes.append(
                    Note(
                        start=note_data["start"],
                        duration=note_data["duration"],
                        pitch=note_data["pitch"],
                        velocity=note_data.get("velocity", 100),
                        channel=note_data.get("channel", track.channel)
                    )
                )
            for pedal_data in track_data.get("pedals", []):
                track.pedals.append(
                    PedalEvent(
                        time=pedal_data["time"],
                        down=pedal_data["down"]
                    )
                )
            track.pedals.sort(key=lambda e: e.time)

        # フィルタトラックの復元
        filter_track = project.get("midi_filter_track")
        if filter_track is not None and 0 <= filter_track < len(self.midi.tracks):
            self.midi.set_filter_track(filter_track)

        self.midi._bump()

        # 音声設定の復元
        project_audio_offset = project.get("audio_offset", 0.0)
        project_audio_volume = project.get("audio_volume", 0.5)
        self.audio.offset = project_audio_offset
        self.audio.volume = project_audio_volume

        # エディタにMIDIを設定
        self.editor.set_midi(self.midi)
        self.audio.set_midi(self.midi)

        # 音声ファイルの読み込みとスペクトラム解析（非同期）
        audio_file = project.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            self._analysis_token += 1
            token = self._analysis_token
            self._analysis_ready = False
            self._analysis_error = None
            self._pending_audio_duration = None
            self._pending_tempo_analysis = None

            self.audio.clear()
            self.audio.offset = project_audio_offset
            self.audio.volume = project_audio_volume
            self.audio.file_path = audio_file
            self.editor.clear_audio()
            self.editor._spectrum_image = None
            self.editor._spectrum_key = None

            def worker():
                try:
                    duration = self.audio.load(audio_file)
                    if token != self._analysis_token:
                        return

                    self._pending_audio_duration = duration

                    self.spectrum.analyze(
                        self.audio.y,
                        self.audio.sr,
                        self.editor.min_pitch,
                        self.editor.max_pitch
                    )
                    if token == self._analysis_token:
                        self._analysis_ready = True
                except Exception as e:
                    if token == self._analysis_token:
                        self._analysis_error = e
                        self._analysis_ready = True

            thread = threading.Thread(
                target=worker,
                daemon=True
            )
            thread.start()
        else:
            self.editor.update_timeline()
        self.refresh_track_combo()
        self.editor.set_track_filter(self.midi.filter_track)
        self.editor.set_play_position(0.0)
        self.editor.scroll_x = 0.0
        self.editor.update()

        self._project_path = path
        self._mark_project_saved()

    def change_tempo(self, value):
        self.midi.set_base_tempo(value)
        self.editor.bpm = value
        self.editor.update()

    def undo(self):
        if not self.midi.undo():
            return

        self.after_edit()

    def redo(self):
        if not self.midi.redo():
            return

        self.after_edit()

    def after_edit(self):
        self.refresh_track_combo()

        self.tempo_box.blockSignals(True)
        self.tempo_box.setValue(
            int(
                round(
                    self.midi.bpm
                )
            )
        )
        self.tempo_box.blockSignals(False)

        num, den = self.midi.time_sig_at(
            0.0
        )

        self.num_box.blockSignals(True)
        self.num_box.setValue(num)
        self.num_box.blockSignals(False)

        self.den_box.blockSignals(True)
        self.den_box.setValue(den)
        self.den_box.blockSignals(False)

        self.editor.selected_notes = []

        self.editor.update_timeline()

        self.editor.update()

    def change_note_length(self, index):
        beats = (
            self.length_combo.currentData()
        )

        self.editor.note_length = beats
        self.editor.placement_beats = beats

        self.editor.update()

    def insert_tempo(self):
        bpm = float(
            self.tempo_box.value()
        )

        self.midi.push_undo()

        self.midi.add_tempo(
            self.editor.play_position,
            bpm
        )

        self.after_edit()

    def insert_timesig(self):
        num = self.num_box.value()
        den = self.den_box.value()

        self.midi.push_undo()

        self.midi.add_time_signature(
            self.editor.play_position,
            num,
            den
        )

        self.after_edit()

    def new_midi(self):
        self._analysis_token += 1
        self._analysis_ready = False
        self._analysis_error = None

        self.midi = MidiData()

        self.editor.set_midi(
            self.midi
        )

        self.tempo_box.blockSignals(True)
        self.tempo_box.setValue(
            int(
                round(
                    self.midi.bpm
                )
            )
        )
        self.tempo_box.blockSignals(False)

        num, den = self.midi.time_sig_at(
            0.0
        )

        self.num_box.blockSignals(True)
        self.num_box.setValue(num)
        self.num_box.blockSignals(False)

        self.den_box.blockSignals(True)
        self.den_box.setValue(den)
        self.den_box.blockSignals(False)

        self.refresh_track_combo()

        self.editor.set_track_filter(
            0
        )

        self.setWindowTitle(
            "WaveNoteMIDIEditor"
        )

    def clear_audio(self):
        self._analysis_token += 1
        self._analysis_ready = False
        self._analysis_error = None

        self.audio.clear()

        self.editor.clear_audio()

        self.offset_box.blockSignals(True)
        self.offset_box.setValue(0.0)
        self.offset_box.blockSignals(False)

        self.setWindowTitle(
            "WaveNoteMIDIEditor"
        )

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "オーディオを開く",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )

        if not path:
            return

        try:
            self._analysis_token += 1
            token = self._analysis_token
            self._analysis_ready = False
            self._analysis_error = None
            self._pending_audio_duration = None
            self._pending_tempo_analysis = None

            # A loaded MIDI already supplies the authoritative tempo map.
            analyze_tempo = not self.midi.has_file

            self.audio.clear()
            self.editor.clear_audio()
            self.editor._spectrum_image = None
            self.editor._spectrum_key = None

            def worker():
                try:
                    duration = self.audio.load(path)
                    if token != self._analysis_token:
                        return

                    self._pending_audio_duration = duration

                    self.spectrum.analyze(
                        self.audio.y,
                        self.audio.sr,
                        self.editor.min_pitch,
                        self.editor.max_pitch
                    )

                    if analyze_tempo:
                        try:
                            tempo_result = self.spectrum.analyze_tempo(
                                self.audio.y,
                                self.audio.sr
                            )
                        except Exception:
                            tempo_result = None

                        if token == self._analysis_token:
                            self._pending_tempo_analysis = tempo_result

                    if token == self._analysis_token:
                        self._analysis_ready = True
                except Exception as e:
                    if token == self._analysis_token:
                        self._analysis_error = e
                        self._analysis_ready = True

            thread = threading.Thread(
                target=worker,
                daemon=True
            )
            thread.start()

            self.editor.set_play_position(0.0)
            self.editor.scroll_x = 0.0
            self.editor.update_timeline()
            self.editor.update()

        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                str(e)
            )

        self.setWindowTitle(
            f"WaveNoteMIDIEditor - {Path(path).name}"
        )

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "プロジェクトを開く",
            "",
            "WaveNote Project (*.wnp);;All Files (*)"
        )

        if path:
            self.load_project(path)

    def open_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "MIDIを開く",
            "",
            "MIDI Files (*.mid *.midi)"
        )

        if not path:
            return

        try:
            self.midi.load(
                path
            )

            self.midi.clear_history()

            self.tempo_box.blockSignals(True)

            self.tempo_box.setValue(
                int(
                    round(
                        self.midi.bpm
                    )
                )
            )

            self.tempo_box.blockSignals(False)

            num, den = self.midi.time_sig_at(
                0.0
            )

            self.num_box.blockSignals(True)
            self.num_box.setValue(num)
            self.num_box.blockSignals(False)

            self.den_box.blockSignals(True)
            self.den_box.setValue(den)
            self.den_box.blockSignals(False)

            self.editor.set_track_filter(
                None
            )

            self.refresh_track_combo()

            self.editor.set_play_position(0.0)
            self.audio.position = 0.0
            self.editor.scroll_x = 0.0
            self.editor.update_timeline()
            self.editor.update()

        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                str(e)
            )

    def save_midi(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "MIDIを保存",
            "",
            "MIDI Files (*.mid *.midi)"
        )

        if not path:
            return

        try:
            self.midi.save(
                path
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                str(e)
            )

    def update_editor(self):
        self.audio.update_position()

        if self._pending_audio_duration is not None:
            self.editor.set_audio_duration(self._pending_audio_duration)
            self._pending_audio_duration = None
            self.editor.update_timeline()

        if self._analysis_ready:
            self._analysis_ready = False

            if self._analysis_error is not None:
                error = self._analysis_error
                self._analysis_error = None
                QMessageBox.critical(
                    self,
                    "エラー",
                    str(error)
                )
            else:
                if self._pending_tempo_analysis is not None:
                    bpm, _beat_origin = self._pending_tempo_analysis
                    self._pending_tempo_analysis = None

                    self.midi.set_base_tempo(bpm)
                    self.midi.set_beat_phase(0.0)
                    self.editor.bpm = bpm

                    self.tempo_box.blockSignals(True)
                    self.tempo_box.setValue(int(round(bpm)))
                    self.tempo_box.blockSignals(False)

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

if __name__ == "__main__":
    app = QApplication(
        sys.argv
    )

    app.setStyle(
        "Fusion"
    )

    window = MainWindow()
    window.show()

    sys.exit(
        app.exec()
    )
