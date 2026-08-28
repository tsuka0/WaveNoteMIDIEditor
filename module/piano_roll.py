import math
import time
import bisect
import numpy as np
from PySide6.QtWidgets import QWidget, QDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QVBoxLayout, QHBoxLayout, QDialogButtonBox, QMenu, QInputDialog
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QImage, QPixmap
from .midi import PedalEvent
from .taptempo import TapTempoEngine, MIN_TAPS_FOR_APPLY
from .i18n import tr
from .features import ENABLE_LYRICS

class PianoRoll(QWidget):
    marker_edited = Signal()
    track_switch_requested = Signal(int)
    tap_tempo_applied = Signal(float, float, float)

    def __init__(
        self,
        audio,
        spectrum,
        midi
    ):
        super().__init__()

        self.setMouseTracking(
            True
        )

        self.audio = audio
        self.spectrum = spectrum
        self.midi = midi

        self.audio.set_midi(
            self.midi
        )

        self.min_pitch = 21
        self.max_pitch = 108

        self.bpm = self.midi.bpm

        self.note_length = 1.0

        self.note_lengths = [
            (tr("全音符", "Whole note"), 4.0),
            (tr("付点2分音符", "Dotted half note"), 3.0),
            (tr("2分音符", "Half note"), 2.0),
            (tr("付点4分音符", "Dotted quarter note"), 1.5),
            (tr("4分音符", "Quarter note"), 1.0),
            (tr("4分3連音符", "Quarter triplet"), 2.0 / 3.0),
            (tr("付点8分音符", "Dotted eighth note"), 0.75),
            (tr("8分音符", "Eighth note"), 0.5),
            (tr("8分3連音符", "Eighth triplet"), 1.0 / 3.0),
            (tr("付点16分音符", "Dotted sixteenth note"), 0.375),
            (tr("16分音符", "Sixteenth note"), 0.25),
            (tr("16分3連音符", "Sixteenth triplet"), 1.0 / 6.0),
            (tr("32分音符", "Thirty-second note"), 0.125),
            (tr("32分3連音符", "Thirty-second triplet"), 1.0 / 12.0),
            (tr("付点64分音符", "Dotted sixty-fourth note"), 0.09375),
            (tr("64分音符", "Sixty-fourth note"), 0.0625),
            (tr("64分3連音符", "Sixty-fourth triplet"), 1.0 / 24.0),
            (tr("付点128分音符", "Dotted 128th note"), 0.046875),
            (tr("128分音符", "128th note"), 0.03125),
        ]

        self.left_width = 76
        self.top_height = 52
        self.bottom_height = 162
        self.velocity_lane_height = 80
        self.pedal_lane_height = 40
        self.scrub_height = 20

        self.note_height = 8

        measure_time = 240.0 / self.bpm

        self.seconds_per_pixel = (
            2.0 *
            measure_time
        ) / (
            1800 -
            self.left_width
        )

        self.spectrum_db_range = 50.0
        self.spectrum_threshold = 0.35
        self.return_to_start_on_stop = True

        self.track_colors = [
            (55, 185, 240),
            (240, 140, 60),
            (130, 220, 100),
            (230, 120, 200),
            (255, 215, 80),
            (160, 130, 250),
            (90, 220, 210),
            (235, 90, 90),
        ]

        self.audio_duration = 30.0
        self.play_position = 0.0

        self.scroll_x = 0.0
        self.scroll_y = 0.0

        self.drag_note = None
        self.drag_mode = None
        self.drag_start = None
        self.drag_original = None
        self.drag_track_index = 0

        self.drag_original_notes = None


        self.selected_notes = []
        self.clipboard_notes = []
        self.clipboard_pedals = []

        self.selection_mode = False
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None

        self.panning = False
        self.pan_start_x = 0.0
        self.pan_start_scroll = 0.0

        self.offset_dragging = False
        self.offset_drag_start = None
        self.offset_drag_value = 0.0

        self.scrubbing = False
        self.scrub_resume = False

        self._pre_play_scroll = 0.0

        self.vel_drag = None
        self._vel_undo_pushed = False

        self.pedal_drag = None
        self._pedal_undo_pushed = False

        self._spectrum_key = None
        self._spectrum_image = None
        self._spectrum_level_key = None
        self._spectrum_lo = -80.0
        self._spectrum_hi = 0.0
        self._spectrum_hop_dt = 0.0

        self._notes_starts_cache = None
        self._notes_starts_version = -1
        self._note_pens = None

        self.press_moved = False

        self.placement_beats = None

        self._previewed_pitch = None
        self._nudge_undo_pushed = False

        # --- 手動テンポ計測(タップテンポ) ---
        self.tap_mode = False
        self.tap_engine = TapTempoEngine()
        self._tap_fit = None
        self._tap_disp_bpm = 120.0
        self._tap_disp_phi = 0.0
        # 今回の再生が開始された位置(計測結果の追加先)
        self._tap_play_start = None

        # --- 歌詞入力モード ---
        self.lyric_mode = False

    def set_tap_onsets(self, times, strengths=None):
        """楽曲全体のオンセット強度をタップテンポエンジンへ登録する。

        WaveTone方式(タップ回帰×音量変化)の周期精密固定に使う。"""
        self.tap_engine.set_onsets(times, strengths)

        self._tap_anim_timer = QTimer(self)
        self._tap_anim_timer.setTimerType(
            Qt.TimerType.PreciseTimer
        )
        self._tap_anim_timer.timeout.connect(
            self._tap_tick
        )

        self.setFocusPolicy(
            Qt.StrongFocus
        )

    def set_audio_duration(
        self,
        duration
    ):
        self.audio_duration = max(
            duration,
            1.0
        )

        self.play_position = 0.0

    def set_midi(
        self,
        midi
    ):
        self.midi = midi

        self.audio.set_midi(midi)

        self.bpm = midi.bpm

        self.selected_notes = []
        self.selection_rect = None
        self.drag_note = None
        self.drag_mode = None

        self.scroll_x = 0.0

        self._notes_starts_cache = None
        self._notes_starts_version = -1
        self._note_pens = None

        self.play_position = 0.0
        self.audio.position = 0.0

        self.update_timeline()

        self.update()

    def clear_audio(
        self
    ):
        self.spectrum.data = None
        self.spectrum.times = None
        self.spectrum.midi_notes = None

        self._spectrum_key = None
        self._spectrum_image = None
        self._spectrum_level_key = None
        self._spectrum_hop_dt = 0.0

        self.play_position = 0.0

        self.update_timeline()

        self.update()

    def set_play_position(
        self,
        position
    ):
        self.play_position = max(
            0.0,
            position
        )

        self.follow_play_position()

    def split_notes_at_play_position(self):
        target_time = self.play_position
        target_notes = []

        if self.selected_notes:
            for note in self.selected_notes:
                if note.start < target_time - 1e-5 and (note.start + note.duration) > target_time + 1e-5:
                    target_notes.append(note)
        else:
            self._ensure_note_cache()

            if self.midi.filter_track is None:
                track_items = list(
                    enumerate(self.midi.tracks)
                )
            else:
                if 0 <= self.midi.filter_track < len(self.midi.tracks):
                    track_items = [
                        (self.midi.filter_track, self.midi.tracks[self.midi.filter_track])
                    ]
                else:
                    track_items = []

            for track_index, track in track_items:
                starts, notes, _, long_notes, long_starts = (
                    self._notes_starts_cache[
                        track_index
                    ]
                )

                i_end = bisect.bisect_left(
                    starts,
                    target_time
                )

                for note in notes[:i_end]:
                    if (note.start + note.duration) > target_time + 1e-5:
                        target_notes.append(note)

                li_end = bisect.bisect_left(
                    long_starts,
                    target_time
                )

                for note in long_notes[:li_end]:
                    if (note.start + note.duration) > target_time + 1e-5:
                        target_notes.append(note)
        
        if not target_notes:
            return
            
        self.midi.push_undo()
        from .midi import Note
        
        for note in target_notes:
            original_duration = note.duration
            note.duration = target_time - note.start
            
            new_note = Note(
                start=target_time,
                duration=original_duration - note.duration,
                pitch=note.pitch,
                velocity=note.velocity,
                channel=getattr(note, "channel", 0)
            )
            
            track_idx = self._note_track_map.get(id(note))
            if track_idx is not None and 0 <= track_idx < len(self.midi.tracks):
                self.midi.tracks[track_idx].notes.append(new_note)
        
        self.midi.sort()
        self.midi._bump()
        self.update()

    def update_timeline(self):
        self.audio_duration = max(
            self.audio.timeline_duration(),
            300.0 if self.audio.y is None else 1.0
        )

    def note_duration(self, start_time=None):
        bpm = self.midi.tempo_at(
            start_time
            if start_time is not None
            else 0.0
        )

        beats = (
            self.placement_beats
            if self.placement_beats is not None
            else self.note_length
        )

        return (
            beats *
            (60.0 / bpm)
        )

    def snap_play_position(
        self,
        value
    ):
        grid = self.note_length

        beat = self.midi.time_to_beat(
            max(
                0.0,
                value
            )
        )

        snapped = round(
            beat /
            grid
        ) * grid

        return max(
            0.0,
            self.midi.beat_to_time(
                snapped
            )
        )

    def scrub(self, x):
        fraction = max(
            0.0,
            min(
                1.0,
                x / max(1.0, self.width())
            )
        )

        time = self.snap_play_position(
            fraction *
            self.audio_duration
        )

        self.audio.position = time

        self.set_play_position(time)

        self.follow_play_position(
            force=True
        )

        self.update()

    def set_track_filter(
        self,
        track_index
    ):
        self.midi.set_filter_track(
            track_index
        )

        self.selected_notes = []
        self.selection_rect = None

        self.audio.invalidate_midi_cache()
        self.update()

    def follow_play_position(
        self,
        force=False
    ):
        if not (self.audio.playing or force):
            return
            
        if getattr(self, "panning", False) and not force:
            return

        visible_time = (self.width() - self.left_width) * self.seconds_per_pixel
        
        current_left_time = self.scroll_x
        current_right_time = self.scroll_x + visible_time
        
        if self.play_position > current_right_time:
            self.scroll_x = max(0.0, self.play_position - visible_time * 0.05)
            self._has_paged = True
            self.update()
        elif self.play_position < current_left_time:
            self.scroll_x = max(0.0, self.play_position - visible_time * 0.05)
            self._has_paged = True
            self.update()

    def time_to_x(
        self,
        time
    ):
        return (
            self.left_width +
            (
                time -
                self.scroll_x
            ) /
            self.seconds_per_pixel
        )

    def x_to_time(
        self,
        x
    ):
        return (
            self.scroll_x +
            (
                x -
                self.left_width
            ) *
            self.seconds_per_pixel
        )

    def pitch_to_y(
        self,
        pitch
    ):
        return (
            self.top_height +
            (
                self.max_pitch -
                pitch
            ) *
            self.note_height -
            self.scroll_y
        )

    def y_to_pitch(
        self,
        y
    ):
        pitch = (
            self.max_pitch -
            int(
                (
                    y -
                    self.top_height +
                    self.scroll_y
                ) /
                self.note_height
            )
        )
        return max(
            self.min_pitch,
            min(
                self.max_pitch,
                pitch
            )
        )

    def _get_snap_grid(self):
        if not self.selected_notes:
            return self.note_length
        min_beat = None
        for n in self.selected_notes:
            b1 = self.midi.time_to_beat(n.start)
            b2 = self.midi.time_to_beat(n.start + n.duration)
            beat_len = b2 - b1
            if min_beat is None or beat_len < min_beat:
                min_beat = beat_len
        if min_beat is None or min_beat <= 0:
            return self.note_length
        return min_beat

    def snap_time(
        self,
        value,
        grid=None,
        mode="floor"
    ):
        import math
        if grid is None:
            grid = self.note_length

        t = max(
            0.0,
            value
        )

        # マウス座標はピクセル単位で、グリッド線は int() で切り捨てて
        # 描画されるため、座標が理想のグリッド時刻より半ピクセル以下
        # 手前になることがある。さらに beat<->time の往復でも浮動小数点
        # 誤差が生じる。このまま floor/ceil すると、コピペの貼り付け開始
        # 位置・ノーツ追加・ゲート(右端)変更・範囲選択などがごくまれに
        # 1グリッド分ずれる原因になるため、半ピクセル分の手前誤差は
        # 意図した位置とみなして吸収する。
        bpm = self.midi.tempo_at(t)

        grid_seconds = (
            grid *
            (60.0 / bpm)
            if grid > 0
            else 0.0
        )

        if grid_seconds > 0:
            tolerance = min(
                0.5 * self.seconds_per_pixel,
                0.25 * grid_seconds
            )

            # ピクセル許容幅が極端に小さいズームでも
            # 浮動小数点誤差は吸収できるように下限を設ける
            tolerance = max(tolerance, 1e-6)
        else:
            tolerance = 1e-6

        tol_units = (
            tolerance *
            bpm /
            60.0 /
            grid
        )

        beat = self.midi.time_to_beat(t)

        units = beat / grid

        if mode == "floor":
            snapped = math.floor(units + tol_units) * grid
        elif mode == "ceil":
            snapped = math.ceil(units - tol_units) * grid
        else:
            snapped = round(units) * grid

        return max(
            0.0,
            self.midi.beat_to_time(
                snapped
            )
        )

    def time_signature_start_beat(
        self,
        time
    ):
        """Map a time-signature change onto the same beat grid we draw."""
        return int(
            round(
                self.midi.time_to_beat(
                    time
                )
            )
        )

    def note_grid_beats(
        self,
        note
    ):
        bpm = self.midi.tempo_at(
            note.start
        )

        beats = (
            note.duration *
            (bpm / 60.0)
        )

        return max(
            0.0625,
            beats
        )

    def toggle_play(self):
        if self.audio.playing:
            if getattr(self, "return_to_start_on_stop", True):
                self.stop()
            else:
                self.pause()
        else:
            self._pre_play_scroll = self.scroll_x
            self._is_following_center = False
            self._has_paged = False
            self.play()

    def play(self):
        self._pre_play_scroll = self.scroll_x
        self._is_following_center = False
        self._has_paged = False

        if not self.audio.playing:
            # Spaceで再生を開始した時点の位置を覚えておき、
            # タップ計測の確定時にこの位置へテンポを追加する。
            # 再生中のシークで書き換わる _start_position とは
            # 別に保持する。
            self._tap_play_start = max(
                0.0,
                float(getattr(self.audio, "position", 0.0))
            )

            self.audio.play()

    def pause(self):
        if self.tap_mode:
            # 停止した時点でタップ計測を確定する
            self.finish_tap_tempo(True)

        self.audio.pause()

        self.set_play_position(
            self.audio.position
        )

        self.update()

    def stop(self):
        if self.tap_mode:
            # 停止した時点でタップ計測を確定する
            self.finish_tap_tempo(True)

        self.audio.stop()

        self.set_play_position(
            self.audio.position
        )

        if getattr(self, "return_to_start_on_stop", True):
            visible_time = (self.width() - self.left_width) * self.seconds_per_pixel
            current_left_time = self.scroll_x
            current_right_time = self.scroll_x + visible_time
            
            if not (current_left_time <= self.play_position <= current_right_time):
                self.scroll_x = getattr(self, "_pre_play_scroll", self.scroll_x)

        self.update()

    # ------------------------------------------------------------------
    # 手動テンポ計測(タップテンポ)
    # ------------------------------------------------------------------
    def tap_tempo_trigger(self):
        """Shift+Space: 再生中のみ計測モードを開始し、
        起動中ならタップを記録する。停止中は受け付けない。"""
        if not self.audio.playing:
            return False

        if not self.tap_mode:
            if self.audio.y is None:
                return False

            self.tap_mode = True
            self._tap_fit = None
            # 追加位置は play() で記録済み(再生開始時の位置)。
            # 念のため未記録の場合だけ現在値から補完する。
            if self._tap_play_start is None:
                self._tap_play_start = max(
                    0.0,
                    float(getattr(self.audio, "_start_position", 0.0))
                )
            self._tap_disp_bpm = max(
                20.0,
                self.midi.bpm
            )
            self.tap_engine.reset()
            self._tap_anim_timer.start(16)

        self._register_tap()
        return True

    def _current_tap_position(self, now):
        """タップ瞬間のオーディオ位置(秒)を取得する。

        内部時計から即時推定し、UIタイマーの遅れを排除する。
        """
        audio = self.audio

        estimate = (
            audio._start_position +
            max(
                0.0,
                now - audio._started_at - audio._latency
            )
        )

        return max(
            0.0,
            min(audio.max_position(), estimate)
        )

    def _register_tap(self):
        now = time.perf_counter()
        position = self._current_tap_position(now)

        self.tap_engine.split_on_jump(
            now,
            position,
            self.audio.playing
        )

        if self.tap_engine.add_tap(now, position):
            self._tap_fit = self.tap_engine.fit()

    def _tap_tick(self):
        fit = self._tap_fit

        if fit is not None:
            # 新しい推定へ滑らかに収束させ、タップのたびに
            # グリッドがリアルタイムに動いて見えるようにする
            k = 1.0 - math.exp(-0.016 / 0.07)
            self._tap_disp_bpm += (
                fit["bpm"] - self._tap_disp_bpm
            ) * k
            self._tap_disp_phi += (
                fit["phi_audio"] - self._tap_disp_phi
            ) * k

        self.update()

    def finish_tap_tempo(self, commit):
        if not self.tap_mode:
            return

        self.tap_mode = False
        self._tap_anim_timer.stop()

        fit = self._tap_fit

        if (
            commit and
            fit is not None and
            fit["n"] >= MIN_TAPS_FOR_APPLY
        ):
            # 内部では小数点以下も正確に計算し、確定時にのみ
            # 最も近い整数BPMへ丸める。位相は丸えたBPMに対し
            # 打鍵位置へ最良一致させる。
            bpm_value = float(math.floor(fit["bpm"] + 0.5))
            phi = self.tap_engine.fit_phase(bpm_value)

            if phi is None:
                phi = fit["phi_audio"]

            # 再生開始位置へ計測結果を追加する
            self.tap_tempo_applied.emit(
                bpm_value,
                phi,
                self._tap_play_start
                if self._tap_play_start is not None
                else 0.0
            )

        self._tap_fit = None
        self.update()

    # ------------------------------------------------------------------
    # 歌詞入力モード
    # ------------------------------------------------------------------
    def _start_range_selection(self, x, y, lane_top):
        """右ドラッグによる手動範囲選択を開始する。"""
        self.selection_mode = True
        self.selection_rect = None

        if y >= lane_top + self.velocity_lane_height:
            self.selection_start = (
                self.snap_time(
                    self.x_to_time(x),
                    mode="floor"
                ),
                self.min_pitch
            )
            self.selection_end = self.selection_start
            self.selection_in_lane = True
            self.selection_in_pedal = True
        else:
            self.selection_start = (
                self.snap_time(
                    self.x_to_time(x),
                    mode="floor"
                ),
                self.y_to_pitch(y)
            )
            self.selection_end = self.selection_start
            self.selection_in_lane = False
            self.selection_in_pedal = False

        self.audio.seek(self.selection_start[0])
        self.set_play_position(self.selection_start[0])
        self.update()

    def set_lyric_mode(self, enabled):
        if not ENABLE_LYRICS:
            return

        enabled = bool(enabled)

        if self.lyric_mode == enabled:
            return

        self.lyric_mode = enabled
        self.unsetCursor()
        self.update()

    def toggle_lyric_mode(self):
        self.set_lyric_mode(not self.lyric_mode)
        return self.lyric_mode

    def _track_index_of(self, note):
        for i, track in enumerate(self.midi.tracks):
            for n in track.notes:
                if n is note:
                    return i

        return None

    def _edit_lyric_chain(self, note):
        """ノートの歌詞を入力する。

        クリックしたノートが選択範囲に含まれる場合は、
        選択されたノーツ(同じトラック内)を時系列順に
        選択範囲の先頭から順番に入力していく。
        選択範囲に無い単一ノートの場合はその1つだけ入力する。
        OKで次へ進み、キャンセル/Escで終了する。
        歌詞はプロジェクト保存時にノーツごとに記録される。
        """
        track_index = self._track_index_of(note)

        if track_index is None:
            return

        sel_ids = {
            id(n)
            for n in self.selected_notes
        }

        selected_in_track = [
            n
            for n in self.midi.tracks[track_index].notes
            if id(n) in sel_ids
        ]

        if (
            id(note) in sel_ids and
            len(selected_in_track) >= 2
        ):
            # 選択されたノーツだけを時系列順に処理する
            # (クリック位置に関係なく選択範囲の先頭から開始)
            ordered = sorted(
                selected_in_track,
                key=lambda n: n.start
            )

            idx = 0
        else:
            # 選択範囲に無い単一ノートはその1つだけ入力する
            ordered = [note]
            idx = 0

        undo_pushed = False

        while 0 <= idx < len(ordered):
            target = ordered[idx]

            text, ok = QInputDialog.getText(
                self,
                tr("歌詞の入力", "Enter Lyrics"),
                tr(f"歌詞 ({idx + 1}/{len(ordered)}):", f"Lyrics ({idx + 1}/{len(ordered)}):"),
                text=getattr(target, "lyric", "")
            )

            if not ok:
                break

            if getattr(target, "lyric", "") != text:
                if not undo_pushed:
                    self.midi.push_undo()
                    undo_pushed = True

                target.lyric = text
                self.midi._bump()

            idx += 1

        if undo_pushed:
            if self.audio.playing:
                self.audio.invalidate_midi_cache()

        self.update()

    def keyPressEvent(
        self,
        event
    ):
        key = event.key()
        modifiers = event.modifiers()

        if (
            key == Qt.Key_Escape and
            not event.isAutoRepeat() and
            self.lyric_mode
        ):
            # Escで歌詞入力モードを終了する
            self.set_lyric_mode(False)
            event.accept()
            return

        if key == Qt.Key_Space:
            self.toggle_play()
            event.accept()
            return

        if event.matches(
            QKeySequence.StandardKey.Copy
        ):
            self.copy_selected()
            event.accept()
            return

        if event.matches(
            QKeySequence.StandardKey.Cut
        ):
            self.cut_selected()
            event.accept()
            return

        if event.matches(
            QKeySequence.StandardKey.Paste
        ):
            self.paste_notes()
            event.accept()
            return

        if event.key() in (
            Qt.Key_Delete,
            Qt.Key_Backspace
        ):
            self.delete_selected()
            event.accept()
            return

        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            if self.midi.filter_track is not None:
                track_index = self.midi.filter_track
                self.selected_notes = list(self.midi.tracks[track_index].notes)
            else:
                self.selected_notes = []
                for track in self.midi.tracks:
                    self.selected_notes.extend(track.notes)
            self.selection_rect = None
            self.update()
            event.accept()
            return

        if event.key() in (
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down
        ):
            if self.selected_notes:
                self.nudge_selected(
                    event
                )
                event.accept()
                return

        super().keyPressEvent(
            event
        )

    def keyReleaseEvent(
        self,
        event
    ):
        if event.key() in (
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down
        ):
            self._nudge_undo_pushed = False
            self._previewed_pitch = None

        super().keyReleaseEvent(
            event
        )

    def nudge_selected(
        self,
        event
    ):
        if self.audio.playing:
            return

        shift = bool(
            event.modifiers() &
            Qt.ShiftModifier
        )

        if event.key() == Qt.Key_Up:
            d_pitch = (
                12 if shift else 1
            )
            d_beats = 0.0
        elif event.key() == Qt.Key_Down:
            d_pitch = (
                -12 if shift else -1
            )
            d_beats = 0.0
        elif event.key() == Qt.Key_Right:
            d_pitch = 0
            d_beats = (
                4 if shift else 1
            )
        else:
            d_pitch = 0
            d_beats = (
                -4 if shift else -1
            )

        if not self._nudge_undo_pushed:
            self.midi.push_undo()
            self._nudge_undo_pushed = True

        grid = self.note_length

        for note in self.selected_notes:
            if d_pitch:
                note.pitch = max(
                    self.min_pitch,
                    min(
                        self.max_pitch,
                        note.pitch +
                        d_pitch
                    )
                )

            if d_beats:
                beat = (
                    self.midi.time_to_beat(
                        note.start
                    )
                )

                new_beat = (
                    round(
                        beat /
                        grid
                    ) +
                    d_beats
                ) * grid

                note.start = max(
                    0.0,
                    min(
                        self.audio_duration,
                        self.midi.beat_to_time(
                            max(
                                0.0,
                                new_beat
                            )
                        )
                    )
                )

        # 重複チェック
        is_duplicate = False
        track_index = self.midi.active_track()
        track = self.midi.tracks[track_index]

        pitch_groups = {}
        for other in track.notes:
            pitch_groups.setdefault(
                other.pitch, []
            ).append(other)
        
        for n in self.selected_notes:
            new_end = n.start + n.duration
            for other in pitch_groups.get(n.pitch, []):
                if other is n:
                    continue
                other_end = other.start + other.duration
                if not (new_end <= other.start + 1e-6 or n.start >= other_end - 1e-6):
                    is_duplicate = True
                    break
            if is_duplicate:
                break
                
        if is_duplicate:
            self.midi.undo()
            self.selected_notes = []
            self.selection_rect = None
            self.update()
            return

        if d_pitch and self.selected_notes:
            self.preview_pitch(
                self.selected_notes[-1].pitch
            )

        self.midi._bump()
        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.update()

    def wheelEvent(
        self,
        event
    ):
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.angleDelta().x()

        if (
            event.modifiers() & Qt.ControlModifier and
            event.position().x() < self.left_width
        ):
            y = event.position().y()

            old_height = self.note_height

            if delta > 0:
                new_height = min(
                    40,
                    old_height * 1.1
                )
            else:
                new_height = max(
                    8,
                    old_height * 0.9
                )

            factor = (
                new_height /
                old_height
            )

            self.note_height = new_height

            self.scroll_y = (
                (
                    y -
                    self.top_height +
                    self.scroll_y
                ) *
                factor -
                (
                    y -
                    self.top_height
                )
            )

            max_scroll = max(
                0,
                (
                    self.max_pitch -
                    self.min_pitch +
                    1
                ) *
                self.note_height -
                (
                    self.height() -
                    self.top_height -
                    self.bottom_height
                )
            )

            self.scroll_y = max(
                0,
                min(
                    self.scroll_y,
                    max_scroll
                )
            )

        elif event.modifiers() & Qt.AltModifier:
            if delta > 0:
                self.track_switch_requested.emit(-1)
            elif delta < 0:
                self.track_switch_requested.emit(1)
            event.accept()
            return

        elif event.modifiers() & Qt.ControlModifier:
            mouse_time = self.x_to_time(
                event.position().x()
            )

            if delta > 0:
                self.seconds_per_pixel *= 0.8
            else:
                self.seconds_per_pixel *= 1.25

            self.seconds_per_pixel = max(
                0.0005,
                min(
                    0.08,
                    self.seconds_per_pixel
                )
            )

            self.scroll_x = (
                mouse_time -
                (
                    event.position().x() -
                    self.left_width
                ) *
                self.seconds_per_pixel
            )

            self.scroll_x = max(
                0.0,
                self.scroll_x
            )

        elif event.modifiers() & Qt.ShiftModifier:
            self.scroll_x -= (
                delta *
                self.seconds_per_pixel *
                3
            )

            self.scroll_x = max(
                0.0,
                self.scroll_x
            )

        else:
            self.scroll_y -= (
                delta *
                0.5
            )

            max_scroll = max(
                0,
                (
                    self.max_pitch -
                    self.min_pitch +
                    1
                ) *
                self.note_height -
                (
                    self.height() -
                    self.top_height -
                    self.bottom_height
                )
            )

            self.scroll_y = max(
                0,
                min(
                    self.scroll_y,
                    max_scroll
                )
            )

        self.update()

    def mousePressEvent(
        self,
        event
    ):
        x = event.position().x()
        y = event.position().y()

        if event.button() == Qt.LeftButton and (
            event.modifiers() & Qt.AltModifier
        ):
            if not hasattr(self.midi, "extra_state"):
                self.midi.extra_state = {}
            self.midi.extra_state["audio_offset"] = self.audio.offset
            self.midi.push_undo()
            
            self.offset_dragging = True
            self.offset_drag_start = QPointF(
                x,
                y
            )
            self.offset_drag_value = (
                self.audio.offset
            )
            self.setCursor(
                Qt.SizeHorCursor
            )
            return

        if event.button() == Qt.MiddleButton:
            if event.modifiers() & Qt.AltModifier:
                if not hasattr(self.midi, "extra_state"):
                    self.midi.extra_state = {}
                self.midi.extra_state["audio_offset"] = self.audio.offset
                self.midi.push_undo()
                
                self.offset_dragging = True
                self.offset_drag_start = QPointF(x, y)
                self.offset_drag_value = self.audio.offset
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.panning = True
                self.pan_start_x = x
                self.pan_start_scroll = self.scroll_x
                self.setCursor(Qt.ClosedHandCursor)
            return

        lane_top = (
            self.height() -
            self.bottom_height
        )

        if self.lyric_mode:
            # 歌詞入力モード中:
            # 左クリック = 歌詞入力(選択範囲内なら連続入力)
            # 右ドラッグ = 手動範囲選択(ノーツの移動等は行わない)
            if event.button() == Qt.LeftButton:
                if (
                    y >= self.top_height and
                    y < lane_top and
                    x >= self.left_width
                ):
                    if self.selection_mode:
                        # 範囲選択中の左クリックは確定として扱う
                        self.selection_end = (
                            self.snap_time(
                                self.x_to_time(x),
                                mode="floor"
                            ),
                            self.y_to_pitch(y)
                        )

                        self.finish_selection()
                        event.accept()
                        return

                    note = self.note_at(x, y)

                    if note is not None:
                        self._edit_lyric_chain(note)

                event.accept()
                return

            if event.button() == Qt.RightButton:
                if not self.audio.playing:
                    self._start_range_selection(x, y, lane_top)

                event.accept()
                return

            event.accept()
            return

        if event.button() == Qt.RightButton:
            if self.audio.playing:
                return

            note = self.note_at(x, y)

            if note is None and y >= lane_top and y < lane_top + self.velocity_lane_height:
                self._vel_press(event, x, y)
                return

            pedal_hit = None
            if note is None and y >= lane_top + self.velocity_lane_height and y < lane_top + self.velocity_lane_height + self.pedal_lane_height:
                events = self.midi.tracks[self.midi.active_track()].pedals
                pedal_hit = self._pedal_event_at(events, x)

            if pedal_hit is not None:
                self.last_selection_time_range = (pedal_hit.time - 0.001, pedal_hit.time + 0.001)
                self.last_selection_in_pedal = True
                self.selected_notes = []
                self.selection_rect = None
                self.update()

                menu = QMenu(self)
                copy_action = menu.addAction(tr("コピー", "Copy"))
                cut_action = menu.addAction(tr("切り取り", "Cut"))
                delete_action = menu.addAction(tr("削除", "Delete"))

                action = menu.exec(event.globalPosition().toPoint())

                if action:
                    if action == copy_action:
                        self.copy_selected()
                    elif action == cut_action:
                        self.cut_selected()
                    elif action == delete_action:
                        self.delete_selected()

                    self._clear_selection()
                    self.update()
                return

            if note is not None:
                if note not in self.selected_notes:
                    self.selected_notes = [note]
                    self.selection_rect = None
                    self.update()

                menu = QMenu(self)

                sel_notes = self.selected_notes

                vel_action = None
                if len(sel_notes) >= 1:
                    vel_action = menu.addAction(tr("ベロシティを設定", "Set Velocity"))

                merge_action = None
                if len(sel_notes) >= 2:
                    merge_action = menu.addAction(tr("ノーツを結合", "Merge Notes"))

                menu.addSeparator()
                copy_action = menu.addAction(tr("コピー", "Copy"))
                cut_action = menu.addAction(tr("切り取り", "Cut"))
                delete_action = menu.addAction(tr("削除", "Delete"))

                octave_up_action = None
                octave_down_action = None
                if len(sel_notes) >= 1:
                    menu.addSeparator()
                    octave_up_action = menu.addAction(tr("オクターブ上", "Octave Up"))
                    octave_down_action = menu.addAction(tr("オクターブ下", "Octave Down"))

                action = menu.exec(event.globalPosition().toPoint())

                if action:
                    if vel_action and action == vel_action:
                        self._set_velocity_dialog(sel_notes[0])
                    elif octave_up_action and action == octave_up_action:
                        self.transpose_selected(12)
                    elif octave_down_action and action == octave_down_action:
                        self.transpose_selected(-12)
                    elif merge_action and action == merge_action:
                        self._merge_notes(sel_notes)
                    elif action == copy_action:
                        self.copy_selected()
                    elif action == cut_action:
                        self.cut_selected()
                    elif action == delete_action:
                        self.delete_selected()

                    self.selected_notes = []
                    self.selection_rect = None
                    self.last_selection_time_range = None
                    self.last_selection_in_pedal = False
                    self.update()

                return

            if self.selection_rect is not None:
                rt1, rt2, rp1, rp2 = self.selection_rect[:4]
                r_in_lane = self.selection_rect[4] if len(self.selection_rect) > 4 else False
                r_in_pedal = self.selection_rect[5] if len(self.selection_rect) > 5 else False
                click_time = self.x_to_time(x)
                if r_in_lane:
                    if r_in_pedal:
                        in_rect = (rt1 <= click_time <= rt2) and (y >= lane_top + self.velocity_lane_height)
                    else:
                        in_rect = (rt1 <= click_time <= rt2) and (lane_top <= y < lane_top + self.velocity_lane_height)
                else:
                    click_pitch = self.y_to_pitch(y)
                    in_rect = (
                        rt1 <= click_time <= rt2 and
                        rp1 <= click_pitch <= rp2 and
                        y < lane_top
                    )

                if in_rect:
                    menu = QMenu(self)

                    sel_notes = self.selected_notes

                    vel_action = None
                    if len(sel_notes) >= 1:
                        vel_action = menu.addAction(tr("ベロシティを設定", "Set Velocity"))

                    merge_action = None
                    if len(sel_notes) >= 2:
                        merge_action = menu.addAction(tr("ノーツを結合", "Merge Notes"))

                    menu.addSeparator()
                    copy_action = menu.addAction(tr("コピー", "Copy"))
                    cut_action = menu.addAction(tr("切り取り", "Cut"))
                    delete_action = menu.addAction(tr("削除", "Delete"))

                    octave_up_action = None
                    octave_down_action = None
                    if len(sel_notes) >= 1:
                        menu.addSeparator()
                        octave_up_action = menu.addAction(tr("オクターブ上", "Octave Up"))
                        octave_down_action = menu.addAction(tr("オクターブ下", "Octave Down"))

                    action = menu.exec(event.globalPosition().toPoint())

                    if action:
                        if vel_action and action == vel_action:
                            self._set_velocity_dialog(sel_notes[0])
                        elif octave_up_action and action == octave_up_action:
                            self.transpose_selected(12)
                        elif octave_down_action and action == octave_down_action:
                            self.transpose_selected(-12)
                        elif merge_action and action == merge_action:
                            self._merge_notes(sel_notes)
                        elif action == copy_action:
                            self.copy_selected()
                        elif action == cut_action:
                            self.cut_selected()
                        elif action == delete_action:
                            self.delete_selected()

                        self.selected_notes = []
                        self.selection_rect = None
                        self.last_selection_time_range = None
                        self.last_selection_in_pedal = False
                        self.update()

                    return
                else:
                    self.selected_notes = []
                    self.selection_rect = None
                    self.last_selection_time_range = None
                    self.last_selection_in_pedal = False
                    self.update()


            self._start_range_selection(x, y, lane_top)
            return
        if (
            y >= lane_top and
            event.button() == Qt.LeftButton
        ):
            if self.audio.playing:
                return

            if (
                y <
                lane_top +
                self.velocity_lane_height
            ):
                self._vel_press(
                    event,
                    x,
                    y
                )
                return

            if (
                y <
                lane_top +
                self.velocity_lane_height +
                self.pedal_lane_height
            ):
                self._pedal_press(
                    event,
                    x,
                    y
                )
                return

        if event.button() != Qt.LeftButton:
            return

        if self.selection_mode:
            self.selection_end = (
                self.snap_time(
                    self.x_to_time(x),
                    mode="floor"
                ),
                self.y_to_pitch(y)
            )

            self.finish_selection()
            return

        if y >= self.height() - self.scrub_height:
            self.scrub_resume = self.audio.playing

            if self.audio.playing:
                self.audio.pause()

            self.scrubbing = True
            self.scrub(x)
            return

        if y < self.top_height:
            time = self.snap_play_position(
                self.x_to_time(x)
            )

            min_time = self.x_to_time(
                self.left_width
            )
            max_time = self.x_to_time(
                self.width()
            )

            time = max(
                min_time,
                min(time, max_time)
            )

            self.audio.seek(time)

            self.set_play_position(time)

            self.update()

            return

        if x < self.left_width:
            if y < self.pitch_to_y(self.max_pitch) or y >= (
                self.pitch_to_y(self.min_pitch) +
                self.note_height
            ):
                return

            pitch = self.y_to_pitch(y)

            self.preview_pitch(
                pitch
            )

            return


        in_rect, on_right_edge, _ = self._click_in_selection_rect(x, y)
        if in_rect and self.selected_notes and not getattr(self, "selection_in_lane", False) and not getattr(self, "selection_in_pedal", False):
            if self.audio.playing and not on_right_edge:
                return

            # Ctrlドラッグで選択範囲を複製する
            ctrl_duplicate = (
                not on_right_edge and
                bool(event.modifiers() & Qt.ControlModifier)
            )

            self.midi.push_undo()
            self.drag_start = QPointF(x, y)
            self.press_moved = False
            self.selection_rect = None
            if on_right_edge:
                self.drag_mode = "resize"
            else:
                self.drag_mode = "move"
                if ctrl_duplicate:
                    clones = self._duplicate_selected_notes(
                        self.selected_notes
                    )
                    if clones:
                        self.selected_notes = clones
            self.drag_original_notes = [
                (n, n.start, n.pitch, n.duration)
                for n in self.selected_notes
            ]
            note_under_cursor = self.note_at(x, y)
            first = note_under_cursor if (note_under_cursor and note_under_cursor in self.selected_notes) else self.selected_notes[0]
            
            self.drag_note = first
            self.drag_track_index = self._note_track_index(first)
            self.drag_original = (first.start, first.duration, first.pitch)
            self.update()
            return

        note = self.note_at(
            x,
            y
        )

        if note:
            note_end_x = self.time_to_x(
                note.start +
                note.duration
            )

            is_resize = abs(
                x -
                note_end_x
            ) <= 8

            if self.audio.playing and not is_resize:
                return

            # Ctrlドラッグでノーツを複製する
            ctrl_duplicate = (
                not is_resize and
                bool(event.modifiers() & Qt.ControlModifier)
            )

            clicked_note = note

            group_drag = (
                note in self.selected_notes and
                len(self.selected_notes) > 1
            )

            self.midi.push_undo()

            if ctrl_duplicate:
                if group_drag:
                    src = list(self.selected_notes)
                else:
                    src = [clicked_note]

                clones = self._duplicate_selected_notes(src)
                if clones:
                    idx = next(
                        i for i, s in enumerate(src)
                        if s is clicked_note
                    )
                    note = clones[idx]

                    if group_drag:
                        self.selected_notes = clones

            self.drag_note = note

            self.drag_track_index = (
                self._note_track_index(note)
            )

            self.drag_start = QPointF(
                x,
                y
            )

            self.press_moved = False

            self.drag_original = (
                note.start,
                note.duration,
                note.pitch
            )

            if is_resize:
                self.drag_mode = "resize"
            else:
                self.drag_mode = "move"

            if (
                note in self.selected_notes and
                len(self.selected_notes) > 1
            ):
                self.drag_original_notes = [
                    (
                        n,
                        n.start,
                        n.pitch,
                        n.duration
                    )
                    for n in self.selected_notes
                ]
            else:
                self.drag_original_notes = None

            if note not in self.selected_notes:
                self._clear_selection()
                self.selected_notes = [note]

            self.update()

            return

        if self.midi.filter_track is None or self.audio.playing:
            return

        grid_top = self.pitch_to_y(self.max_pitch)
        grid_bottom = (
            self.pitch_to_y(self.min_pitch) +
            self.note_height
        )

        if y < grid_top or y >= grid_bottom:
            return

        start = self.snap_time(
            self.x_to_time(x),
            mode="floor"
        )

        pitch = self.y_to_pitch(
            y
        )

        self.midi.push_undo()
        self._clear_selection()

        note = self.midi.add_note(
            start,
            self.note_duration(start),
            pitch
        )

        if note is None:
            self.midi.undo() # 重複で追加できなかったのでundoを消す
            return

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.drag_note = note
        self.drag_track_index = (
            self.midi.active_track()
        )
        self.drag_mode = "move"
        self.drag_start = QPointF(
            x,
            y
        )
        self.drag_original = (
            note.start,
            note.duration,
            note.pitch
        )
        self.press_moved = False

        self.preview_pitch(
            pitch
        )

        self.update()

    def mouseDoubleClickEvent(
        self,
        event
    ):
        if event.button() == Qt.LeftButton:
            x = event.position().x()
            y = event.position().y()

            if self.lyric_mode:
                # 歌詞入力モード中はダブルクリック削除を無効化する
                event.accept()
                return

            if y < self.top_height:
                if self.edit_marker_at(x, y):
                    event.accept()
                    return

                super().mouseDoubleClickEvent(event)
                return

            lane_top = (
                self.height() -
                self.bottom_height
            )

            if self.audio.playing and y >= self.top_height and y < self.height() - self.scrub_height:
                event.accept()
                return

            if y >= lane_top:
                if (
                    y <
                    lane_top +
                    self.velocity_lane_height
                ):
                    self._vel_double_click(
                        x,
                        y
                    )
                elif (
                    y <
                    lane_top +
                    self.velocity_lane_height +
                    self.pedal_lane_height
                ):
                    self._pedal_double_click(
                        x,
                        y
                    )

                event.accept()
                return

            if (
                y >= self.height() - self.scrub_height or
                x < self.left_width
            ):
                super().mouseDoubleClickEvent(event)
                return

            note = self.note_at(x, y)

            if note:
                self.midi.push_undo()

                self.midi.remove_note(note)

                self.selected_notes = [
                    n for n in self.selected_notes
                    if n != note
                ]
                self.selection_rect = None

                self.update()
                event.accept()
                return

        super().mouseDoubleClickEvent(event)

    def edit_marker_at(self, x, y):
        tempo_hit = None
        timesig_hit = None

        if self.top_height - 26 <= y <= self.top_height - 12:
            tempo_hit = self._hit_tempo_marker(x)

        if self.top_height - 16 <= y <= self.top_height - 1:
            timesig_hit = self._hit_timesig_marker(x)

        if tempo_hit is not None:
            self._edit_tempo_marker(tempo_hit)
            return True

        if timesig_hit is not None:
            self._edit_timesig_marker(timesig_hit)
            return True

        return False

    def _hit_tempo_marker(self, px):
        best = None
        best_d = None

        for t_sec, bpm in self.midi.tempos:
            x = self.time_to_x(t_sec)

            if x < self.left_width - 1:
                continue

            d = abs(px - (x + 3))

            if d > 44:
                continue

            if best_d is None or d < best_d:
                best = (t_sec, bpm)
                best_d = d

        return best

    def _hit_timesig_marker(self, px):
        best = None
        best_d = None

        for t_sec, num, den in self.midi.time_signatures:
            x = self.time_to_x(t_sec)

            if x < self.left_width - 1:
                continue

            d = abs(px - (x + 3))

            if d > 26:
                continue

            if best_d is None or d < best_d:
                best = (t_sec, num, den)
                best_d = d

        return best

    def _edit_tempo_marker(self, marker):
        t_sec, bpm = marker

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("テンポ変更", "Edit Tempo"))

        layout = QVBoxLayout(dlg)

        pos_label = QLabel(
            tr(f"位置 {t_sec:.2f}s", f"Position {t_sec:.2f}s")
        )
        layout.addWidget(pos_label)

        spin = QDoubleSpinBox()
        spin.setRange(20.0, 999.0)
        spin.setDecimals(0)
        spin.setSuffix(" BPM")
        spin.setValue(round(bpm))
        layout.addWidget(spin)

        del_check = QCheckBox(
            tr("このテンポを削除", "Delete This Tempo")
        )

        if len(self.midi.tempos) <= 1:
            del_check.setEnabled(False)
            del_check.setToolTip(
                tr("最後のテンポは削除できません", "The last tempo cannot be deleted")
            )

        layout.addWidget(del_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)

        # プレビュー用のテンポ変更をリアルタイムに反映するための一時退避
        self.midi.push_undo()

        def preview_tempo(val):
            if del_check.isChecked():
                return # 削除チェック時はプレビュー更新しない
            
            self.midi.undo()
            self.midi.push_undo()
            self.midi.add_tempo(t_sec, val)
            self.bpm = self.midi.bpm
            self.update_timeline()
            self.update()

        spin.valueChanged.connect(preview_tempo)

        def preview_delete(checked):
            self.midi.undo()
            self.midi.push_undo()
            if checked:
                self.midi.remove_tempo(t_sec)
            else:
                self.midi.add_tempo(t_sec, spin.value())
            self.bpm = self.midi.bpm
            self.update_timeline()
            self.update()
            
        del_check.toggled.connect(preview_delete)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.Accepted:
            self._after_marker_edit()
        else:
            self.midi.undo()
            self.update_timeline()
            self.update()

    def _edit_timesig_marker(self, marker):
        t_sec, num, den = marker

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("拍子変更", "Edit Time Signature"))

        layout = QVBoxLayout(dlg)

        pos_label = QLabel(
            tr(f"位置 {t_sec:.2f}s", f"Position {t_sec:.2f}s")
        )
        layout.addWidget(pos_label)

        row = QWidget(dlg)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        num_spin = QSpinBox()
        num_spin.setRange(1, 32)
        num_spin.setValue(max(1, int(num)))

        slash_label = QLabel("/")

        den_spin = QSpinBox()
        den_spin.setRange(1, 32)
        den_spin.setValue(max(1, int(den)))

        row_layout.addWidget(num_spin)
        row_layout.addWidget(slash_label)
        row_layout.addWidget(den_spin)
        layout.addWidget(row)

        del_check = QCheckBox(
            tr("この拍子を削除", "Delete This Time Signature")
        )

        if len(self.midi.time_signatures) <= 1:
            del_check.setEnabled(False)
            del_check.setToolTip(
                tr("最後の拍子は削除できません", "The last time signature cannot be deleted")
            )

        layout.addWidget(del_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        self.midi.push_undo()

        if del_check.isChecked():
            self.midi.remove_time_signature(t_sec)
        else:
            self.midi.add_time_signature(
                t_sec,
                num_spin.value(),
                den_spin.value()
            )

        self._after_marker_edit()

    def _after_marker_edit(self):
        self.bpm = self.midi.bpm
        self.update_timeline()
        self.update()
        self.marker_edited.emit()

    def auto_scroll(self):
        is_dragging = (
            self.selection_mode or
            self.drag_note is not None or
            self.vel_drag is not None or
            self.pedal_drag is not None
        )

        if not is_dragging:
            return

        from PySide6.QtGui import QCursor
        pos = self.mapFromGlobal(QCursor.pos())
        x = pos.x()
        y = pos.y()

        # ウィジェットの範囲外にあるかどうかの判定にマージンを使う
        # しかし、ウィジェット内で端に寄っている場合にもスクロールする
        margin = 30
        scroll_amount = 0.0

        if x < self.left_width + margin:
            scroll_amount = -20.0 * self.seconds_per_pixel
        elif x > self.width() - margin:
            scroll_amount = 20.0 * self.seconds_per_pixel

        if scroll_amount != 0.0:
            visible_time = (self.width() - self.left_width) * self.seconds_per_pixel
            max_scroll = max(0.0, self.audio_duration - visible_time)
            
            new_scroll = max(0.0, min(self.scroll_x + scroll_amount, max_scroll))
            
            if new_scroll != self.scroll_x:
                diff_scroll = new_scroll - self.scroll_x
                diff_px = diff_scroll / self.seconds_per_pixel
                self.scroll_x = new_scroll
                
                if self.selection_mode and self.selection_start:
                    lane_top = self.height() - self.bottom_height
                    self.selection_end = (
                        self.snap_time(
                            self.x_to_time(x),
                            mode="floor"
                        ),
                        self.y_to_pitch(min(y, lane_top - 1))
                    )
                elif self.drag_note is not None:
                    self.drag_start.setX(self.drag_start.x() - diff_px)
                    self._note_drag_move(x, y)
                elif self.vel_drag is not None:
                    self.vel_drag["x0"] -= diff_px
                    self._vel_drag_move(x, y)
                elif self.pedal_drag is not None:
                    if self.pedal_drag["mode"] == "erase":
                        self.pedal_drag["x0"] -= diff_px
                    self._pedal_drag_move(x, y)
                
                self.update()

    def mouseMoveEvent(
        self,
        event
    ):
        x = event.position().x()
        y = event.position().y()

        self.update_cursor(
            x,
            y
        )

        if self.vel_drag is not None:
            self._vel_drag_move(
                x,
                y
            )
            return

        if self.pedal_drag is not None:
            self._pedal_drag_move(
                x,
                y
            )
            return

        if self.scrubbing:
            self.scrub(x)
            return

        if self.panning:
            visible_time = (
                self.width() -
                self.left_width
            ) * self.seconds_per_pixel

            max_scroll = max(
                0.0,
                self.audio_duration -
                visible_time
            )

            self.scroll_x = max(
                0.0,
                min(
                    self.pan_start_scroll -
                    (
                        x -
                        self.pan_start_x
                    ) *
                    self.seconds_per_pixel,
                    max_scroll
                )
            )

            self.update()
            return

        if self.offset_dragging:
            self.audio.offset = (
                self.offset_drag_value +
                (
                    x -
                    self.offset_drag_start.x()
                ) *
                self.seconds_per_pixel
            )
            
            if not hasattr(self.midi, "extra_state"):
                self.midi.extra_state = {}
            self.midi.extra_state["audio_offset"] = self.audio.offset

            self.update()
            return

        if self.selection_mode:
            if self.selection_start:
                lane_top = self.height() - self.bottom_height
                self.selection_end = (
                    self.snap_time(
                        self.x_to_time(x),
                        mode="floor"
                    ),
                    # Clamp to the bottom-most visible row instead of
                    # min_pitch, so entering the lane area does not
                    # extend the selection past the visible grid.
                    self.y_to_pitch(min(y, lane_top - 1))
                )
                self.update()

            return

        if self.drag_note:
            self._note_drag_move(x, y)
            return

        self.update()

    def _note_drag_move(self, x, y):
        dx = (
            x -
            self.drag_start.x()
        )

        dy = (
            y -
            self.drag_start.y()
        )

        original_start = (
            self.drag_original[0]
        )

        original_duration = (
            self.drag_original[1]
        )

        original_pitch = (
            self.drag_original[2]
        )

        time_diff = 0.0
        pitch_diff = 0
        diff_duration = 0.0

        if self.drag_mode == "move":
            if (
                not self.press_moved and
                (
                    abs(dx) > 3 or
                    abs(dy) > 3
                )
            ):
                self.press_moved = True

            snapped_start_mouse = self.snap_time(
                self.x_to_time(
                    self.drag_start.x()
                )
            )
            snapped_current_mouse = self.snap_time(
                self.x_to_time(x)
            )

            new_start = (
                original_start +
                (
                    snapped_current_mouse -
                    snapped_start_mouse
                )
            )

            new_pitch = (
                original_pitch -
                round(
                    dy /
                    self.note_height
                )
            )

            if self.drag_original_notes:
                time_diff = new_start - original_start
                pitch_diff = new_pitch - original_pitch

                for n, o_start, o_pitch, o_duration in (
                    self.drag_original_notes
                ):
                    n.start = max(
                        0.0,
                        o_start + time_diff
                    )

                    n.pitch = max(
                        self.min_pitch,
                        min(
                            self.max_pitch,
                            o_pitch + pitch_diff
                        )
                    )
            else:
                self.drag_note.start = max(
                    0.0,
                    new_start
                )

                self.drag_note.pitch = max(
                    self.min_pitch,
                    min(
                        self.max_pitch,
                        new_pitch
                    )
                )

            if self._previewed_pitch != (
                self.drag_note.pitch
            ):
                self.preview_pitch(
                    self.drag_note.pitch
                )

        elif self.drag_mode == "resize":
            if self.drag_original_notes:
                # 複数ノーツの場合は「選択範囲の右端」を基準にする。
                # 単一ノーツ時と同じく、つかんだ端がそのままマウスに追従し、
                # 各ノーツは長さの変化量のみを受け取る
                original_end = max(
                    o_start + o_duration
                    for _, o_start, _, o_duration in
                    self.drag_original_notes
                )
            else:
                original_end = original_start + original_duration

            new_end = self.snap_time(
                self.x_to_time(x),
                grid=self.note_length
            )
            
            diff_duration = new_end - original_end
            min_beats = self.note_length
            
            if self.drag_original_notes:
                for n, o_start, o_pitch, o_duration in self.drag_original_notes:
                    bpm = self.midi.tempo_at(o_start)
                    min_duration = min_beats * (60.0 / bpm)
                    
                    n.duration = max(
                        min_duration,
                        o_duration + diff_duration
                    )
            else:
                bpm = self.midi.tempo_at(self.drag_note.start)
                min_duration = min_beats * (60.0 / bpm)
                
                self.drag_note.duration = max(
                    min_duration,
                    original_duration + diff_duration
                )

        self.update()

    def mouseReleaseEvent(
        self,
        event
    ):
        if event.button() == Qt.MiddleButton:
            if self.offset_dragging:
                self.offset_dragging = False
                self.offset_drag_start = None
            self.panning = False
            self.unsetCursor()

        if event.button() == Qt.RightButton:
            if self.selection_mode:
                self.finish_selection()

        if event.button() in (
            Qt.LeftButton,
            Qt.RightButton
        ):
            if self.vel_drag is not None:
                self._vel_drag_end()

            if self.pedal_drag is not None:
                self._pedal_drag_end()

        if event.button() == Qt.LeftButton:
            if self.scrubbing:
                self.scrubbing = False

                if self.scrub_resume:
                    self.scrub_resume = False
                    self.audio.play()

            if self.offset_dragging:
                self.offset_dragging = False
                self.offset_drag_start = None
                self.unsetCursor()
            elif (
                self.drag_note and
                self.drag_mode == "resize"
            ):
                bpm = self.midi.tempo_at(
                    self.drag_note.start
                )

                self.placement_beats = (
                    self.drag_note.duration *
                    (bpm / 60.0)
                )

        if self.drag_note is not None:
            is_duplicate = False
            self._ensure_note_cache()
            notes_to_check = self.selected_notes if self.drag_original_notes else [self.drag_note]
            exclude_set = set(id(n) for n in notes_to_check)
            
            for n in notes_to_check:
                n_track_idx = self._note_track_index(n)
                if n_track_idx is None:
                    continue
                new_end = n.start + n.duration
                if self._has_overlap_in_track_exclude(
                    n_track_idx,
                    exclude_set,
                    n.start,
                    new_end,
                    n.pitch
                ):
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                self.midi.undo()
                self._clear_selection()
                self.update()
            else:
                self.midi._bump()
                if self.audio.playing:
                    self.audio.invalidate_midi_cache()
            
            # 操作が終わったら選択状態を消す
            self._clear_selection()

        self.drag_note = None
        self.drag_mode = None
        self.drag_start = None
        self.drag_original = None
        self.drag_track_index = 0
        self.drag_original_notes = None
        self.press_moved = False
        self._previewed_pitch = None
        self._nudge_undo_pushed = False

        self._refresh_cursor_under_mouse()

    def _refresh_cursor_under_mouse(self):
        from PySide6.QtGui import QCursor
        pos = self.mapFromGlobal(QCursor.pos())
        self.update_cursor(
            pos.x(),
            pos.y()
        )

    def update_cursor(
        self,
        x,
        y
    ):
        if self.vel_drag is not None:
            self.setCursor(
                Qt.SizeVerCursor
            )
            return

        if self.pedal_drag is not None:
            self.setCursor(
                Qt.PointingHandCursor
            )
            return

        if self.panning:
            self.setCursor(
                Qt.ClosedHandCursor
            )
            return

        if self.offset_dragging:
            self.setCursor(
                Qt.SizeHorCursor
            )
            return

        if self.drag_note:
            if self.drag_mode == "resize":
                self.setCursor(
                    Qt.SizeHorCursor
                )
            else:
                self.setCursor(
                    Qt.SizeAllCursor
                )
            return

        if self.selection_mode:
            self.setCursor(
                Qt.CrossCursor
            )
            return

        if self.lyric_mode:
            self.setCursor(
                Qt.CrossCursor
            )
            return

        if (
            x < self.left_width or
            y < self.top_height
        ):
            self.unsetCursor()
            return

        lane_top = (
            self.height() -
            self.bottom_height
        )

        if y >= lane_top:
            if (
                y <
                lane_top +
                self.velocity_lane_height
            ):
                if (
                    self._velocity_bar_at(x)
                    is not None
                ):
                    self.setCursor(
                        Qt.SizeVerCursor
                    )
                else:
                    self.setCursor(
                        Qt.CrossCursor
                    )
            elif (
                y <
                lane_top +
                self.velocity_lane_height +
                self.pedal_lane_height
            ):
                if (
                    self._pedal_event_at(
                        self.midi.tracks[
                            self.midi.active_track()
                        ].pedals,
                        x
                    ) is not None
                ):
                    self.setCursor(
                        Qt.PointingHandCursor
                    )
                else:
                    self.setCursor(
                        Qt.CrossCursor
                    )
            return

        in_rect, on_right_edge, _ = self._click_in_selection_rect(x, y)
        if in_rect and self.selected_notes and not getattr(self, "selection_in_lane", False) and not getattr(self, "selection_in_pedal", False):
            if on_right_edge:
                self.setCursor(Qt.SizeHorCursor)
            elif self.audio.playing:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.SizeAllCursor)
            return

        note = self.note_at(
            x,
            y
        )

        if note:
            note_end_x = self.time_to_x(
                note.start +
                note.duration
            )

            if abs(
                x -
                note_end_x
            ) <= 8:
                self.setCursor(
                    Qt.SizeHorCursor
                )
            elif self.audio.playing:
                self.setCursor(
                    Qt.CrossCursor
                )
            else:
                self.setCursor(
                    Qt.SizeAllCursor
                )
        else:
            self.setCursor(
                Qt.CrossCursor
            )

    def leaveEvent(
        self,
        event
    ):
        self.unsetCursor()

        super().leaveEvent(event)

    def _clear_selection(self):
        self.selected_notes = []
        self.selection_rect = None
        self.last_selection_time_range = None
        self.last_selection_in_pedal = False

    def finish_selection(self):
        if not self.selection_start or not self.selection_end:
            return

        t1 = min(
            self.selection_start[0],
            self.selection_end[0]
        )

        t2 = max(
            self.selection_start[0],
            self.selection_end[0]
        )

        p1 = min(
            self.selection_start[1],
            self.selection_end[1]
        )

        p2 = max(
            self.selection_start[1],
            self.selection_end[1]
        )

        if getattr(self, "selection_in_pedal", False):
            self._clear_selection()
            has_pedal_events = False
            if self.midi.filter_track is not None:
                if 0 <= self.midi.filter_track < len(self.midi.tracks):
                    for ev in self.midi.tracks[self.midi.filter_track].pedals:
                        if t1 <= ev.time <= t2:
                            has_pedal_events = True
                            break
            else:
                for track in self.midi.tracks:
                    for ev in track.pedals:
                        if t1 <= ev.time <= t2:
                            has_pedal_events = True
                            break
                    if has_pedal_events:
                        break
        else:
            self._ensure_note_cache()
            result = []

            if self.midi.filter_track is None:
                track_items = list(
                    enumerate(self.midi.tracks)
                )
            else:
                index = self.midi.filter_track
                if 0 <= index < len(self.midi.tracks):
                    track_items = [
                        (index, self.midi.tracks[index])
                    ]
                else:
                    track_items = []

            for track_index, track in track_items:
                starts, notes, _, long_notes, long_starts = (
                    self._notes_starts_cache[
                        track_index
                    ]
                )

                i1 = bisect.bisect_left(
                    starts,
                    t2
                )

                for note in notes[:i1]:
                    if (
                        t1 <= note.start < t2 and
                        p1 <= note.pitch <= p2
                    ):
                        note._original_track = track_index
                        result.append(note)

                li1 = bisect.bisect_left(
                    long_starts,
                    t2
                )

                for note in long_notes[:li1]:
                    if (
                        t1 <= note.start < t2 and
                        p1 <= note.pitch <= p2
                    ):
                        note._original_track = track_index
                        result.append(note)

            self.selected_notes = result

        if getattr(self, "selection_in_pedal", False):
            cancel = not has_pedal_events
        else:
            cancel = not self.selected_notes

        self.last_selection_time_range = (t1, t2)
        self.last_selection_in_pedal = getattr(self, "selection_in_pedal", False)

        self.selection_mode = False
        self.selection_start = None
        self.selection_end = None

        if cancel:
            self.selection_rect = None
        else:
            in_lane = getattr(self, "selection_in_lane", False)
            in_pedal = getattr(self, "selection_in_pedal", False)
            self.selection_rect = (t1, t2, p1, p2, in_lane, in_pedal)

        self.update()

    def select_all(self):
        self._clear_selection()
        
        self.selection_in_lane = False
        self.selection_in_pedal = False

        notes = []
        if self.midi.filter_track is None:
            for track in self.midi.tracks:
                notes.extend(track.notes)
        else:
            if 0 <= self.midi.filter_track < len(self.midi.tracks):
                notes.extend(self.midi.tracks[self.midi.filter_track].notes)

        if not notes:
            return

        self.selected_notes = list(notes)

        t1 = min(n.start for n in notes)
        t2 = max(n.start + n.duration for n in notes)
        p1 = min(n.pitch for n in notes)
        p2 = max(n.pitch for n in notes)

        # 範囲選択の四角枠 (t1, t2, p1, p2, in_lane, in_pedal)
        self.selection_rect = (t1, t2, p1, p2, False, False)
        
        self.update()

    def copy_selected(self):
        if getattr(self, "last_selection_in_pedal", False):
            if not self.last_selection_time_range:
                return
            t1, t2 = self.last_selection_time_range
            
            if self.midi.filter_track is None:
                track_items = list(enumerate(self.midi.tracks))
            else:
                idx = self.midi.filter_track
                track_items = [(idx, self.midi.tracks[idx])] if 0 <= idx < len(self.midi.tracks) else []
                
            selected_pedals = []
            for t_idx, track in track_items:
                for ev in track.pedals:
                    if t1 <= ev.time <= t2:
                        ev._original_track = t_idx
                        selected_pedals.append(ev)
                        
            if not selected_pedals:
                return
            self.clipboard_pedals = self.midi.copy_pedals(selected_pedals)
            self.clipboard_notes = []
        else:
            if not self.selected_notes:
                return
            # 全トラック表示での貼り付け時に所属トラックを保持できるよう、
            # 現在の所属トラック情報をコピー前に更新する
            for n in self.selected_notes:
                t_idx = self._note_track_index(n)
                if t_idx is not None:
                    n._original_track = t_idx
            self.clipboard_notes = self.midi.copy_notes(self.selected_notes)
            self.clipboard_pedals = []

    def cut_selected(self):
        self.midi.push_undo()
        self.copy_selected()

        if getattr(self, "last_selection_in_pedal", False):
            self.delete_selected()
            return
        
        if not self.selected_notes:
            return

        for note in list(
            self.selected_notes
        ):
            self.midi.remove_note(
                note
            )

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.selected_notes = []
        self.selection_rect = None

        self.update()

    def paste_notes(self):
        if not self.clipboard_notes and not self.clipboard_pedals:
            return

        self.midi.push_undo()

        start_time = self.snap_time(
            self.play_position
        )

        # スナップにより再生バーより左側に
        # 貼り付けられる場合は切り上げて補正する
        if start_time < self.play_position - 1e-6:
            start_time = self.snap_time(
                self.play_position,
                mode="ceil"
            )

        if self.clipboard_notes:
            created = (
                self.midi.paste_notes(
                    self.clipboard_notes,
                    start_time
                )
            )
            self.selected_notes = created
            self.last_selection_time_range = None
            self.last_selection_in_pedal = False
        else:
            created_pedals = self.midi.paste_pedals(
                self.clipboard_pedals,
                start_time
            )
            if created_pedals:
                t1 = min(p.time for p in created_pedals)
                t2 = max(p.time for p in created_pedals)
                self.last_selection_time_range = (t1, t2)
                self.last_selection_in_pedal = True
                self.selected_notes = []

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.selection_rect = None
        self.update()

    def delete_selected(self):
        has_notes = bool(self.selected_notes)
        has_range = hasattr(self, "last_selection_time_range") and self.last_selection_time_range is not None
        
        if not has_notes and not has_range:
            return

        self.midi.push_undo()

        for note in list(
            self.selected_notes
        ):
            self.midi.remove_note(
                note
            )

        if has_range:
            if getattr(self, "last_selection_in_pedal", False):
                t1, t2 = self.last_selection_time_range
                track_idx = self.midi.active_track()
                events = self.midi.tracks[track_idx].pedals
                removed = [ev for ev in events if t1 <= ev.time <= t2]
                for ev in removed:
                    events.remove(ev)
                if removed and self.midi.filter_track is None and track_idx == 0:
                    self.midi.sync_pedals(0)
            self.last_selection_time_range = None
            self.last_selection_in_pedal = False

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.selected_notes = []
        self.selection_rect = None

        self.update()

    def _duplicate_selected_notes(
        self,
        source_notes
    ):
        """選択中のノーツを同じ位置に複製し、複製したノーツのリストを返す"""
        self._ensure_note_cache()

        placements = []
        for n in source_notes:
            t_idx = self._note_track_index(n)
            if t_idx is None or not (0 <= t_idx < len(self.midi.tracks)):
                continue
            placements.append((n, t_idx))

        if not placements:
            return None

        clones = []
        for n, t_idx in placements:
            new_note = n.clone()
            new_note._original_track = t_idx
            self.midi.tracks[t_idx].notes.append(new_note)
            clones.append(new_note)

        self.midi.sort()
        self.midi._bump()

        return clones

    def _note_track_index(
        self,
        note
    ):
        self._ensure_note_cache()
        idx = self._note_track_map.get(id(note))
        if idx is not None:
            return idx
        return self.midi.active_track()

    def _has_overlap_in_track_exclude(
        self,
        track_index,
        exclude_set,
        check_start,
        check_end,
        check_pitch
    ):
        if track_index < 0 or track_index >= len(self.midi.tracks):
            return False
            
        track = self.midi.tracks[track_index]
        for other in track.notes:
            if id(other) in exclude_set:
                continue
            if other.pitch != check_pitch:
                continue
            
            other_end = other.start + other.duration
            if (
                check_end > other.start + 1e-6 and
                check_start < other_end - 1e-6
            ):
                return True
                
        return False

    def _click_in_selection_rect(self, x, y):
        if self.selection_rect is None:
            return False, False, False
        rt1, rt2, rp1, rp2 = self.selection_rect[:4]
        r_in_lane = self.selection_rect[4] if len(self.selection_rect) > 4 else False
        r_in_pedal = self.selection_rect[5] if len(self.selection_rect) > 5 else False
        click_time = self.x_to_time(x)
        if r_in_lane:
            in_rect = rt1 <= click_time <= rt2
        else:
            click_pitch = self.y_to_pitch(y)
            in_rect = (
                rt1 <= click_time <= rt2 and
                rp1 <= click_pitch <= rp2
            )
        if not in_rect:
            return False, False, False
        rect_end_x = self.time_to_x(rt2)
        on_right_edge = abs(x - rect_end_x) <= 8
        return True, on_right_edge, False

    def note_at(
        self,
        x,
        y
    ):
        pitch = self.y_to_pitch(y)
        time = self.x_to_time(x)

        self._ensure_note_cache()
        
        if self.midi.filter_track is None:
            track_items = reversed(list(enumerate(self.midi.tracks)))
        else:
            index = self.midi.filter_track
            if 0 <= index < len(self.midi.tracks):
                track_items = [(index, self.midi.tracks[index])]
            else:
                track_items = []
                
        for track_index, track in track_items:
            starts, notes, max_duration, long_notes, long_starts = self._notes_starts_cache[track_index]
            
            i_end = bisect.bisect_right(starts, time)
            i_start = bisect.bisect_left(starts, time - max_duration - 0.001)
            
            for note in reversed(notes[i_start:i_end]):
                if note.pitch == pitch and note.start <= time <= note.start + note.duration:
                    return note

            li_end = bisect.bisect_right(long_starts, time)
            li_start = bisect.bisect_left(long_starts, time - max_duration - 0.001)

            for note in reversed(long_notes[li_start:li_end]):
                if note.pitch == pitch and note.start <= time <= note.start + note.duration:
                    return note

        return None

    def preview_pitch(
        self,
        pitch
    ):
        self._previewed_pitch = pitch

        self.audio.preview_note(
            pitch
        )

    def _velocity_bars(self):
        bars = []
        self._ensure_note_cache()

        visible_start = max(0.0, self.scroll_x)
        visible_end = self.x_to_time(self.width())

        if self.midi.filter_track is None:
            track_items = list(enumerate(self.midi.tracks))
        else:
            index = self.midi.filter_track
            if 0 <= index < len(self.midi.tracks):
                track_items = [(index, self.midi.tracks[index])]
            else:
                track_items = []

        for track_index, track in track_items:
            starts, notes, max_duration, long_notes, long_starts = self._notes_starts_cache[track_index]

            margin = max_duration + 0.001
            win_start = visible_start - margin
            win_end = visible_end + margin

            i0 = bisect.bisect_left(starts, win_start)
            i1 = bisect.bisect_right(starts, win_end)

            li0 = bisect.bisect_left(long_starts, win_start)
            li1 = bisect.bisect_right(long_starts, win_end)

            for note in notes[i0:i1]:
                x = self.time_to_x(note.start)
                width = min(max(3, int(note.duration / self.seconds_per_pixel)), 12)

                if x + width < self.left_width or x > self.width():
                    continue

                bars.append((note, x, width, track_index))

            for note in long_notes[li0:li1]:
                x = self.time_to_x(note.start)
                width = min(max(3, int(note.duration / self.seconds_per_pixel)), 12)

                if x + width < self.left_width or x > self.width():
                    continue

                bars.append((note, x, width, track_index))

        return bars

    def _velocity_bar_at(self, x):
        for note, bar_x, width, track_idx in (
            self._velocity_bars()
        ):
            if (
                bar_x - 1 <=
                x <=
                bar_x + width + 1
            ):
                return note, bar_x, width

        return None

    def _y_to_velocity(self, y):
        lane_top = (
            self.height() -
            self.bottom_height
        )

        usable = (
            self.velocity_lane_height -
            6
        )

        fraction = 1.0 - (
            y -
            (lane_top + 3)
        ) / usable

        return int(
            round(
                max(
                    0.0,
                    min(1.0, fraction)
                ) * 127.0
            )
        )

    def _velocity_to_y(self, value):
        lane_top = (
            self.height() -
            self.bottom_height
        )

        usable = (
            self.velocity_lane_height -
            6
        )

        value = max(
            0.0,
            min(1.0, value / 127.0)
        )

        return (
            lane_top +
            3 +
            (1.0 - value) *
            usable
        )

    def _apply_velocity(
        self,
        note,
        value
    ):
        value = max(
            0,
            min(
                127,
                int(round(value))
            )
        )

        if note.velocity == value:
            return

        if not self._vel_undo_pushed:
            self.midi.push_undo()
            self._vel_undo_pushed = True

        note.velocity = value

        self.midi._bump()

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

    def _vel_press(
        self,
        event,
        x,
        y
    ):
        self.vel_drag = None
        self._vel_undo_pushed = False

        value = self._y_to_velocity(y)

        mode = (
            "gradient"
            if event.button() == Qt.RightButton
            else "line"
        )

        self.vel_drag = {
            "x0": x,
            "y0": y,
            "x": x,
            "y": y,
            "value": value,
            "mode": mode
        }

        bar = self._velocity_bar_at(x)

        if bar is not None:
            self._apply_velocity(
                bar[0],
                value
            )

        self.update()

    def _vel_drag_move(
        self,
        x,
        y
    ):
        drag = self.vel_drag

        if drag is None:
            return

        value = self._y_to_velocity(y)

        x0 = drag["x0"]

        lo = min(x0, x)
        hi = max(x0, x)

        span = hi - lo

        if span < 1e-6:
            bar = self._velocity_bar_at(x)

            if bar is not None:
                self._apply_velocity(
                    bar[0],
                    value
                )
        else:
            for note, bar_x, width, track_idx in (
                self._velocity_bars()
            ):
                center = (
                    bar_x +
                    width * 0.5
                )

                if not (lo <= center <= hi):
                    continue

                t = (
                    center - lo
                ) / span

                if drag["mode"] == "gradient":
                    t = t * t * (3.0 - 2.0 * t)

                note_value = (
                    drag["value"] +
                    (value - drag["value"]) *
                    t
                )

                self._apply_velocity(
                    note,
                    note_value
                )

        drag["x"] = x
        drag["y"] = y

        self.update()

    def _vel_drag_end(self):
        self.vel_drag = None
        self._vel_undo_pushed = False
        self.update()

    def _set_velocity_dialog(self, note):
        if (
            note in self.selected_notes and
            len(self.selected_notes) > 1
        ):
            notes = list(
                self.selected_notes
            )
        else:
            notes = [note]

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("ベロシティ", "Velocity"))

        layout = QVBoxLayout(dlg)

        if len(notes) > 1:
            layout.addWidget(
                QLabel(
                    tr(f"選択中の {len(notes)} ノート", f"{len(notes)} notes selected")
                )
            )

        spin = QSpinBox()
        spin.setRange(1, 127)
        spin.setValue(note.velocity)
        layout.addWidget(spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        value = spin.value()

        if all(
            n.velocity == value
            for n in notes
        ):
            return

        self.midi.push_undo()

        for n in notes:
            n.velocity = value

        self.midi._bump()

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.update()

    def transpose_selected(self, offset):
        if not self.selected_notes:
            return

        self.midi.push_undo()

        for note in self.selected_notes:
            note.pitch = max(
                self.min_pitch,
                min(
                    self.max_pitch,
                    note.pitch + offset
                )
            )

        # 重複チェック
        is_duplicate = False
        exclude_set = {id(n) for n in self.selected_notes}
        
        for n in self.selected_notes:
            track_idx = self._note_track_index(n)
            new_end = n.start + n.duration
            if self._has_overlap_in_track_exclude(track_idx, exclude_set, n.start, new_end, n.pitch):
                is_duplicate = True
                break

        if is_duplicate:
            self.midi.undo()
            self.selected_notes = []
            self.selection_rect = None
            self.update()
            return

        self.preview_pitch(
            self.selected_notes[-1].pitch
        )

        self.midi._bump()
        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.update()

    def _merge_notes(self, notes):
        if len(notes) < 2:
            return

        groups = {}
        for note in notes:
            track_idx = self._note_track_index(note)
            key = (track_idx, note.pitch)
            if key not in groups:
                groups[key] = []
            groups[key].append(note)

        # 結合可能なグループ（同じピッチが2つ以上）があるか確認
        has_mergeable = False
        for group in groups.values():
            if len(group) >= 2:
                has_mergeable = True
                break
                
        if not has_mergeable:
            return

        self.midi.push_undo()
        from .midi import Note
        
        for (track_idx, pitch), group in groups.items():
            if len(group) < 2:
                continue

            track = self.midi.tracks[track_idx]
            
            group.sort(key=lambda n: n.start)
            start_time = group[0].start
            end_time = max(n.start + n.duration for n in group)
            velocity = group[0].velocity
            ch = getattr(group[0], "channel", 0)

            for note in group:
                if note in track.notes:
                    track.notes.remove(note)
                if note in self.selected_notes:
                    self.selected_notes.remove(note)

            new_note = Note(
                start=start_time,
                duration=end_time - start_time,
                pitch=pitch,
                velocity=velocity,
                channel=ch
            )
            track.notes.append(new_note)
            self.selected_notes.append(new_note)

        self.midi.sort()
        self.midi._bump()
        
        if self.audio.playing:
            self.audio.invalidate_midi_cache()
            
        self.update()

    def _vel_double_click(self, x, y):
        bar = self._velocity_bar_at(x)

        if bar is None:
            return

        self._set_velocity_dialog(
            bar[0]
        )

    def _pedal_event_at(self, events, x):
        best = None
        best_d = None

        t = self.x_to_time(x)
        times = [e.time for e in events]
        # x-6 から x+6 の範囲
        t0 = self.x_to_time(x - 6)
        t1 = self.x_to_time(x + 6)
        
        i0 = bisect.bisect_left(times, t0)
        i1 = bisect.bisect_right(times, t1)

        for ev in events[i0:i1]:
            d = abs(
                x -
                self.time_to_x(ev.time)
            )

            if d > 6:
                continue

            if best_d is None or d < best_d:
                best = ev
                best_d = d

        return best

    def _pedal_after_change(self):
        self.midi._bump()

        if self.audio.playing:
            self.audio.invalidate_midi_cache()

        self.update_timeline()
        self.update()

    def _pedal_press(
        self,
        event,
        x,
        y
    ):
        self.pedal_drag = None
        self._pedal_undo_pushed = False

        track_index = self.midi.active_track()

        events = self.midi.tracks[
            track_index
        ].pedals

        hit = self._pedal_event_at(
            events,
            x
        )

        if event.button() == Qt.LeftButton:
            is_selected = False
            in_selection_rect = False
            t1 = t2 = 0
            if getattr(self, "last_selection_in_pedal", False) and hasattr(self, "last_selection_time_range") and self.last_selection_time_range:
                t1, t2 = self.last_selection_time_range
                click_time = self.x_to_time(x)
                if t1 <= click_time <= t2:
                    in_selection_rect = True
                if hit is not None and t1 <= hit.time <= t2:
                    is_selected = True

            if is_selected or (in_selection_rect and self.last_selection_time_range):
                if not is_selected:
                    t1, t2 = self.last_selection_time_range
                selected_events = [ev for ev in events if t1 <= ev.time <= t2]
                if selected_events:
                    grid = self._get_snap_grid()
                    self.pedal_drag = {
                        "mode": "move_multiple",
                        "events": selected_events,
                        "press_time": self.x_to_time(x),
                        "changed": False,
                        "original_times": {id(ev): ev.time for ev in selected_events},
                        "grid": grid
                    }
                    self._clear_selection()
                    return
            elif hit is not None:
                self._clear_selection()
                self.midi.push_undo()
                new_ev = self.midi.toggle_pedal(track_index, hit.time)
                self._pedal_after_change()
                if new_ev is not None:
                    grid = self._get_snap_grid()
                    self.pedal_drag = {
                        "mode": "move",
                        "event": new_ev,
                        "press_time": new_ev.time,
                        "changed": False,
                        "grid": grid
                    }
            else:
                self._clear_selection()
                t = self.snap_time(self.x_to_time(x))
                self.midi.push_undo()
                self.midi.toggle_pedal(track_index, t)
                self.pedal_drag = {
                    "mode": "paint",
                    "last": t,
                    "press_t": t,
                    "changed": True
                }
                self._pedal_after_change()
        else:
            if hit is not None:
                self.midi.push_undo()

                self.midi.tracks[
                    track_index
                ].pedals.remove(hit)

                self._pedal_after_change()
            else:
                self.pedal_drag = {
                    "mode": "erase",
                    "x0": x
                }

    def _pedal_drag_move(
        self,
        x,
        y
    ):
        drag = self.pedal_drag

        if drag is None:
            return

        track_index = self.midi.active_track()

        events = self.midi.tracks[
            track_index
        ].pedals

        mode = drag["mode"]

        if mode == "move":
            t = self.snap_time(
                self.x_to_time(x),
                drag.get("grid", self.note_length)
            )

            ev = drag["event"]

            if abs(ev.time - t) > 1e-6:
                if not self._pedal_undo_pushed:
                    self.midi.push_undo()
                    self._pedal_undo_pushed = True

                events_list = self.midi.tracks[track_index].pedals
                ev.time = t
                events_list.sort(key=lambda e: e.time)
                self.midi._bump()

                drag["changed"] = True

                self._pedal_after_change()

        elif mode == "move_multiple":
            t = self.snap_time(
                self.x_to_time(x),
                drag.get("grid", self.note_length)
            )
            press_t = self.snap_time(drag["press_time"], drag.get("grid", self.note_length))
            dt = t - press_t

            if abs(dt) > 1e-6:
                if not self._pedal_undo_pushed:
                    self.midi.push_undo()
                    self._pedal_undo_pushed = True
                drag["changed"] = True

            if drag.get("changed", False):
                events_list = self.midi.tracks[track_index].pedals
                for ev in drag["events"]:
                    orig_t = drag["original_times"][id(ev)]
                    new_t = max(0.0, orig_t + dt)
                    ev.time = new_t
                    
                events_list.sort(key=lambda e: e.time)
                self.midi._bump()
                if hasattr(self, "last_selection_time_range") and self.last_selection_time_range:
                    t1, t2 = drag.get("original_range", self.last_selection_time_range)
                    if "original_range" not in drag:
                        drag["original_range"] = (t1, t2)
                    
                    self.last_selection_time_range = (max(0.0, t1 + dt), max(0.0, t2 + dt))

                self._pedal_after_change()

        elif mode == "paint":
            t = self.snap_time(
                self.x_to_time(x)
            )

            last = drag["last"]

            if t == last:
                return

            drag["last"] = t

            if any(
                abs(ev.time - t) < 1e-6
                for ev in events
            ):
                return

            if not self._pedal_undo_pushed:
                self.midi.push_undo()
                self._pedal_undo_pushed = True

            self.midi.toggle_pedal(
                track_index,
                t
            )

            drag["changed"] = True

            self._pedal_after_change()

        elif mode == "erase":
            t0 = self.x_to_time(
                drag["x0"]
            )

            t1 = self.x_to_time(x)

            lo = min(t0, t1)
            hi = max(t0, t1)

            removed = [
                ev
                for ev in events
                if lo <= ev.time <= hi
            ]

            if removed:
                if not self._pedal_undo_pushed:
                    self.midi.push_undo()
                    self._pedal_undo_pushed = True

                for ev in removed:
                    events.remove(ev)

                self._pedal_after_change()

    def _pedal_drag_end(self):
        drag = self.pedal_drag

        if drag is None:
            return

        track_index = self.midi.active_track()

        if drag["mode"] == "move" and not drag["changed"]:
            if not self._pedal_undo_pushed:
                self.midi.push_undo()
                self._pedal_undo_pushed = True

            ev = drag["event"]
            ev.down = not ev.down

            self._pedal_after_change()

        elif (
            drag["mode"] == "paint" and
            not drag["changed"]
        ):
            if not self._pedal_undo_pushed:
                self.midi.push_undo()
                self._pedal_undo_pushed = True

            self.midi.toggle_pedal(
                track_index,
                drag["press_t"]
            )

            self._pedal_after_change()

        if drag["mode"] in ("move", "move_multiple") and drag.get("changed", False):
            events = self.midi.tracks[track_index].pedals
            is_duplicate = False
            
            for i in range(len(events) - 1):
                if abs(events[i].time - events[i+1].time) < 1e-6:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                self.midi.undo()
                self._pedal_after_change()

        self.pedal_drag = None
        self._pedal_undo_pushed = False

    def _pedal_double_click(self, x, y):
        track_index = self.midi.active_track()

        hit = self._pedal_event_at(
            self.midi.tracks[
                track_index
            ].pedals,
            x
        )

        if hit is None:
            return

        self.midi.push_undo()

        hit.down = not hit.down

        self._pedal_after_change()

    def draw_pitch_grid(
        self,
        painter
    ):
        black_notes = {
            1,
            3,
            6,
            8,
            10
        }

        for pitch in range(
            self.min_pitch,
            self.max_pitch + 1
        ):
            y = self.pitch_to_y(
                pitch
            )

            if (
                y +
                self.note_height <
                self.top_height
            ):
                continue

            if (
                y >
                self.height() -
                self.bottom_height
            ):
                continue

            if pitch % 12 in black_notes:
                background = QColor(
                    20,
                    21,
                    25
                )
            else:
                background = QColor(
                    38,
                    40,
                    45
                )

            painter.fillRect(
                self.left_width,
                int(y),
                self.width() -
                self.left_width,
                self.note_height,
                background
            )

            painter.setPen(
                QPen(
                    QColor(
                        60,
                        62,
                        68
                    )
                )
            )

            painter.drawLine(
                self.left_width,
                int(y),
                self.width(),
                int(y)
            )

    def _ensure_spectrum_levels(
        self
    ):
        data = self.spectrum.data

        if data is None or data.size == 0:
            return

        # The display range determines the lower dB bound as well as the
        # source data.  Include both in the cache key so moving the
        # sensitivity slider recalculates the normalisation levels.
        key = (
            id(data),
            self.spectrum_db_range
        )

        if key == self._spectrum_level_key:
            return

        hi = float(
            np.percentile(
                data,
                99.8
            )
        )

        lo = hi - self.spectrum_db_range

        self._spectrum_hi = hi
        self._spectrum_lo = lo
        self._spectrum_level_key = key

    def _build_spectrum_image(
        self
    ):
        data = self.spectrum.data

        if data is None or data.size == 0:
            return None

        self._ensure_spectrum_levels()

        rows, cols = data.shape

        hop = (
            self.spectrum.hop_length /
            self.spectrum.sr
            if self.spectrum.sr
            else 0.0
        )

        if hop <= 0:
            return None

        down = 1

        while cols // down > 65536:
            down *= 2

        self._spectrum_hop_dt = hop * down

        flipped = data[::-1]

        if down > 1:
            usable = (
                cols -
                cols % down
            )

            flipped = flipped[
                :,
                :usable
            ].reshape(
                rows,
                usable // down,
                down
            ).mean(axis=2)

        span = max(
            self._spectrum_hi -
            self._spectrum_lo,
            1e-6
        )

        v = np.clip(
            (
                flipped -
                self._spectrum_lo
            ) / span,
            0.0,
            1.0
        )

        g = v ** 2.5

        t = np.clip(
            g,
            0.0,
            1.0
        )

        stops = [
            0.0,
            0.125,
            0.375,
            0.625,
            0.875,
            1.0
        ]

        r_pts = [
            0.0,
            0.0,
            0.0,
            255.0,
            255.0,
            128.0
        ]

        g_pts = [
            0.0,
            0.0,
            255.0,
            255.0,
            0.0,
            0.0
        ]

        b_pts = [
            128.0,
            255.0,
            255.0,
            0.0,
            0.0,
            0.0
        ]

        r = np.interp(
            t,
            stops,
            r_pts
        )

        g_ch = np.interp(
            t,
            stops,
            g_pts
        )

        b_ch = np.interp(
            t,
            stops,
            b_pts
        )

        alpha = np.where(
            g <
            self.spectrum_threshold,
            0.0,
            25.0 +
            g * 85.0
        )

        img = np.empty(
            (
                rows,
                flipped.shape[1],
                4
            ),
            dtype=np.uint8
        )

        img[:, :, 0] = np.clip(
            r,
            0,
            255
        ).astype(np.uint8)

        img[:, :, 1] = np.clip(
            g_ch,
            0,
            255
        ).astype(np.uint8)

        img[:, :, 2] = np.clip(
            b_ch,
            0,
            255
        ).astype(np.uint8)

        img[:, :, 3] = np.clip(
            alpha,
            0,
            255
        ).astype(np.uint8)

        return QImage(
            img.data,
            img.shape[1],
            rows,
            img.strides[0],
            QImage.Format.Format_RGBA8888
        ).copy()

    def warm_spectrum_cache(
        self
    ):
        if self.spectrum.data is None:
            return

        key = (
            id(self.spectrum.data),
            self.spectrum_db_range,
            self.spectrum_threshold
        )

        if key != self._spectrum_key:
            self._spectrum_key = key
            self._spectrum_image = (
                self._build_spectrum_image()
            )

    def draw_spectrum(
        self,
        painter
    ):
        if self.spectrum.data is None:
            return

        data = self.spectrum.data

        if data.size == 0:
            return

        dest_w = (
            self.width() -
            self.left_width
        )

        if dest_w <= 0:
            return

        hop_dt = (
            self.spectrum.hop_length /
            self.spectrum.sr
            if self.spectrum.sr
            else 0.0
        )

        if hop_dt <= 0:
            return

        key = (
            id(data),
            self.spectrum_db_range,
            self.spectrum_threshold
        )

        if key != self._spectrum_key:
            self._spectrum_key = key
            self._spectrum_image = (
                self._build_spectrum_image()
            )

        image = self._spectrum_image

        if image is None:
            return

        eff_dt = (
            self._spectrum_hop_dt
            if self._spectrum_hop_dt > 0
            else hop_dt
        )

        offset = self.audio.offset

        src_x = (
            self.scroll_x -
            offset
        ) / eff_dt

        img_w = float(
            image.width()
        )

        if src_x >= img_w:
            return

        src_x = max(
            0.0,
            src_x
        )

        src_w = min(
            dest_w *
            self.seconds_per_pixel /
            eff_dt,
            img_w - src_x - 1.0
        )

        if src_w <= 0:
            return

        dest_x = (
            float(self.left_width) +
            (
                src_x *
                eff_dt +
                offset -
                self.scroll_x
            ) /
            self.seconds_per_pixel
        )

        dest_w = (
            src_w *
            eff_dt /
            self.seconds_per_pixel
        )

        painter.drawImage(
            QRectF(
                dest_x,
                float(
                    self.top_height -
                    self.scroll_y
                ),
                float(dest_w),
                float(
                    data.shape[0] *
                    self.note_height
                )
            ),
            image,
            QRectF(
                src_x,
                0.0,
                src_w,
                float(
                    data.shape[0]
                )
            )
        )

    def draw_grid(
        self,
        painter
    ):
        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        beat_start = self.midi.time_to_beat(
            visible_start
        )

        beat_end = self.midi.time_to_beat(
            visible_end
        )

        bpm = self.midi.tempo_at(
            visible_start
        )

        beat_px = (
            60.0 / bpm
        ) / self.seconds_per_pixel

        # 拍子の分母に応じたグリッド単位(四分音符ビート単位)
        # 例: 4/4=1.0, 3/4=1.0, 6/8=0.5, 3/8=0.5, x/16=0.25
        sigs_all = self.midi.time_signatures

        segments = []

        for i, (t, num, den) in enumerate(sigs_all):
            ss = self.time_signature_start_beat(
                t
            )

            if i + 1 < len(sigs_all):
                se = self.time_signature_start_beat(
                    sigs_all[i + 1][0]
                )
            else:
                se = float("inf")

            if se <= ss:
                continue

            unit = self.midi.bar_length_beats(
                1,
                den
            )

            segments.append(
                (
                    ss,
                    se,
                    unit
                )
            )

        visible_segments = [
            (
                ss,
                se,
                u
            )
            for ss, se, u in segments
            if se > beat_start and
            ss < beat_end
        ]

        unit_first = next(
            (
                u
                for _, _, u in visible_segments
            ),
            1.0
        )

        unit_px = unit_first * beat_px

        draw_unit = (
            unit_px >= 3.0
        )

        note_bottom = (
            self.height() -
            self.bottom_height
        )

        b0 = math.ceil(beat_start)

        linear = True

        for tt, _bb in self.midi.tempos:
            if (
                visible_start < tt <
                visible_end
            ):
                linear = False
                break

        if linear:
            x0 = self.time_to_x(
                self.midi.beat_to_time(
                    b0
                )
            )

            def line_x(beat):
                return (
                    x0 +
                    (beat - b0) *
                    beat_px
                )

        else:
            def line_x(beat):
                return self.time_to_x(
                    self.midi.beat_to_time(
                        beat
                    )
                )

        if draw_unit:
            painter.setPen(
                QPen(
                    QColor(
                        170,
                        170,
                        180,
                        165
                    ),
                    1
                )
            )

            for ss, se, u in visible_segments:
                z = min(se, beat_end)

                k0 = max(
                    0,
                    math.ceil(
                        (
                            beat_start -
                            ss
                        ) / u - 1e-9
                    )
                )

                k1 = math.floor(
                    (z - ss) / u +
                    1e-9
                )

                for k in range(k0, k1 + 1):
                    x = line_x(ss + k * u)

                    if x < self.left_width:
                        continue

                    if x > self.width():
                        break

                    painter.drawLine(
                        int(x),
                        self.top_height,
                        int(x),
                        note_bottom
                    )

        painter.setPen(
            QPen(
                QColor(
                    195,
                    195,
                    205,
                    155
                ),
                2
            )
        )

        sigs = self.midi.time_signatures

        for i, (t, num, den) in enumerate(sigs):
            # 分母を反映した小節長(四分音符ビート単位、例: 3/8=1.5)
            bar = self.midi.bar_length_beats(num, den)

            seg_start = self.time_signature_start_beat(
                t
            )

            if i + 1 < len(sigs):
                seg_end = self.time_signature_start_beat(
                    sigs[i + 1][0]
                )
            else:
                seg_end = beat_end

            if seg_end <= seg_start:
                continue

            k0 = max(
                0,
                math.ceil(
                    (
                        beat_start -
                        seg_start
                    ) / bar - 1e-9
                )
            )

            k1 = math.floor(
                (
                    min(seg_end, beat_end) -
                    seg_start
                ) / bar + 1e-9
            )

            for k in range(k0, k1 + 1):
                b = seg_start + k * bar

                x = line_x(b)

                if x < self.left_width:
                    continue

                if x > self.width():
                    break

                painter.drawLine(
                    int(x),
                    self.top_height,
                    int(x),
                    note_bottom
                )

    def draw_tap_overlay(
        self,
        painter
    ):
        if not self.tap_mode:
            return

        top = self.top_height
        bottom = self.height() - self.bottom_height

        if bottom <= top:
            return

        bpm = max(
            1.0,
            self._tap_disp_bpm
        )

        spb = 60.0 / bpm

        phi = self._tap_disp_phi

        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        k0 = int(
            math.floor(
                (visible_start - phi) / spb
            )
        ) - 1

        k1 = int(
            math.ceil(
                (visible_end - phi) / spb
            )
        ) + 1

        pulse = TapTempoEngine.beat_pulse(
            self._tap_fit
        )

        width = self.width()

        for k in range(k0, k1 + 1):
            beat_time = phi + k * spb

            x = self.time_to_x(beat_time)

            if x < self.left_width:
                continue

            if x > width:
                break

            major = (
                k % 4 == 0
                if k >= 0
                else False
            )

            base_alpha = 120 if major else 70
            alpha = int(
                base_alpha * (0.6 + 0.4 * pulse)
            )

            painter.setPen(
                QPen(
                    QColor(
                        90,
                        215,
                        255,
                        alpha
                    ),
                    2 if major else 1
                )
            )

            painter.drawLine(
                int(x),
                top,
                int(x),
                bottom
            )

        font = QFont(
            "Segoe UI",
            10
        )

        painter.setFont(font)

        fit = self._tap_fit

        if fit is None or fit["n"] < 2:
            text = tr(
                "拍に合わせて Shift+Space を連打してください（Space: 停止して適用 / Esc: キャンセル）",
                "Tap Shift+Space along the beat (Space: stop and apply / Esc: cancel)"
            )

            fm = painter.fontMetrics()

            tw = fm.horizontalAdvance(text)

            bx = max(
                self.left_width,
                (
                    width + self.left_width - tw
                ) // 2
            )

            painter.fillRect(
                bx - 8,
                top + 6,
                tw + 16,
                fm.height() + 8,
                QColor(
                    20,
                    22,
                    28,
                    200
                )
            )

            painter.setPen(
                QPen(
                    QColor(
                        235,
                        235,
                        240
                    )
                )
            )

            painter.drawText(
                bx,
                top + 8 +
                fm.ascent() + 4,
                text
            )
            return

        rms = fit["rms_ms"]

        disp_bpm = int(math.floor(self._tap_disp_bpm + 0.5))

        if fit["n"] >= MIN_TAPS_FOR_APPLY:
            text = tr(
                f"{disp_bpm} BPM   "
                f"{fit['n']}タップ  ±{rms:.0f}ms",
                f"{disp_bpm} BPM   "
                f"{fit['n']} taps  ±{rms:.0f}ms"
            )
        else:
            text = tr(
                f"{disp_bpm} BPM   "
                f"{fit['n']}/{MIN_TAPS_FOR_APPLY}タップ",
                f"{disp_bpm} BPM   "
                f"{fit['n']}/{MIN_TAPS_FOR_APPLY} taps"
            )

        fm = painter.fontMetrics()

        tw = fm.horizontalAdvance(text)

        bx = width - tw - 24

        painter.fillRect(
            bx - 8,
            top + 6,
            tw + 16,
            fm.height() + 8,
            QColor(
                20,
                22,
                28,
                200
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    90,
                    215,
                    255
                )
            )
        )

        painter.drawText(
            bx,
            top + 8 +
            fm.ascent() + 4,
            text
        )

    def _ensure_note_cache(self):
        if (
            getattr(self, "_notes_starts_version", -1) !=
            self.midi.mutation_version
        ):
            self._notes_starts_cache = {}
            self._note_pens = {}
            self._note_track_map = {}
            total_count = 0

            for track_index, track in enumerate(
                self.midi.tracks
            ):
                short_notes = []
                long_notes = []
                max_short_duration = 0.0

                for note in track.notes:
                    self._note_track_map[id(note)] = track_index
                    total_count += 1
                    if note.duration > 8.0:
                        long_notes.append(note)
                    else:
                        short_notes.append(note)
                        if note.duration > max_short_duration:
                            max_short_duration = note.duration

                short_notes.sort(key=lambda n: n.start)
                starts = [n.start for n in short_notes]

                long_notes.sort(key=lambda n: n.start)
                long_starts = [n.start for n in long_notes]

                self._notes_starts_cache[track_index] = (
                    starts,
                    short_notes,
                    max_short_duration,
                    long_notes,
                    long_starts
                )

                rgb = getattr(self, "track_colors", [(100, 100, 200)])[
                    track_index % len(getattr(self, "track_colors", [(100, 100, 200)]))
                ]

                fill_brush = QBrush(
                    QColor(
                        rgb[0],
                        rgb[1],
                        rgb[2],
                        230
                    )
                )

                outline_pen = QPen(
                    QColor(
                        245,
                        250,
                        255,
                        235
                    ),
                    1
                )

                sel_brush = QBrush(
                    QColor(
                        255,
                        255,
                        255,
                        240
                    )
                )

                sel_pen = QPen(
                    QColor(
                        255,
                        255,
                        255,
                        255
                    ),
                    2
                )

                self._note_pens[track_index] = (
                    fill_brush,
                    outline_pen,
                    sel_brush,
                    sel_pen
                )

            self._cached_note_count = total_count
            self._notes_starts_version = (
                self.midi.mutation_version
            )

    def draw_notes(
        self,
        painter
    ):
        self._ensure_note_cache()

        # ドラッグ中のノーツは位置が毎フレーム変わるため、
        # mutation_version を更新しない限り starts キャッシュ(描画範囲
        # の判定に使用)は古いまま。キャッシュ経由で描画すると画面外
        # から移動してきたノーツが描画漏れするため、ドラッグ対象は
        # 通常ループから外し、末尾でライブ座標により描画する。
        drag_ids = set()

        if self.drag_original_notes:
            for n, _os, _op, _od in self.drag_original_notes:
                drag_ids.add(id(n))

        if self.drag_note is not None:
            drag_ids.add(id(self.drag_note))

        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        note_top = self.top_height
        note_bottom = (
            self.height() -
            self.bottom_height
        )

        if self.midi.filter_track is None:
            track_items = list(
                enumerate(self.midi.tracks)
            )
        else:
            index = self.midi.filter_track

            if 0 <= index < len(self.midi.tracks):
                track_items = [
                    (index, self.midi.tracks[index])
                ]
            else:
                track_items = []

        for track_index, track in track_items:
            starts, notes, max_duration, long_notes, long_starts = (
                self._notes_starts_cache[
                    track_index
                ]
            )

            margin = (
                max_duration +
                0.001
            )

            win_start = visible_start - margin
            win_end = visible_end + margin

            i0 = bisect.bisect_left(
                starts,
                win_start
            )

            i1 = bisect.bisect_right(
                starts,
                win_end
            )

            li0 = bisect.bisect_left(
                long_starts,
                win_start
            )

            li1 = bisect.bisect_right(
                long_starts,
                win_end
            )

            fill_brush, outline_pen, sel_brush, sel_pen = (
                self._note_pens[track_index]
            )

            sel_set = (
                {
                    id(note)
                    for note in self.selected_notes
                }
                if self.selected_notes
                else None
            )

            for note in notes[i0:i1]:
                if id(note) in drag_ids:
                    continue

                y = self.pitch_to_y(
                    note.pitch
                )

                if (
                    y +
                    self.note_height <
                    note_top
                ):
                    continue

                if y > note_bottom:
                    continue

                x = self.time_to_x(
                    note.start
                )

                width = (
                    note.duration /
                    self.seconds_per_pixel
                )

                if (
                    x +
                    width <
                    self.left_width
                ):
                    continue

                if x > self.width():
                    continue

                if (
                    sel_set is not None and
                    id(note) in sel_set
                ):
                    painter.setPen(sel_pen)
                    painter.setBrush(sel_brush)
                else:
                    painter.setPen(outline_pen)
                    painter.setBrush(fill_brush)

                painter.drawRoundedRect(
                    int(x),
                    int(y + 2),
                    max(
                        1,
                        int(width)
                    ),
                    self.note_height - 4,
                    3,
                    3
                )

            for note in long_notes[li0:li1]:
                if id(note) in drag_ids:
                    continue

                y = self.pitch_to_y(
                    note.pitch
                )

                if (
                    y +
                    self.note_height <
                    note_top
                ):
                    continue

                if y > note_bottom:
                    continue

                x = self.time_to_x(
                    note.start
                )

                width = (
                    note.duration /
                    self.seconds_per_pixel
                )

                if (
                    x +
                    width <
                    self.left_width
                ):
                    continue

                if x > self.width():
                    continue

                if (
                    sel_set is not None and
                    id(note) in sel_set
                ):
                    painter.setPen(sel_pen)
                    painter.setBrush(sel_brush)
                else:
                    painter.setPen(outline_pen)
                    painter.setBrush(fill_brush)

                painter.drawRoundedRect(
                    int(x),
                    int(y + 2),
                    max(
                        1,
                        int(width)
                    ),
                    self.note_height - 4,
                    3,
                    3
                )

        drag_entries = []

        if self.drag_original_notes:
            for n, _os, _op, _od in self.drag_original_notes:
                drag_entries.append(n)

        if (
            self.drag_note is not None and
            not any(
                e is self.drag_note
                for e in drag_entries
            )
        ):
            drag_entries.append(self.drag_note)

        if drag_entries:
            sel_set = (
                {
                    id(note)
                    for note in self.selected_notes
                }
                if self.selected_notes
                else None
            )

            for drag_note in drag_entries:
                track_index = self._note_track_map.get(
                    id(drag_note),
                    getattr(self, "drag_track_index", 0)
                )

                pens = self._note_pens.get(
                    track_index
                )

                if pens is None:
                    continue

                (
                    fill_brush,
                    outline_pen,
                    sel_brush,
                    sel_pen
                ) = pens

                y = self.pitch_to_y(
                    drag_note.pitch
                )

                if (
                    y +
                    self.note_height <
                    note_top or
                    y > note_bottom
                ):
                    continue

                x = self.time_to_x(
                    drag_note.start
                )

                width = (
                    drag_note.duration /
                    self.seconds_per_pixel
                )

                if (
                    x +
                    width <
                    self.left_width or
                    x > self.width()
                ):
                    continue

                if (
                    sel_set is not None and
                    id(drag_note) in sel_set
                ):
                    painter.setPen(sel_pen)
                    painter.setBrush(sel_brush)
                else:
                    painter.setPen(outline_pen)
                    painter.setBrush(fill_brush)

                painter.drawRoundedRect(
                    int(x),
                    int(y + 2),
                    max(
                        1,
                        int(width)
                    ),
                    self.note_height - 4,
                    3,
                    3
                )

    def draw_lyrics(self, painter):
        """ノーツに設定された歌詞をノートの左上に描画する。"""
        if self.midi.filter_track is None:
            track_items = list(
                enumerate(self.midi.tracks)
            )
        else:
            index = self.midi.filter_track

            if 0 <= index < len(self.midi.tracks):
                track_items = [
                    (index, self.midi.tracks[index])
                ]
            else:
                track_items = []

        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        note_bottom = (
            self.height() -
            self.bottom_height
        )

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for _track_index, track in track_items:
            for note in track.notes:
                lyric = getattr(note, "lyric", "")

                if not lyric:
                    continue

                end = note.start + note.duration

                if (
                    end < visible_start or
                    note.start > visible_end
                ):
                    continue

                y = self.pitch_to_y(note.pitch)

                if (
                    y < self.top_height or
                    y > note_bottom
                ):
                    continue

                x = self.time_to_x(note.start)

                painter.setPen(
                    QColor(255, 235, 150, 235)
                )
                painter.setBrush(Qt.NoBrush)

                painter.drawText(
                    QPointF(x + 1.0, y - 1.0),
                    lyric
                )

    def draw_selection(
        self,
        painter
    ):
        rect = None
        in_lane = False
        in_pedal = False

        if self.selection_mode and self.selection_start and self.selection_end:
            t1 = min(
                self.selection_start[0],
                self.selection_end[0]
            )

            t2 = max(
                self.selection_start[0],
                self.selection_end[0]
            )

            p1 = max(
                self.min_pitch,
                min(
                    self.max_pitch,
                    self.selection_start[1]
                )
            )

            p2 = max(
                self.min_pitch,
                min(
                    self.max_pitch,
                    self.selection_end[1]
                )
            )

            rect = (t1, t2, p1, p2)
            in_lane = getattr(self, "selection_in_lane", False)
            in_pedal = getattr(self, "selection_in_pedal", False)

        elif self.selection_rect is not None:
            t1, t2, p1, p2 = self.selection_rect[:4]
            rect = (t1, t2, p1, p2)
            in_lane = self.selection_rect[4] if len(self.selection_rect) > 4 else False
            in_pedal = self.selection_rect[5] if len(self.selection_rect) > 5 else False

        if rect is None:
            return

        t1, t2, p1, p2 = rect

        x1 = self.time_to_x(
            t1
        )

        x2 = self.time_to_x(
            t2
        )

        if in_lane:
            if in_pedal:
                y1 = self.height() - self.bottom_height + self.velocity_lane_height
                y2 = y1 + self.pedal_lane_height
            else:
                y1 = self.height() - self.bottom_height
                y2 = y1 + self.velocity_lane_height
        else:
            p_hi = max(p1, p2)
            p_lo = min(p1, p2)

            # 上下端もノート行(音階)にスナップさせる
            y1 = self.pitch_to_y(p_hi)
            y2 = self.pitch_to_y(p_lo) + self.note_height

        painter.setPen(
            QPen(
                QColor(
                    255,
                    210,
                    80,
                    220
                ),
                1,
                Qt.DashLine
            )
        )

        painter.setBrush(
            QBrush(
                QColor(
                    255,
                    210,
                    80,
                    45
                )
            )
        )

        painter.drawRect(
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        )

    def draw_keyboard(
        self,
        painter
    ):
        black_notes = {
            1,
            3,
            6,
            8,
            10
        }

        painter.setBrush(
            Qt.NoBrush
        )

        painter.fillRect(
            0,
            self.top_height,
            self.left_width,
            self.height() -
            self.top_height -
            self.bottom_height,
            QColor(
                28,
                29,
                33
            )
        )

        names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B"
        ]

        label_metrics = painter.fontMetrics()
        label_left = 8
        label_right = 4

        for pitch in range(
            self.min_pitch,
            self.max_pitch + 1
        ):
            y = self.pitch_to_y(
                pitch
            )

            if (
                y +
                self.note_height <
                self.top_height
            ):
                continue

            if (
                y >
                self.height() -
                self.bottom_height
            ):
                continue

            note_class = pitch % 12

            octave = (
                pitch //
                12 -
                1
            )

            if note_class in black_notes:
                painter.fillRect(
                    0,
                    int(y),
                    self.left_width,
                    self.note_height,
                    QColor(
                        8,
                        8,
                        10
                    )
                )

                painter.setPen(
                    QPen(
                        QColor(
                            240,
                            240,
                            245
                        )
                    )
                )

            else:
                painter.fillRect(
                    0,
                    int(y),
                    self.left_width,
                    self.note_height,
                    QColor(
                        215,
                        216,
                        220
                    )
                )

                painter.setPen(
                    QPen(
                        QColor(
                            30,
                            30,
                            35
                        )
                    )
                )

            painter.drawRect(
                0,
                int(y),
                self.left_width - 1,
                self.note_height - 1
            )

            label = f"{names[note_class]}{octave}"

            if (
                label_metrics.horizontalAdvance(label) <=
                self.left_width - label_left - label_right and
                label_metrics.height() <=
                self.note_height - 2
            ):
                baseline = (
                    y +
                    (
                        self.note_height +
                        label_metrics.ascent() -
                        label_metrics.descent()
                    ) / 2
                )

                painter.drawText(
                    label_left,
                    int(baseline),
                    label
                )

    def draw_time_labels(
        self,
        painter
    ):
        painter.fillRect(
            0,
            0,
            self.width(),
            self.top_height,
            QColor(
                24,
                25,
                29
            )
        )

        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        beat_start = self.midi.time_to_beat(
            visible_start
        )

        beat_end = self.midi.time_to_beat(
            visible_end
        )

        painter.setFont(
            QFont(
                "Segoe UI",
                9
            )
        )

        bpm = self.midi.tempo_at(
            visible_start
        )

        beat_px = (
            60.0 / bpm
        ) / self.seconds_per_pixel

        # 拍子の分母に応じたグリッド単位で目盛りを描画
        sigs_tl = self.midi.time_signatures

        tl_segments = []

        for i, (t, num, den) in enumerate(sigs_tl):
            ss = self.time_signature_start_beat(
                t
            )

            if i + 1 < len(sigs_tl):
                se = self.time_signature_start_beat(
                    sigs_tl[i + 1][0]
                )
            else:
                se = float("inf")

            if se <= ss:
                continue

            tl_segments.append(
                (
                    ss,
                    se,
                    self.midi.bar_length_beats(
                        1,
                        den
                    )
                )
            )

        unit_tl = next(
            (
                u
                for ss, se, u in tl_segments
                if se > beat_start and
                ss < beat_end
            ),
            1.0
        )

        if unit_tl * beat_px >= 3.0:
            painter.setPen(
                QPen(
                    QColor(
                        170,
                        170,
                        180,
                        165
                    ),
                    1
                )
            )

            for ss, se, u in tl_segments:
                z = min(se, beat_end)

                k0 = max(
                    0,
                    math.ceil(
                        (
                            beat_start -
                            ss
                        ) / u - 1e-9
                    )
                )

                k1 = math.floor(
                    (z - ss) / u +
                    1e-9
                )

                for k in range(k0, k1 + 1):
                    x = self.time_to_x(
                        self.midi.beat_to_time(
                            ss + k * u
                        )
                    )

                    if x < self.left_width:
                        continue

                    if x > self.width():
                        break

                    painter.drawLine(
                        int(x),
                        0,
                        int(x),
                        self.top_height
                    )

        painter.setPen(
            QPen(
                QColor(
                    175,
                    175,
                    185,
                    170
                ),
                2
            )
        )

        sigs = self.midi.time_signatures

        for i, (t, num, den) in enumerate(sigs):
            # 分母を反映した小節長(四分音符ビート単位、例: 3/8=1.5)
            bar = self.midi.bar_length_beats(num, den)

            seg_start = self.time_signature_start_beat(
                t
            )

            if i + 1 < len(sigs):
                seg_end = self.time_signature_start_beat(
                    sigs[i + 1][0]
                )
            else:
                seg_end = beat_end

            if seg_end <= seg_start:
                continue

            k0 = max(
                0,
                math.ceil(
                    (
                        beat_start -
                        seg_start
                    ) / bar - 1e-9
                )
            )

            k1 = math.floor(
                (
                    min(seg_end, beat_end) -
                    seg_start
                ) / bar + 1e-9
            )

            for k in range(k0, k1 + 1):
                b = seg_start + k * bar

                t_m = self.midi.beat_to_time(
                    b
                )

                x = self.time_to_x(
                    t_m
                )

                if x < self.left_width:
                    continue

                if x > self.width():
                    break

                painter.drawLine(
                    int(x),
                    0,
                    int(x),
                    self.top_height
                )

        # 小節番号ラベル: 拍子ごとの小節境界を直接たどり、
        # ズームに応じて間引きして描画する
        last_label_x = float("-inf")

        for i, (t_sec, num, den) in enumerate(sigs):
            bar = self.midi.bar_length_beats(num, den)

            seg_start = self.time_signature_start_beat(t_sec)

            if i + 1 < len(sigs):
                seg_end = self.time_signature_start_beat(
                    sigs[i + 1][0]
                )
            else:
                seg_end = beat_end

            if seg_end <= seg_start:
                continue

            label_every = max(
                1,
                int(
                    round(
                        80.0 /
                        max(bar * beat_px, 1.0)
                    )
                )
            )

            k0 = max(
                0,
                math.ceil(
                    (beat_start - seg_start) / bar - 1e-9
                )
            )

            k1 = math.floor(
                (
                    min(seg_end, beat_end) -
                    seg_start
                ) / bar + 1e-9
            )

            base_measure = self.midi.segment_start_measure(i)

            for k in range(k0, k1 + 1):
                if k % label_every != 0:
                    continue

                t_m = self.midi.beat_to_time(
                    seg_start + k * bar
                )

                x = self.time_to_x(t_m)

                if x < self.left_width:
                    continue

                if x - last_label_x < 4.0:
                    continue

                last_label_x = x

                painter.setPen(
                    QPen(
                        QColor(
                            215,
                            215,
                            220
                        )
                    )
                )

                painter.drawText(
                    int(x + 4),
                    14,
                    str(base_measure + k + 1)
                )

                painter.drawText(
                    int(x + 4),
                    27,
                    f"{t_m:.2f}s"
                )

        painter.setPen(
            QPen(
                QColor(
                    255,
                    200,
                    60
                )
            )
        )

        last_x = float("-inf")

        for t_sec, bpm in self.midi.tempos:
            if (
                t_sec <
                visible_start -
                0.001
            ):
                continue

            if t_sec > visible_end:
                break

            x = self.time_to_x(
                t_sec
            )

            if x < self.left_width:
                continue

            if (
                x - last_x <
                4.0
            ):
                continue

            last_x = x

            painter.drawLine(
                int(x),
                self.top_height - 13,
                int(x),
                self.top_height
            )

            painter.drawText(
                int(x + 3),
                self.top_height - 13,
                f"BPM{bpm:g}"
            )

        painter.setPen(
            QPen(
                QColor(
                    110,
                    220,
                    140
                )
            )
        )

        last_x = float("-inf")

        for t_sec, num, den in (
            self.midi.time_signatures
        ):
            if (
                t_sec <
                visible_start -
                0.001
            ):
                continue

            if t_sec > visible_end:
                break

            x = self.time_to_x(
                t_sec
            )

            if x < self.left_width:
                continue

            if (
                x - last_x <
                4.0
            ):
                continue

            last_x = x

            painter.drawLine(
                int(x),
                self.top_height - 4,
                int(x),
                self.top_height
            )

            painter.drawText(
                int(x + 3),
                self.top_height - 4,
                f"{num}/{den}"
            )

    def draw_play_position(
        self,
        painter
    ):
        x = self.time_to_x(
            self.play_position
        )

        if (
            x <
            self.left_width or
            x >
            self.width()
        ):
            return

        painter.setPen(
            QPen(
                QColor(
                    255,
                    65,
                    70
                ),
                2
            )
        )

        painter.drawLine(
            int(x),
            0,
            int(x),
            self.height() -
            self.bottom_height
        )

    def draw_scrub_area(
        self,
        painter
    ):
        painter.fillRect(
            0,
            self.height() -
            self.bottom_height,
            self.width(),
            self.bottom_height,
            QColor(
                22,
                23,
                27
            )
        )

        self._ensure_note_cache()
        count = getattr(self, "_cached_note_count", 0)

        num, den = self.midi.time_sig_at(
            self.play_position
        )

        measure, beat_in, _num = (
            self.midi.measure_beat(
                self.play_position
            )
        )

        bpm = self.midi.tempo_at(
            self.play_position
        )

        painter.setFont(
            QFont(
                "Segoe UI",
                9
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    210,
                    210,
                    215
                )
            )
        )

        # painter.drawText(
        #     10,
        #     self.height() - 26,
        #     f"ノーツ {count} | "
        #     f"拍子 {num}/{den} | "
        #     f"{measure + 1}小節 "
        #     f"{beat_in + 1}拍 | "
        #     f"{bpm:g} BPM | "
        #     f"{self.play_position:.2f} / "
        #     f"{self.audio_duration:.2f}s"
        # )

        scrub_y = (
            self.height() -
            self.scrub_height
        )

        painter.fillRect(
            0,
            scrub_y,
            self.width(),
            self.scrub_height,
            QColor(
                15,
                16,
                20
            )
        )

        fraction = (
            self.play_position /
            self.audio_duration
            if self.audio_duration > 0
            else 0.0
        )

        painter.fillRect(
            0,
            scrub_y,
            int(
                self.width() *
                fraction
            ),
            self.scrub_height,
            QColor(
                45,
                48,
                58
            )
        )

        x_play = int(
            self.width() *
            fraction
        )

        painter.fillRect(
            x_play,
            scrub_y,
            2,
            self.scrub_height,
            QColor(
                255,
                65,
                70
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    150,
                    152,
                    160
                )
            )
        )

        painter.drawText(
            max(
                10,
                x_play + 6
            ),
            scrub_y + 14,
            f"{self.play_position:.2f}s"
        )

    def draw_lane_grid(
        self,
        painter,
        top,
        bottom
    ):
        visible_start = max(
            0.0,
            self.scroll_x
        )

        visible_end = self.x_to_time(
            self.width()
        )

        beat_start = self.midi.time_to_beat(
            visible_start
        )

        beat_end = self.midi.time_to_beat(
            visible_end
        )

        bpm_lane = self.midi.tempo_at(
            visible_start
        )

        beat_px_lane = (
            60.0 / bpm_lane
        ) / self.seconds_per_pixel

        sigs_ln = self.midi.time_signatures

        ln_segments = []

        for i, (t, num, den) in enumerate(sigs_ln):
            ss = self.time_signature_start_beat(
                t
            )

            if i + 1 < len(sigs_ln):
                se = self.time_signature_start_beat(
                    sigs_ln[i + 1][0]
                )
            else:
                se = float("inf")

            if se <= ss:
                continue

            ln_segments.append(
                (
                    ss,
                    se,
                    self.midi.bar_length_beats(
                        1,
                        den
                    )
                )
            )

        painter.setPen(
            QPen(
                QColor(
                    125,
                    125,
                    140,
                    60
                ),
                1
            )
        )

        for ss, se, u in ln_segments:
            z = min(se, beat_end)

            if u * beat_px_lane < 3.0:
                continue

            k0 = max(
                0,
                math.ceil(
                    (
                        beat_start -
                        ss
                    ) / u - 1e-9
                )
            )

            k1 = math.floor(
                (z - ss) / u +
                1e-9
            )

            for k in range(k0, k1 + 1):
                x = self.time_to_x(
                    self.midi.beat_to_time(
                        ss + k * u
                    )
                )

                if x < self.left_width:
                    continue

                if x > self.width():
                    break

                painter.drawLine(
                    int(x),
                    top,
                    int(x),
                    bottom
                )

        painter.setPen(
            QPen(
                QColor(
                    165,
                    165,
                    175,
                    95
                ),
                1
            )
        )

        sigs = self.midi.time_signatures

        for i, (t, num, den) in enumerate(sigs):
            # 分母を反映した小節長(四分音符ビート単位、例: 3/8=1.5)
            bar = self.midi.bar_length_beats(num, den)

            seg_start = self.time_signature_start_beat(t)

            if i + 1 < len(sigs):
                seg_end = self.time_signature_start_beat(
                    sigs[i + 1][0]
                )
            else:
                seg_end = beat_end

            if seg_end <= seg_start:
                continue

            k0 = max(
                0,
                math.ceil(
                    (
                        beat_start -
                        seg_start
                    ) / bar - 1e-9
                )
            )

            k1 = math.floor(
                (
                    min(seg_end, beat_end) -
                    seg_start
                ) / bar + 1e-9
            )

            for k in range(k0, k1 + 1):
                x = self.time_to_x(
                    self.midi.beat_to_time(
                        seg_start + k * bar
                    )
                )

                if x < self.left_width:
                    continue

                if x > self.width():
                    break

                painter.drawLine(
                    int(x),
                    top,
                    int(x),
                    bottom
                )

    def draw_velocity_lane(
        self,
        painter
    ):
        lane_top = (
            self.height() -
            self.bottom_height
        )

        lane_bottom = (
            lane_top +
            self.velocity_lane_height
        )

        painter.fillRect(
            0,
            lane_top,
            self.width(),
            self.velocity_lane_height,
            QColor(
                26,
                27,
                31
            )
        )

        painter.fillRect(
            0,
            lane_top,
            self.left_width,
            self.velocity_lane_height,
            QColor(
                30,
                31,
                35
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    255,
                    255,
                    255,
                    40
                ),
                1
            )
        )

        painter.drawLine(
            0,
            lane_top,
            self.width(),
            lane_top
        )

        painter.setFont(
            QFont(
                "Segoe UI",
                8
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    175,
                    175,
                    185
                )
            )
        )

        painter.drawText(
            6,
            lane_top + 14,
            tr("ベロシティ", "Velocity")
        )

        painter.drawText(
            6,
            lane_bottom - 6,
            "0 - 127"
        )

        self.draw_lane_grid(
            painter,
            lane_top,
            lane_bottom
        )

        painter.setPen(
            QPen(
                QColor(
                    255,
                    255,
                    255,
                    24
                ),
                1
            )
        )

        y_64 = int(
            self._velocity_to_y(64)
        )

        painter.drawLine(
            self.left_width,
            y_64,
            self.width(),
            y_64
        )

        sel_brush = QBrush(
            QColor(
                255,
                190,
                55,
                240
            )
        )

        sel_set = (
            {
                id(note)
                for note in self.selected_notes
            }
            if self.selected_notes
            else None
        )

        painter.setPen(
            QPen(
                QColor(
                    0,
                    0,
                    0,
                    0
                ),
                0
            )
        )

        bar_bottom = (
            lane_bottom - 2
        )

        for note, bar_x, width, track_idx in (
            self._velocity_bars()
        ):
            y = int(
                self._velocity_to_y(
                    note.velocity
                )
            )

            if sel_set is not None and id(note) in sel_set:
                painter.setBrush(sel_brush)
            else:
                rgb = self.track_colors[
                    track_idx % len(self.track_colors)
                ]
                painter.setBrush(
                    QBrush(
                        QColor(
                            rgb[0],
                            rgb[1],
                            rgb[2],
                            225
                        )
                    )
                )

            painter.drawRect(
                int(bar_x),
                y,
                width,
                bar_bottom - y
            )

        drag = self.vel_drag

        if (
            drag is not None and
            drag.get("x") is not None
        ):
            x0 = drag["x0"]
            y0 = drag["y0"]
            x1 = drag["x"]
            y1 = drag["y"]

            lo = min(x0, x1)
            hi = max(x0, x1)

            span = hi - lo

            if span >= 1e-6:
                v0 = drag["value"]
                v1 = self._y_to_velocity(y1)

                painter.setPen(
                    QPen(
                        QColor(
                            255,
                            255,
                            255,
                            170
                        ),
                        1,
                        Qt.DashLine
                    )
                )

                painter.drawLine(
                    int(x0),
                    int(y0),
                    int(x1),
                    int(y1)
                )

                painter.setPen(
                    QPen(
                        QColor(
                            0,
                            0,
                            0,
                            0
                        ),
                        0
                    )
                )

                painter.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            255,
                            230
                        )
                    )
                )

                for note, bar_x, width, track_idx in (
                    self._velocity_bars()
                ):
                    center = (
                        bar_x +
                        width * 0.5
                    )

                    if not (lo <= center <= hi):
                        continue

                    t = (
                        center - lo
                    ) / span

                    if drag["mode"] == "gradient":
                        t = t * t * (3.0 - 2.0 * t)

                    value = (
                        v0 +
                        (v1 - v0) * t
                    )

                    yv = self._velocity_to_y(
                        value
                    )

                    painter.drawEllipse(
                        QPointF(
                            center,
                            yv
                        ),
                        2.5,
                        2.5
                    )

    def draw_pedal_lane(
        self,
        painter
    ):
        lane_top = (
            self.height() -
            self.bottom_height +
            self.velocity_lane_height
        )

        lane_bottom = (
            lane_top +
            self.pedal_lane_height
        )

        painter.fillRect(
            0,
            lane_top,
            self.width(),
            self.pedal_lane_height,
            QColor(
                23,
                24,
                28
            )
        )

        painter.fillRect(
            0,
            lane_top,
            self.left_width,
            self.pedal_lane_height,
            QColor(
                27,
                28,
                32
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    255,
                    255,
                    255,
                    40
                ),
                1
            )
        )

        painter.drawLine(
            0,
            lane_top,
            self.width(),
            lane_top
        )

        painter.setFont(
            QFont(
                "Segoe UI",
                8
            )
        )

        painter.setPen(
            QPen(
                QColor(
                    175,
                    175,
                    185
                )
            )
        )

        painter.drawText(
            6,
            lane_top + 14,
            tr("ペダル(CC64)", "Pedal (CC64)")
        )

        self.draw_lane_grid(
            painter,
            lane_top,
            lane_bottom
        )

        track_index = self.midi.active_track()

        events = self.midi.tracks[
            track_index
        ].pedals

        events.sort(
            key=lambda e: e.time
        )

        for down, up in (
            self.midi.pedal_pairs(
                track_index
            )
        ):
            x1 = self.time_to_x(down)
            x2 = self.time_to_x(up)

            if (
                x2 < self.left_width or
                x1 > self.width()
            ):
                continue

            painter.fillRect(
                int(x1),
                lane_top,
                int(x2 - x1),
                self.pedal_lane_height,
                QColor(
                    120,
                    230,
                    150,
                    26
                )
            )

            painter.setPen(
                QPen(
                    QColor(
                        120,
                        230,
                        150
                    ),
                    2
                )
            )

            painter.drawLine(
                int(x1),
                lane_top + 2,
                int(x2),
                lane_top + 2
            )

        visible_start = max(0.0, self.scroll_x)
        visible_end = self.x_to_time(self.width())
        
        times = [e.time for e in events]
        i0 = bisect.bisect_left(times, visible_start - 1.0)
        i1 = bisect.bisect_right(times, visible_end + 1.0)

        for ev in events[i0:i1]:
            x = int(
                self.time_to_x(ev.time)
            )

            if (
                x <
                self.left_width - 6 or
                x >
                self.width() + 6
            ):
                continue

            is_selected = False
            if getattr(self, "last_selection_in_pedal", False) and hasattr(self, "last_selection_time_range") and self.last_selection_time_range:
                t1, t2 = self.last_selection_time_range
                if t1 <= ev.time <= t2:
                    is_selected = True

            if is_selected:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
            elif ev.down:
                painter.setBrush(
                    QBrush(
                        QColor(
                            120,
                            230,
                            150
                        )
                    )
                )

                painter.setPen(
                    QPen(
                        QColor(
                            190,
                            255,
                            205
                        ),
                        1
                    )
                )
            else:
                painter.setBrush(
                    QBrush(
                        QColor(
                            235,
                            130,
                            130
                        )
                    )
                )

                painter.setPen(
                    QPen(
                        QColor(
                            255,
                            190,
                            190
                        ),
                        1
                    )
                )

            if ev.down:
                painter.drawPolygon(
                    [
                        QPointF(
                            x,
                            lane_top + 2
                        ),
                        QPointF(
                            x - 7,
                            lane_top + 16
                        ),
                        QPointF(
                            x + 7,
                            lane_top + 16
                        ),
                    ]
                )
            else:
                painter.drawPolygon(
                    [
                        QPointF(
                            x,
                            lane_bottom - 2
                        ),
                        QPointF(
                            x - 7,
                            lane_bottom - 16
                        ),
                        QPointF(
                            x + 7,
                            lane_bottom - 16
                        ),
                    ]
                )



    def paintEvent(
        self,
        event
    ):
        painter = QPainter(
            self
        )

        try:
            painter.setRenderHint(
                QPainter.Antialiasing,
                False
            )

            painter.fillRect(
                self.rect(),
                QColor(
                    25,
                    26,
                    30
                )
            )

            self.draw_pitch_grid(
                painter
            )

            self.draw_spectrum(
                painter
            )

            self.draw_grid(
                painter
            )

            self.draw_notes(
                painter
            )

            if self.lyric_mode:
                self.draw_lyrics(
                    painter
                )

            self.draw_keyboard(
                painter
            )

            self.draw_time_labels(
                painter
            )

            if self.tap_mode:
                self.draw_tap_overlay(
                    painter
                )

            self.draw_play_position(
                painter
            )

            self.draw_scrub_area(
                painter
            )

            self.draw_velocity_lane(
                painter
            )

            self.draw_pedal_lane(
                painter
            )

            self.draw_selection(
                painter
            )
        finally:
            painter.end()
