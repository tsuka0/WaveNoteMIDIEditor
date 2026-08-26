from dataclasses import dataclass, field
import copy
import bisect
import gzip
import json
import math
import uuid as uuidlib
import mido
from .i18n import tr

# Synthesizer V Studio の時間単位 blick。
# 1四分音符 = 705,600,000 blick (公式スクリプトAPIの SV.QUARTER 相当)
BLICKS_PER_QUARTER = 705600000

@dataclass
class Note:
    start: float
    duration: float
    pitch: int
    velocity: int = 100
    channel: int = 0
    lyric: str = ""

    def clone(self):
        return copy.copy(self)

@dataclass
class PedalEvent:
    time: float
    down: bool

@dataclass
class Track:
    name: str = field(default_factory=lambda: tr("トラック 1", "Track 1"))
    notes: list = field(default_factory=list)
    pedals: list = field(default_factory=list)
    channel: int = 0

class MidiData:
    def __init__(self):
        self.tracks = [Track()]
        self.bpm = 120
        self.tempos = [(0.0, 120.0)]
        self.time_signatures = [(0.0, 4, 4)]
        self.filter_track = None
        self.play_all_tracks = False
        self.beat_phase = 0.0
        self.has_file = False

        self._undo = []
        self._redo = []

        self.mutation_version = 0

        self._tempo_times = []
        self._tempo_cum_time = []
        self._tempo_cum_beat = []
        self._sig_times = []
        self._sig_cum_measure = []
        self._notes_cache = None
        self._notes_cache_version = -1

        self._rebuild_tempo_cache()
        self._rebuild_sig_cache()

    def snapshot(self):
        return {
            "tracks": [
                Track(
                    name=track.name,
                    notes=[
                        Note(
                            note.start,
                            note.duration,
                            note.pitch,
                            note.velocity,
                            getattr(note, 'channel', 0),
                            getattr(note, 'lyric', "")
                        )
                        for note in track.notes
                    ],
                    pedals=[
                        copy.copy(p)
                        for p in track.pedals
                    ],
                    channel=track.channel
                )
                for track in self.tracks
            ],
            "tempos": self.tempos.copy(),
            "time_signatures": self.time_signatures.copy(),
            "beat_phase": self.beat_phase,
            "extra_state": getattr(self, "extra_state", {}).copy()
        }

    def restore(self, snap):
        self.extra_state = snap.get("extra_state", {}).copy()
        self.tracks = [
            Track(
                name=track.name,
                notes=[
                    Note(
                        note.start,
                        note.duration,
                        note.pitch,
                        note.velocity,
                        getattr(note, 'channel', 0),
                        getattr(note, 'lyric', "")
                    )
                    for note in track.notes
                ],
                pedals=[
                    PedalEvent(
                        pedal.time,
                        pedal.down
                    )
                    for pedal in track.pedals
                ],
                channel=getattr(track, 'channel', 0)
            )
            for track in snap["tracks"]
        ]

        self.tempos = [
            (t, b)
            for t, b in snap["tempos"]
        ]

        self.time_signatures = [
            (t, n, d)
            for t, n, d in snap["time_signatures"]
        ]

        self.beat_phase = float(
            snap.get("beat_phase", 0.0)
        )

        self.bpm = self.tempos[0][1]

        self._refresh_caches()
        self._bump()

        if self.filter_track is not None:
            self.filter_track = max(
                0,
                min(
                    self.filter_track,
                    len(self.tracks) - 1
                )
            )

    def push_undo(self):
        self._undo.append(
            self.snapshot()
        )

        if len(self._undo) > 100:
            self._undo.pop(0)

        self._redo.clear()

    def undo(self):
        if not self._undo:
            return False

        self._redo.append(
            self.snapshot()
        )

        self.restore(
            self._undo.pop()
        )

        return True

    def redo(self):
        if not self._redo:
            return False

        self._undo.append(
            self.snapshot()
        )

        self.restore(
            self._redo.pop()
        )

        return True

    def clear_history(self):
        self._undo.clear()
        self._redo.clear()

    @property
    def notes(self):
        if (
            self._notes_cache is not None and
            self._notes_cache_version ==
            self.mutation_version
        ):
            return self._notes_cache

        combined = []

        for track in self.tracks:
            combined.extend(track.notes)

        self._notes_cache = combined
        self._notes_cache_version = self.mutation_version

        return combined

    def visible_notes(self):
        if self.filter_track is None:
            return self.notes

        if 0 <= self.filter_track < len(self.tracks):
            return self.tracks[self.filter_track].notes

        return []

    def pedal_pairs(self, track_index):
        pairs = []

        down = None

        for ev in sorted(
            self.tracks[track_index].pedals,
            key=lambda e: e.time
        ):
            if ev.down:
                if down is None:
                    down = ev.time
            elif down is not None:
                pairs.append((down, ev.time))
                down = None

        if down is not None:
            max_end = max(
                (
                    note.start + note.duration
                    for note in self.tracks[track_index].notes
                ),
                default=down
            )

            if max_end > down:
                pairs.append((down, max_end))

        return pairs

    def pedal_state_at(self, track_index, time):
        state = False

        for ev in sorted(
            self.tracks[track_index].pedals,
            key=lambda e: e.time
        ):
            if ev.time <= time:
                state = ev.down
            else:
                break

        return state

    def add_pedal(self, track_index, time, down):
        time = max(0.0, float(time))

        events = self.tracks[track_index].pedals

        for ev in events:
            if abs(ev.time - time) < 1e-6:
                ev.down = bool(down)
                self._bump()
                return ev

        ev = PedalEvent(time, bool(down))

        events.append(ev)
        events.sort(key=lambda e: e.time)

        self._bump()
        if self.filter_track is None and track_index == 0:
            self.sync_pedals(0)

        return ev

    def toggle_pedal(self, track_index, time):
        time = max(0.0, float(time))

        events = self.tracks[track_index].pedals

        for ev in events:
            if abs(ev.time - time) < 1e-6:
                if ev.down:
                    ev.down = False
                else:
                    events.remove(ev)
                self._bump()
                if self.filter_track is None and track_index == 0:
                    self.sync_pedals(0)
                return ev if ev.down else None

        down = not self.pedal_state_at(
            track_index,
            time
        )

        ev = PedalEvent(time, down)

        events.append(ev)
        events.sort(key=lambda e: e.time)

        self._bump()
        if self.filter_track is None and track_index == 0:
            self.sync_pedals(0)

        return ev

    def move_pedal(self, track_index, event, time):
        time = max(0.0, float(time))

        events = self.tracks[track_index].pedals

        other = next(
            (
                ev
                for ev in events
                if (
                    ev is not event and
                    abs(ev.time - time) < 1e-6
                )
            ),
            None
        )

        if other is not None:
            # 重複配置できないように移動をキャンセル
            return
        else:
            event.time = time
            events.sort(key=lambda e: e.time)

        self._bump()
        if self.filter_track is None and track_index == 0:
            self.sync_pedals(0)

    def remove_pedal(self, track_index, time):
        events = self.tracks[track_index].pedals

        removed = [
            ev
            for ev in events
            if abs(ev.time - time) < 1e-6
        ]

        for ev in removed:
            events.remove(ev)

        if removed:
            self._bump()
            if self.filter_track is None and track_index == 0:
                self.sync_pedals(0)

        return bool(removed)

    def sync_pedals(self, source_track_index):
        source_pedals = self.tracks[source_track_index].pedals
        for i, track in enumerate(self.tracks):
            if i == source_track_index:
                continue
            track.pedals = [PedalEvent(ev.time, ev.down) for ev in source_pedals]
        self._bump()

    def _sustain_extended_notes(self, notes, pairs):
        if not pairs:
            return list(notes)

        downs = [p[0] for p in pairs]
        ups = [p[1] for p in pairs]

        result = []

        for note in notes:
            end = note.start + note.duration

            i = bisect.bisect_right(downs, end) - 1

            if i >= 0 and ups[i] > end:
                result.append(
                    Note(
                        note.start,
                        ups[i] - note.start,
                        note.pitch,
                        note.velocity,
                        getattr(note, 'channel', 0),
                        getattr(note, 'lyric', "")
                    )
                )
            else:
                result.append(note.clone())

        return result

    def visible_extended_notes(self):
        if (
            not self.play_all_tracks and
            self.filter_track is not None and
            0 <= self.filter_track < len(self.tracks)
        ):
            notes = self._sustain_extended_notes(
                self.tracks[self.filter_track].notes,
                self.pedal_pairs(self.filter_track)
            )
            ch = self.tracks[self.filter_track].channel
            for n in notes:
                n.channel = ch
            return notes

        result = []

        for i, track in enumerate(self.tracks):
            notes = self._sustain_extended_notes(
                track.notes,
                self.pedal_pairs(i)
            )
            for n in notes:
                n.channel = track.channel
            result.extend(notes)

        return result

    def max_extended_end(self):
        end = 0.0

        for i, track in enumerate(self.tracks):
            pairs = self.pedal_pairs(i)

            if not pairs:
                for note in track.notes:
                    end = max(
                        end,
                        note.start + note.duration
                    )

                continue

            downs = [p[0] for p in pairs]
            ups = [p[1] for p in pairs]

            for note in track.notes:
                n_end = note.start + note.duration

                j = bisect.bisect_right(downs, n_end) - 1

                if j >= 0 and ups[j] > n_end:
                    end = max(end, ups[j])
                else:
                    end = max(end, n_end)

        return end

    def active_track(self):
        if self.filter_track is None:
            return 0

        return max(
            0,
            min(
                self.filter_track,
                len(self.tracks) - 1
            )
        )

    def add_track(self, name=None):
        track = Track(
            name or tr(f"トラック {len(self.tracks) + 1}", f"Track {len(self.tracks) + 1}")
        )

        if self.filter_track is None and self.tracks:
            track.pedals = [PedalEvent(ev.time, ev.down) for ev in self.tracks[0].pedals]

        self.tracks.append(track)

        self._bump()

        return track

    def set_filter_track(self, index):
        self.filter_track = index

    def _time_to_beat_mapper(self):
        self._ensure_caches()

        tempos = list(self.tempos)
        cum_time = list(self._tempo_cum_time)
        cum_beat = list(self._tempo_cum_beat)
        phase = self.beat_phase

        def mapper(time):
            i = bisect.bisect_right(cum_time, time) - 1

            if i < 0:
                return phase + time * tempos[0][1] / 60.0

            return (
                phase +
                cum_beat[i] +
                (
                    time -
                    cum_time[i]
                ) *
                tempos[i][1] /
                60.0
            )

        return mapper

    def _apply_tempo_map_change(self, apply):
        has_content = any(
            track.notes or track.pedals
            for track in self.tracks
        )

        if not has_content:
            apply()
            return

        old_time_to_beat = self._time_to_beat_mapper()

        apply()

        for track in self.tracks:
            for note in track.notes:
                start_beat = old_time_to_beat(note.start)
                end_beat = old_time_to_beat(
                    note.start + note.duration
                )

                new_start = self.beat_to_time(start_beat)

                note.duration = max(
                    1e-3,
                    self.beat_to_time(end_beat) - new_start
                )
                note.start = new_start

            for pedal in track.pedals:
                pedal.time = self.beat_to_time(
                    old_time_to_beat(pedal.time)
                )

        self._bump()

    def set_base_tempo(self, bpm):
        def apply():
            self.tempos[0] = (0.0, float(bpm))
            self.bpm = float(bpm)
            self._refresh_caches()

        self._apply_tempo_map_change(apply)

    def set_beat_phase(self, beats):
        self.beat_phase = float(beats)

    def apply_tempo_fit(self, start_time, bpm, phi_time):
        """タップ計測結果をテンポマップへ反映する。

        start_time: 新テンポを追加する位置(再生開始位置・秒)。
        bpm: 推定BPM。
        phi_time: 拍番号0に相当するオーディオ時刻(秒)。
        start_time 以降のテンポマーカーを計測結果で置き換える。
        位相は追加位置自身を基準にし、その位置が小節頭「1」なら
        新グリッドでも小節頭「1」の真上に乗るよう補正する
        (タップ位相ではなく譜面上の位置を優先する)。
        既存ノーツは旧グリッド上の拍位置を保ったまま新グリッドへ
        再配置される。
        """
        bpm = float(bpm)
        t_new = max(0.0, float(start_time))
        spb = 60.0 / max(1e-6, bpm)

        self._ensure_caches()

        # 追加位置の現在の拍値を、その位置の拍子の小節頭に揃える。
        # ほぼ小節頭で開始されていれば小節頭「1」が動かず、
        # そこを基準に新BPMのグリッドが刻まれる。
        # 小節途中からの開始の場合は近傍の整数拍に揃える。
        base_beat = self.time_to_beat(t_new)

        num, den = self.time_sig_at(t_new)
        bar_beats = self.bar_length_beats(num, den)

        bar_target = (
            math.floor(base_beat / bar_beats + 0.5) *
            bar_beats
        )

        if abs(bar_target - base_beat) <= max(0.5, bar_beats * 0.25):
            target = bar_target
        else:
            target = math.floor(base_beat + 0.5)

        delta = target - base_beat

        def apply():
            out = [
                (t, b)
                for t, b in self.tempos
                if t < t_new - 1e-6
            ]

            if not out:
                out = [(0.0, bpm)]
            else:
                out.append((t_new, bpm))

            self.tempos = out
            self.bpm = self.tempos[0][1]
            self.beat_phase += delta
            self._refresh_caches()

        self._apply_tempo_map_change(apply)

    def add_tempo(self, time, bpm):
        def apply():
            t_new = max(0.0, float(time))

            out = []
            replaced = False

            for t, b in self.tempos:
                if abs(t - t_new) < 1e-6:
                    out.append((t_new, float(bpm)))
                    replaced = True
                else:
                    out.append((t, b))

            if not replaced:
                out.append((t_new, float(bpm)))
                out.sort(key=lambda x: x[0])

            self.tempos = out
            self.bpm = self.tempos[0][1]
            self._refresh_caches()

        self._apply_tempo_map_change(apply)

    def add_time_signature(self, time, numerator, denominator):
        time = max(0.0, float(time))
        numerator = max(1, int(numerator))
        denominator = max(1, int(denominator))

        out = []
        replaced = False

        for t, n, d in self.time_signatures:
            if abs(t - time) < 1e-6:
                out.append((time, numerator, denominator))
                replaced = True
            else:
                out.append((t, n, d))

        if not replaced:
            out.append((time, numerator, denominator))
            out.sort(key=lambda x: x[0])

        self.time_signatures = out
        self._rebuild_sig_cache()

    def remove_tempo(self, time):
        def apply():
            out = [
                (t, b)
                for t, b in self.tempos
                if abs(t - time) >= 1e-6
            ]

            if not out:
                return

            self.tempos = out
            self.bpm = self.tempos[0][1]
            self._refresh_caches()

        self._apply_tempo_map_change(apply)

    def remove_time_signature(self, time):
        out = [
            (t, n, d)
            for t, n, d in self.time_signatures
            if abs(t - time) >= 1e-6
        ]

        if not out:
            return

        self.time_signatures = out
        self._rebuild_sig_cache()

    def _rebuild_tempo_cache(self):
        tempos = self.tempos

        self._tempo_times = [
            t
            for t, _ in tempos
        ]

        cum_time = [0.0]
        cum_beat = [0.0]

        for i in range(len(tempos) - 1):
            t0 = tempos[i][0]
            t1 = tempos[i + 1][0]
            bpm = tempos[i][1]

            beats = (
                (t1 - t0) *
                bpm /
                60.0
            )

            cum_time.append(cum_time[-1] + (t1 - t0))
            cum_beat.append(cum_beat[-1] + beats)

        self._tempo_cum_time = cum_time
        self._tempo_cum_beat = cum_beat

    def _rebuild_sig_cache(self):
        sigs = self.time_signatures

        self._sig_times = [
            t
            for t, _num, _den in sigs
        ]

        cum_measure = [0]

        for i in range(len(sigs) - 1):
            bar = self.bar_length_beats(
                sigs[i][1],
                sigs[i][2]
            )

            seg_beats = (
                self.time_to_beat(sigs[i + 1][0]) -
                self.time_to_beat(sigs[i][0])
            )

            cum_measure.append(
                cum_measure[-1] +
                max(
                    0,
                    int(
                        round(
                            seg_beats / bar
                        )
                    )
                )
            )

        self._sig_cum_measure = cum_measure

    def _refresh_caches(self):
        self._rebuild_tempo_cache()
        self._rebuild_sig_cache()

    def _ensure_caches(self):
        tempos = self.tempos

        if (
            not tempos or
            len(self._tempo_times) != len(tempos) or
            self._tempo_times[0] != tempos[0][0] or
            self._tempo_times[-1] != tempos[-1][0]
        ):
            self._rebuild_tempo_cache()

        sigs = self.time_signatures

        if (
            not sigs or
            len(self._sig_times) != len(sigs) or
            self._sig_times[0] != sigs[0][0] or
            self._sig_times[-1] != sigs[-1][0]
        ):
            self._rebuild_sig_cache()

    def _bump(self):
        self.mutation_version += 1
        self._notes_cache = None
        self._notes_cache_version = -1

    def bar_length_beats(self, num, den):
        """1小節の長さを四分音符ビート単位で返す (4/4=4.0, 3/4=3.0, 3/8=1.5, 6/8=3.0)"""
        num = max(1, int(num))
        den = max(1, int(den))
        return num * 4.0 / den

    def time_sig_at(self, time):
        self._ensure_caches()

        sigs = self.time_signatures

        if not sigs:
            return 4, 4

        i = bisect.bisect_right(
            self._sig_times,
            time
        ) - 1

        i = max(0, i)

        return (
            max(1, int(sigs[i][1])),
            max(1, int(sigs[i][2]))
        )

    def measure_beat(self, time):
        self._ensure_caches()

        beat = self.time_to_beat(time)

        sigs = self.time_signatures

        if not sigs:
            return 0, 0, 4

        i = bisect.bisect_right(
            self._sig_times,
            time
        ) - 1

        i = max(0, i)

        num = max(
            1,
            int(sigs[i][1])
        )

        den = max(
            1,
            int(sigs[i][2])
        )

        bar = self.bar_length_beats(num, den)

        # 拍子の1拍の長さ(四分音符ビート単位、例: 3/8なら0.5)
        unit = 4.0 / den

        seg_start = self.time_to_beat(
            sigs[i][0]
        )

        off = beat - seg_start

        # beat<->time round trips can leave values like 23.999999999999996
        # instead of 24.0, which made measure labels duplicate or vanish.
        snapped = round(off)

        if abs(off - snapped) < 1e-6:
            off = float(snapped)

        return (
            self._sig_cum_measure[i] + int(off // bar),
            int(round(off / unit)) % num,
            num
        )

    def segment_start_measure(self, index):
        """拍子セグメント開始時点の累積小節数を返す"""
        self._ensure_caches()

        if 0 <= index < len(self._sig_cum_measure):
            return self._sig_cum_measure[index]

        return 0

    def tempo_at(self, time):
        self._ensure_caches()

        i = bisect.bisect_right(
            self._tempo_times,
            time
        ) - 1

        if i < 0:
            return self.tempos[0][1]

        return self.tempos[i][1]

    def beat_to_time(self, beat):
        self._ensure_caches()

        beat = beat - self.beat_phase

        i = bisect.bisect_right(
            self._tempo_cum_beat,
            beat
        ) - 1

        if i < 0:
            return beat * 60.0 / self.tempos[0][1]

        return (
            self._tempo_cum_time[i] +
            (
                beat -
                self._tempo_cum_beat[i]
            ) *
            60.0 /
            self.tempos[i][1]
        )

    def time_to_beat(self, time):
        self._ensure_caches()

        i = bisect.bisect_right(
            self._tempo_cum_time,
            time
        ) - 1

        if i < 0:
            return (
                self.beat_phase +
                time * self.tempos[0][1] / 60.0
            )

        return (
            self.beat_phase +
            self._tempo_cum_beat[i] +
            (
                time -
                self._tempo_cum_time[i]
            ) *
            self.tempos[i][1] /
            60.0
        )

    def seconds_to_tick(self, seconds, ticks_per_beat=480):
        self._ensure_caches()

        i = bisect.bisect_right(
            self._tempo_cum_time,
            seconds
        ) - 1

        if i < 0:
            return (
                seconds *
                self.tempos[0][1] /
                60.0 *
                ticks_per_beat
            )

        return (
            (
                self._tempo_cum_beat[i] +
                (
                    seconds -
                    self._tempo_cum_time[i]
                ) *
                self.tempos[i][1] /
                60.0
            ) *
            ticks_per_beat
        )

    def add_note(self, start, duration, pitch, velocity=100):
        track_idx = self.active_track()
        track = self.tracks[track_idx]
        note = Note(start, duration, pitch, velocity, track.channel)

        # 重複チェック: 同じピッチで時間が被るノーツがあれば追加しない
        new_end = note.start + note.duration
        for n in track.notes:
            if n.pitch == note.pitch:
                n_end = n.start + n.duration
                if not (new_end <= n.start + 1e-6 or note.start >= n_end - 1e-6):
                    return None

        track.notes.append(note)

        self.sort()
        self._bump()

        return note

    def remove_note(self, note):
        for track in self.tracks:
            if note in track.notes:
                track.notes.remove(note)

        self._bump()

    def sort(self):
        for track in self.tracks:
            track.notes.sort(
                key=lambda x: (
                    x.start,
                    x.pitch
                )
            )
            track.pedals.sort(key=lambda x: x.time)

    def copy_notes(self, notes):
        copied = []
        for note in notes:
            new_note = note.clone()
            new_note._original_track = getattr(note, '_original_track', 0)
            copied.append(new_note)
        return copied

    def paste_notes(self, notes, start_time, pitch_offset=0):
        if not notes:
            return []

        base_time = min(note.start for note in notes)
        created = []
        is_all_tracks = (self.filter_track is None)

        for note in notes:
            if is_all_tracks:
                t_idx = getattr(note, '_original_track', self.active_track())
                if t_idx >= len(self.tracks):
                    t_idx = self.active_track()
            else:
                t_idx = self.active_track()
                
            track = self.tracks[t_idx]

            new_note = Note(
                note.start - base_time + start_time,
                note.duration,
                max(0, min(127, note.pitch + pitch_offset)),
                note.velocity,
                getattr(note, 'channel', track.channel),
                getattr(note, 'lyric', "")
            )
            new_note._original_track = t_idx
            
            is_duplicate = False
            new_end = new_note.start + new_note.duration
            for n in track.notes:
                if n.pitch == new_note.pitch:
                    n_end = n.start + n.duration
                    if not (new_end <= n.start + 1e-6 or new_note.start >= n_end - 1e-6):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                track.notes.append(new_note)
                created.append(new_note)

        self.sort()
        self._bump()
        return created

    def copy_pedals(self, pedals):
        copied = []
        for pedal in pedals:
            new_pedal = copy.copy(pedal)
            new_pedal._original_track = getattr(pedal, '_original_track', 0)
            copied.append(new_pedal)
        return copied

    def paste_pedals(self, pedals, start_time):
        if not pedals:
            return []

        base_time = min(p.time for p in pedals)
        created = []
        is_all_tracks = (self.filter_track is None)

        for pedal in pedals:
            if is_all_tracks:
                t_idx = getattr(pedal, '_original_track', self.active_track())
                if t_idx >= len(self.tracks):
                    t_idx = self.active_track()
            else:
                t_idx = self.active_track()
                
            track = self.tracks[t_idx]
            
            new_pedal = PedalEvent(
                pedal.time - base_time + start_time,
                pedal.down
            )
            new_pedal._original_track = t_idx
            
            track.pedals.append(new_pedal)
            created.append(new_pedal)

        self.sort()
        self._bump()
        return created

    def save(self, path):
        midi = mido.MidiFile(
            ticks_per_beat=480,
            charset='utf-8'
        )

        ticks_per_beat = 480

        conductor = mido.MidiTrack()

        midi.tracks.append(conductor)

        events = []

        for t_sec, bpm in self.tempos:
            tick = int(
                round(
                    self.seconds_to_tick(
                        t_sec,
                        ticks_per_beat
                    )
                )
            )

            events.append(
                (
                    tick,
                    ("set_tempo", mido.bpm2tempo(bpm)),
                )
            )

        for t_sec, num, den in self.time_signatures:
            tick = int(
                round(
                    self.seconds_to_tick(
                        t_sec,
                        ticks_per_beat
                    )
                )
            )

            events.append(
                (
                    tick,
                    ("time_signature", num, den),
                )
            )

        events.sort(
            key=lambda x: x[0]
        )

        prev_tick = 0

        for tick, (kind, a, *rest) in events:
            delta = max(
                0,
                tick - prev_tick
            )

            if kind == "set_tempo":
                conductor.append(
                    mido.MetaMessage(
                        "set_tempo",
                        tempo=a,
                        time=delta
                    )
                )
            else:
                conductor.append(
                    mido.MetaMessage(
                        "time_signature",
                        numerator=a,
                        denominator=rest[0],
                        time=delta
                    )
                )

            prev_tick = tick

        # トラックに設定されたチャンネルを尊重して出力する。
        # 未設定(重複)の場合のみ空きチャンネルを自動割り当てする
        # (自動割り当てではチャンネル9=GMドラムを避ける)
        used_channels = set()

        for track_idx, track in enumerate(self.tracks):
            if not track.notes and not track.pedals:
                continue

            mtrack = mido.MidiTrack()
            midi.tracks.append(mtrack)

            if track.name:
                mtrack.append(
                    mido.MetaMessage(
                        "track_name",
                        name=track.name,
                        time=0
                    )
                )

            export_channel = getattr(track, "channel", 0)

            if (
                not isinstance(export_channel, int) or
                not (0 <= export_channel <= 15) or
                export_channel in used_channels
            ):
                export_channel = next(
                    (
                        c for c in range(16)
                        if c != 9 and c not in used_channels
                    ),
                    0
                )

            used_channels.add(export_channel)

            events = []

            for note in track.notes:
                start_tick = int(
                    round(
                        self.seconds_to_tick(
                            note.start,
                            ticks_per_beat
                        )
                    )
                )

                end_tick = int(
                    round(
                        self.seconds_to_tick(
                            note.start + note.duration,
                            ticks_per_beat
                        )
                    )
                )

                if end_tick <= start_tick:
                    end_tick = start_tick + max(1, int(round(ticks_per_beat / 480.0)))

                events.append(
                    (start_tick, 1, note)
                )

                events.append(
                    (end_tick, 0, note)
                )

            for pedal in track.pedals:
                tick = int(
                    round(
                        self.seconds_to_tick(
                            pedal.time,
                            ticks_per_beat
                        )
                    )
                )
                
                if pedal.down:
                    tick += 1

                events.append(
                    (tick, 2, pedal.down)
                )

            events.sort(
                key=lambda x: (
                    x[0],
                    x[1]
                )
            )

            current_tick = 0

            track_channel = getattr(track, "channel", export_channel)

            # ノートが出力に使うチャンネルを決定する:
            # - トラックの設定と異なるチャンネルを持つノーツはそのチャンネルを
            #   尊重する(1トラック内の複数チャンネルデータを保護)
            # - それ以外はトラックに割り当てられたチャンネルを使用する
            def resolve_note_channel(payload):
                ch = getattr(payload, "channel", None)

                if (
                    not isinstance(ch, int) or
                    not (0 <= ch <= 15) or
                    ch == track_channel
                ):
                    return export_channel

                return ch

            for tick, event_type, payload in events:
                delta = max(
                    0,
                    tick - current_tick
                )

                current_tick = tick

                if event_type == 1:
                    mtrack.append(
                        mido.Message(
                            "note_on",
                            note=max(0, min(127, payload.pitch)),
                            velocity=max(0, min(127, payload.velocity)),
                            channel=resolve_note_channel(payload),
                            time=delta
                        )
                    )
                elif event_type == 2:
                    mtrack.append(
                        mido.Message(
                            "control_change",
                            control=64,
                            value=(
                                127
                                if payload
                                else 0
                            ),
                            channel=export_channel,
                            time=delta
                        )
                    )
                else:
                    mtrack.append(
                        mido.Message(
                            "note_off",
                            note=max(0, min(127, payload.pitch)),
                            velocity=0,
                            channel=resolve_note_channel(payload),
                            time=delta
                        )
                    )

        midi.save(path)

    # ------------------------------------------------------------------
    # Synthesizer V Studio (.svp) 入出力
    # ------------------------------------------------------------------
    def _svp_tempo_marks(self):
        """書き出し用にテンポマークを整理する (時刻秒, BPM)。"""
        marks = []

        for t_sec, bpm in self.tempos:
            t = max(0.0, float(t_sec))

            if marks and abs(marks[-1][0] - t) < 1e-9:
                marks[-1] = (t, float(bpm))
            else:
                marks.append((t, float(bpm)))

        marks.sort(key=lambda x: x[0])

        if not marks or marks[0][0] > 1e-9:
            bpm0 = marks[0][1] if marks else 120.0
            marks.insert(0, (0.0, bpm0))

        return marks

    def _svp_build_converter(self, marks):
        """秒 <-> blick 変換関数をテンポマップから作る。"""
        positions = [m[0] for m in marks]
        cum_blick = [0.0]

        for i in range(len(marks) - 1):
            dt = marks[i + 1][0] - marks[i][0]
            cum_blick.append(
                cum_blick[-1] +
                dt * marks[i][1] / 60.0 *
                BLICKS_PER_QUARTER
            )

        def sec2blink(sec):
            i = bisect.bisect_right(positions, sec) - 1

            if i < 0:
                i = 0
                sec = positions[0]

            return (
                cum_blick[i] +
                (sec - positions[i]) *
                marks[i][1] / 60.0 *
                BLICKS_PER_QUARTER
            )

        def blick2sec(blick):
            i = bisect.bisect_right(cum_blick, blick) - 1

            if i < 0:
                i = 0
                blick = cum_blick[0]

            return (
                positions[i] +
                (blick - cum_blick[i]) *
                60.0 / marks[i][1]
            )

        return sec2blink, blick2sec

    def save_svp(self, path):
        """Synthesizer V Studio (.svp) 形式で書き出す。

        実際の .svp ディスク形式 (version数値 / time.tempo・time.meter /
        tracks[].mainGroup 埋め込み / onset・pitch キー) で出力する。
        時間単位は blick (1四分音符 = 705,600,000 blick)。
        ノーツの歌詞は lyrics フィールドに書き込まれる。
        """
        self._ensure_caches()

        marks = self._svp_tempo_marks()
        sec2blink, _ = self._svp_build_converter(marks)

        tempo_out = [
            {
                "position": int(round(sec2blink(t))),
                "bpm": float(bpm),
            }
            for t, bpm in marks
        ]

        # 拍子は「小節番号(index)」指定なので、時刻から小節番号へ変換する
        sig_entries = sorted(
            (
                (max(0.0, float(t)), int(num), int(den))
                for t, num, den in self.time_signatures
            ),
            key=lambda x: x[0],
        )

        if not sig_entries or sig_entries[0][0] > 1e-9:
            sig_entries.insert(0, (0.0, 4, 4))

        meter_out = []
        bar_index = 0
        pos_beat = 0.0
        prev_bar_beats = None

        for i, (t, num, den) in enumerate(sig_entries):
            if i == 0:
                cur_index = 0
            else:
                target_beat = sec2blink(t) / BLICKS_PER_QUARTER
                delta_beats = target_beat - pos_beat

                if delta_beats < prev_bar_beats * 0.5:
                    continue

                delta_bars = max(
                    1,
                    int(math.floor(delta_beats / prev_bar_beats + 0.5)),
                )
                bar_index += delta_bars
                pos_beat += delta_bars * prev_bar_beats
                cur_index = bar_index

            meter_out.append(
                {
                    "index": cur_index,
                    "numerator": num,
                    "denominator": den,
                }
            )

            prev_bar_beats = num * 4.0 / den

        if not meter_out:
            meter_out = [{"index": 0, "numerator": 4, "denominator": 4}]

        default_params = {
            name: {"mode": "cubic", "points": []}
            for name in (
                "pitchDelta",
                "vibratoEnv",
                "loudness",
                "tension",
                "breathiness",
                "voicing",
                "gender",
            )
        }

        tracks_out = []

        for index, track in enumerate(self.tracks):
            group_id = str(uuidlib.uuid4())

            notes_out = []
            for note in sorted(track.notes, key=lambda n: n.start):
                t_blick = int(round(sec2blink(note.start)))
                d_blick = max(
                    1,
                    int(round(sec2blink(note.start + note.duration))) -
                    t_blick,
                )

                notes_out.append(
                    {
                        "onset": max(0, t_blick),
                        "duration": d_blick,
                        "lyrics": getattr(note, "lyric", ""),
                        "phonemes": "",
                        "pitch": int(max(0, min(127, int(note.pitch)))),
                        "attributes": {},
                    }
                )

            tracks_out.append(
                {
                    "name": track.name or f"Track {index + 1}",
                    "dispColor": "ff7db235",
                    "dispOrder": index,
                    "renderEnabled": True,
                    "mixer": {
                        "gainDecibel": 0.0,
                        "pan": 0.0,
                        "mute": False,
                        "solo": False,
                        "display": True,
                    },
                    "mainGroup": {
                        "name": "main",
                        "uuid": group_id,
                        "parameters": dict(default_params),
                        "notes": notes_out,
                    },
                    "mainRef": {
                        "groupID": group_id,
                        "blickOffset": 0,
                        "pitchOffset": 0,
                        "isInstrumental": False,
                        "database": {"name": "", "language": "", "phoneset": ""},
                        "audio": {"filename": "", "duration": 0.0},
                        "dictionary": "",
                        "voice": {},
                    },
                    "groups": [],
                }
            )

        data = {
            "version": 113,
            "time": {
                "meter": meter_out,
                "tempo": tempo_out,
            },
            "library": [],
            "tracks": tracks_out,
            "renderConfig": {
                "destination": "./",
                "filename": "untitled",
                "numChannels": 1,
                "aspirationFormat": "noAspiration",
                "bitDepth": 16,
                "sampleRate": 44100,
                "exportMixDown": True,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def _svp_select_project(text):
        """NUL区切りで複数JSONが連結されていても対応する。

        Synthesizer V Studio 2 は v1/v2 のデータを同一ファイルに
        保存することがあるため、version が最も大きいものを採用する。
        """
        best = None
        best_version = None

        for chunk in text.split("\x00"):
            chunk = chunk.strip().lstrip("\ufeff")

            if not chunk:
                continue

            try:
                doc = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if not isinstance(doc, dict):
                continue

            try:
                version_num = float(doc.get("version"))
            except (TypeError, ValueError):
                version_num = float("-inf")

            if best is None or version_num > best_version:
                best = doc
                best_version = version_num

        if best is None:
            raise ValueError(
                tr("有効なプロジェクトデータ(.svp JSON)が見つかりません", "No valid project data (.svp JSON) found")
            )

        return best

    def load_svp(self, path):
        """Synthesizer V Studio (.svp) を読み込む。

        実際のディスク形式 (time.tempo / time.meter /
        tracks[].mainGroup / onset・pitch キー) を読む。
        library 参照 (tracks[].groups) や旧来の別表記にも対応し、
        歌詞(lyrics)はノーツごとに復元される。
        Synthesizer V Studio 1 / 2 のどちらで保存されたファイルでも
        読み込める (gzip圧縮 / 先頭BOM / NUL区切りの複数JSONに対応)。
        """
        with open(path, "rb") as f:
            raw = f.read()

        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)

        data = self._svp_select_project(
            raw.decode("utf-8", errors="replace")
        )

        def find_list(*keys):
            cur = data

            for key in keys:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return None

            return cur if isinstance(cur, list) else None

        tempo_raw = (
            find_list("time", "tempo") or
            find_list("tempos") or
            find_list("timeAxis", "tempo") or
            []
        )

        ts_raw = (
            find_list("time", "meter") or
            find_list("timeSignatures") or
            find_list("timeAxis", "measure") or
            []
        )

        tempo_marks = []

        for mark in tempo_raw:
            try:
                pos = float(mark.get("position", 0))
                bpm = float(
                    mark.get("bpm", mark.get("beatPerMinute", 120))
                )
            except (TypeError, ValueError, AttributeError):
                continue

            if bpm <= 0:
                continue

            tempo_marks.append((pos, bpm))

        if not tempo_marks:
            tempo_marks = [(0.0, 120.0)]

        tempo_marks.sort(key=lambda x: x[0])

        if tempo_marks[0][0] > 0:
            tempo_marks.insert(0, (0.0, tempo_marks[0][1]))

        positions = [m[0] for m in tempo_marks]
        cum_sec = [0.0]

        for i in range(len(tempo_marks) - 1):
            db = tempo_marks[i + 1][0] - tempo_marks[i][0]
            cum_sec.append(
                cum_sec[-1] +
                db / BLICKS_PER_QUARTER *
                60.0 / tempo_marks[i][1]
            )

        def blick2sec(blick):
            i = bisect.bisect_right(positions, blick) - 1

            if i < 0:
                i = 0
                blick = positions[0]

            return (
                cum_sec[i] +
                (blick - positions[i]) /
                BLICKS_PER_QUARTER *
                60.0 / tempo_marks[i][1]
            )

        self.tempos = [
            (round(blick2sec(pos), 9), bpm)
            for pos, bpm in tempo_marks
        ]
        self.bpm = self.tempos[0][1]

        # 拍子: position(blick) か index(小節番号) のどちらかで与えられる
        time_signatures = []

        if any(isinstance(s, dict) and "position" in s for s in ts_raw):
            for sig in ts_raw:
                try:
                    pos = float(sig.get("position", 0))
                    num = int(sig.get("numerator", 4))
                    den = int(sig.get("denominator", 4))
                except (TypeError, ValueError, AttributeError):
                    continue

                time_signatures.append((blick2sec(pos), num, den))
        else:
            # index は小節番号。テンポマップに沿って時刻へ変換する
            cur_bar = 0
            cur_pos_beat = 0.0
            cur_bar_beats = None

            for sig in ts_raw:
                try:
                    idx = int(sig.get("index", 0))
                    num = int(sig.get("numerator", 4))
                    den = int(sig.get("denominator", 4))
                except (TypeError, ValueError, AttributeError):
                    continue

                if idx > cur_bar and cur_bar_beats is not None:
                    cur_pos_beat += (idx - cur_bar) * cur_bar_beats
                    cur_bar = idx

                time_signatures.append((blick2sec(cur_pos_beat * BLICKS_PER_QUARTER), num, den))

                cur_bar_beats = num * 4.0 / den

        self.time_signatures = time_signatures or [(0.0, 4, 4)]

        library = {}

        for key in ("library", "noteGroups"):
            for group in (data.get(key) or []):
                if isinstance(group, dict) and group.get("uuid"):
                    library.setdefault(group["uuid"], group)

        new_tracks = []

        for track_data in (data.get("tracks") or []):
            if not isinstance(track_data, dict):
                continue

            main_group = track_data.get("mainGroup")

            refs = []

            main_ref = track_data.get("mainRef")

            if isinstance(main_ref, dict):
                refs.append(main_ref)

            for ref in (track_data.get("groups") or []):
                if isinstance(ref, dict):
                    refs.append(ref)

            notes = []

            for ref in refs:
                group_id = ref.get("groupID")
                group = None

                if (
                    isinstance(main_group, dict) and
                    main_group.get("uuid") == group_id
                ):
                    group = main_group

                if group is None:
                    group = library.get(group_id)

                if group is None:
                    continue

                offset = float(
                    ref.get("blickOffset", ref.get("timeOffset", 0)) or 0
                )
                pitch_offset = float(ref.get("pitchOffset", 0) or 0)

                for note_data in (group.get("notes") or []):
                    try:
                        onset = offset + float(
                            note_data.get("onset", note_data.get("t", 0))
                        )
                        duration = float(note_data.get("duration", 0))
                        number = note_data.get(
                            "pitch",
                            note_data.get("number"),
                        )

                        if number is None or duration <= 0 or onset < 0:
                            continue

                        pitch = int(round(float(number) + pitch_offset))
                    except (TypeError, ValueError):
                        continue

                    pitch = max(0, min(127, pitch))

                    start = blick2sec(onset)
                    end = blick2sec(onset + duration)
                    duration_sec = max(1e-3, end - start)

                    lyric = str(note_data.get("lyrics", "") or "")

                    notes.append(
                        Note(
                            start,
                            duration_sec,
                            pitch,
                            100,
                            0,
                            lyric,
                        )
                    )

            if notes:
                notes.sort(key=lambda x: (x.start, x.pitch))

                name = (
                    track_data.get("name") or
                    tr(f"トラック {len(new_tracks) + 1}", f"Track {len(new_tracks) + 1}")
                )

                new_tracks.append(
                    Track(
                        name=name,
                        notes=notes,
                        channel=0,
                    )
                )

        if not new_tracks:
            new_tracks = [Track()]

        self.tracks = new_tracks
        self.filter_track = 0
        self.beat_phase = 0.0
        self.has_file = True

        self.sort()
        self.extra_state = {}
        self._refresh_caches()
        self._bump()


    def load(self, path):
        try:
            midi = mido.MidiFile(path, charset='utf-8', clip=True)
        except UnicodeDecodeError:
            try:
                midi = mido.MidiFile(path, charset='cp932', clip=True)
            except UnicodeDecodeError:
                midi = mido.MidiFile(path, charset='latin-1', clip=True)

        ticks_per_beat = midi.ticks_per_beat

        tempo_ticks = []

        for track in midi.tracks:
            abs_tick = 0

            for msg in track:
                abs_tick += msg.time

                if msg.type == "set_tempo":
                    tempo_ticks.append(
                        (abs_tick, msg.tempo)
                    )

        tempo_ticks.sort()

        if not tempo_ticks:
            tempo_ticks = [(0, 500000)]

        if tempo_ticks[0][0] != 0:
            tempo_ticks.insert(
                0,
                (0, tempo_ticks[0][1])
            )

        tick_arr = [
            t
            for t, _ in tempo_ticks
        ]

        tempo_us = [
            te
            for _, te in tempo_ticks
        ]

        sec_arr = [0.0]

        for i in range(len(tempo_ticks) - 1):
            sec_arr.append(
                sec_arr[-1] +
                (
                    tick_arr[i + 1] -
                    tick_arr[i]
                ) *
                tempo_us[i] /
                (ticks_per_beat * 1e6)
            )

        def ticks_to_seconds(tick):
            i = bisect.bisect_right(
                tick_arr,
                tick
            ) - 1

            i = max(0, i)

            return (
                sec_arr[i] +
                (
                    tick -
                    tick_arr[i]
                ) *
                tempo_us[i] /
                (ticks_per_beat * 1e6)
            )

        self.tempos = [
            (
                ticks_to_seconds(tick),
                mido.tempo2bpm(tempo)
            )
            for tick, tempo in tempo_ticks
        ]

        self.bpm = self.tempos[0][1]

        self.beat_phase = 0.0

        sig_ticks = []

        for track in midi.tracks:
            abs_tick = 0

            for msg in track:
                abs_tick += msg.time

                if msg.type == "time_signature":
                    sig_ticks.append(
                        (
                            abs_tick,
                            msg.numerator,
                            msg.denominator
                        )
                    )

        sig_ticks.sort()

        if not sig_ticks:
            sig_ticks = [(0, 4, 4)]

        if sig_ticks[0][0] != 0:
            sig_ticks.insert(
                0,
                (0, sig_ticks[0][1], sig_ticks[0][2])
            )

        self.time_signatures = [
            (
                ticks_to_seconds(tick),
                max(1, int(numerator)),
                max(1, int(denominator))
            )
            for tick, numerator, denominator in sig_ticks
        ]

        new_tracks = []

        for mtrack in midi.tracks:
            notes = []
            pedals = []
            active = {}
            abs_tick = 0

            for msg in mtrack:
                abs_tick += msg.time

                if (
                    msg.type == "note_on" and
                    msg.velocity > 0
                ):
                    active[(msg.channel, msg.note)] = (
                        abs_tick,
                        msg.velocity
                    )

                elif (
                    msg.type == "note_off" or
                    (
                        msg.type == "note_on" and
                        msg.velocity == 0
                    )
                ):
                    info = active.pop(
                        (msg.channel, msg.note),
                        None
                    )

                    if info is not None:
                        start_tick, velocity = info

                        start = ticks_to_seconds(start_tick)

                        min_tick_diff = max(1, ticks_per_beat / 480.0)
                        min_duration = ticks_to_seconds(start_tick + min_tick_diff) - start

                        duration = max(
                            ticks_to_seconds(abs_tick) - start,
                            min_duration
                        )

                        notes.append(
                            Note(
                                start,
                                duration,
                                msg.note,
                                velocity,
                                msg.channel
                            )
                        )

                elif (
                    msg.type == "control_change" and
                    msg.control == 64
                ):
                    pedals.append(
                        PedalEvent(
                            ticks_to_seconds(abs_tick),
                            msg.value >= 64
                        )
                    )

            if notes or pedals:
                notes.sort(
                    key=lambda x: (
                        x.start,
                        x.pitch
                    )
                )

                name = (
                    mtrack.name or
                    tr(f"トラック {len(new_tracks) + 1}", f"Track {len(new_tracks) + 1}")
                )

                track_channel = (
                    notes[0].channel
                    if notes
                    else len(new_tracks)
                )

                new_tracks.append(
                    Track(
                        name=name,
                        notes=notes,
                        pedals=pedals,
                        channel=track_channel
                    )
                )

        if not new_tracks:
            new_tracks = [Track()]

        self.tracks = new_tracks
        self.filter_track = 0
        self.has_file = True

        self.sort()
        self.extra_state = {}
        self._refresh_caches()
        self._bump()
