from dataclasses import dataclass, field
import copy
import bisect
import mido

@dataclass
class Note:
    start: float
    duration: float
    pitch: int
    velocity: int = 100
    channel: int = 0

    def clone(self):
        return copy.copy(self)

@dataclass
class PedalEvent:
    time: float
    down: bool

@dataclass
class Track:
    name: str = "トラック 1"
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
                            getattr(note, 'channel', 0)
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
                    channel=track.channel
                )
                for track in self.tracks
            ],
            "tempos": list(self.tempos),
            "time_signatures": list(self.time_signatures),
        }

    def restore(self, snap):
        self.tracks = [
            Track(
                name=track.name,
                notes=[
                    Note(
                        note.start,
                        note.duration,
                        note.pitch,
                        note.velocity,
                        getattr(note, 'channel', 0)
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
                        getattr(note, 'channel', 0)
                    )
                )
            else:
                result.append(note.clone())

        return result

    def visible_extended_notes(self):
        if (
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
            name or f"トラック {len(self.tracks) + 1}"
        )

        if self.filter_track is None and self.tracks:
            track.pedals = [PedalEvent(ev.time, ev.down) for ev in self.tracks[0].pedals]

        self.tracks.append(track)

        self._bump()

        return track

    def set_filter_track(self, index):
        self.filter_track = index

    def set_base_tempo(self, bpm):
        self.tempos[0] = (0.0, float(bpm))
        self.bpm = float(bpm)
        self._refresh_caches()

    def set_beat_phase(self, beats):
        self.beat_phase = float(beats)

    def add_tempo(self, time, bpm):
        time = max(0.0, float(time))

        out = []
        replaced = False

        for t, b in self.tempos:
            if abs(t - time) < 1e-6:
                out.append((time, float(bpm)))
                replaced = True
            else:
                out.append((t, b))

        if not replaced:
            out.append((time, float(bpm)))
            out.sort(key=lambda x: x[0])

        self.tempos = out
        self.bpm = self.tempos[0][1]
        self._refresh_caches()

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
            num = max(1, int(sigs[i][1]))

            seg_beats = (
                self.time_to_beat(sigs[i + 1][0]) -
                self.time_to_beat(sigs[i][0])
            )

            cum_measure.append(
                cum_measure[-1] +
                int(
                    round(
                        seg_beats / num
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

        seg_start = self.time_to_beat(
            sigs[i][0]
        )

        off = beat - seg_start

        return (
            self._sig_cum_measure[i] + int(off // num),
            int(off) % num,
            num
        )

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

    def copy_notes(self, notes):
        return [
            note.clone()
            for note in notes
        ]

    def paste_notes(self, notes, start_time, pitch_offset=0):
        if not notes:
            return []

        base_time = min(
            note.start
            for note in notes
        )

        track = self.tracks[
            self.active_track()
        ]

        created = []

        for note in notes:
            new_note = Note(
                start_time + (note.start - base_time),
                note.duration,
                note.pitch + pitch_offset,
                note.velocity,
                getattr(note, 'channel', track.channel)
            )

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

            # Assign a unique channel per track, skipping channel 9 (drums in GM)
            export_channel = track_idx % 15
            if export_channel >= 9:
                export_channel += 1

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
                            note=payload.pitch,
                            velocity=payload.velocity,
                            channel=export_channel,
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
                            note=payload.pitch,
                            velocity=0,
                            channel=export_channel,
                            time=delta
                        )
                    )

        midi.save(path)

    def load(self, path):
        midi = mido.MidiFile(path, charset='utf-8')

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

                        duration = max(
                            ticks_to_seconds(abs_tick) -
                            start,
                            0.05
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
                    f"トラック {len(new_tracks) + 1}"
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
        self._refresh_caches()
        self._bump()
