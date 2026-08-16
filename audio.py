import threading
import time
import bisect
import numpy as np
import sounddevice as sd
import librosa
import midiout

class AudioData:
    def __init__(self):
        self.y = None
        self.sr = 44100
        self.position = 0.0
        self.playing = False
        self.paused = False

        self._started_at = 0.0
        self._start_position = 0.0
        self._thread = None
        self._play_gen = 0
        self._latency = 0.0

        self.midi = None
        self.midi_phase = {}
        self.midi_active = set()

        self._r_notes = []
        self._r_order = []
        self._r_starts = []
        self._r_ends = []
        self._r_max_dur = 0.0

        self._render_t = None
        self._render_phase = None
        self._render_sin = None
        self._render_wave = None
        self._render_env = None

        self.volume = 0.5
        self.offset = 0.0
        self.preview_pitch = None
        self.preview_until = 0.0

        self.output_device = "internal"
        self._midi_out = None
        self._midi_out_preview_pitch = None
        self._midi_lock = threading.RLock()

        self._preview_target = {
            "pitch": None,
            "trigger": 0
        }
        self._preview_thread = None
        self._preview_stream = None
        self._preview_stop_event = threading.Event()

        self._render_progress = 0.0

        self._tl_version = -1
        self._tl_dur_audio = None
        self._tl_dur_offset = None
        self._tl_duration = 1.0

        self.file_path = None
        self._midi_cache_dirty = False

    def set_midi(self, midi):
        self.midi = midi
        self._midi_cache_dirty = True

    def invalidate_midi_cache(self):
        self._midi_cache_dirty = True

    def set_output_device(self, device):
        with self._midi_lock:
            self.output_device = device

            if device == "internal":
                self._close_midi_out()
                return

            if not self._open_midi_out():
                self.output_device = "internal"

    def _open_midi_out(self):
        with self._midi_lock:
            if self._midi_out is not None:
                return True

            device = midiout.MidiOutDevice()

            if not device.open(self.output_device):
                return False

            self._midi_out = device

            return True

    def _close_midi_out(self):
        with self._midi_lock:
            if self._midi_out is None:
                return

            try:
                self._midi_out.close()
            except Exception:
                pass

            self._midi_out = None
            self._midi_out_preview_pitch = None

    def _silence_midi_out(self):
        with self._midi_lock:
            if self._midi_out is None:
                return

            try:
                self._midi_out.all_notes_off()
            except Exception:
                pass

    def load(self, path):
        self.stop()

        self.y, self.sr = librosa.load(
            path,
            sr=None,
            mono=True
        )

        self.file_path = path
        self.position = 0.0
        self._start_position = 0.0
        self.paused = False

        return len(self.y) / self.sr

    def duration(self):
        if self.y is None:
            return 0.0

        return len(self.y) / self.sr

    def timeline_duration(self):
        if self.midi is None:
            return max(
                0.0,
                self.duration() + self.offset
            )

        audio_dur = self.duration()

        version = self.midi.mutation_version

        if (
            version != self._tl_version or
            audio_dur != self._tl_dur_audio or
            self.offset != self._tl_dur_offset
        ):
            end = audio_dur + self.offset

            end = max(
                end,
                self.midi.max_extended_end()
            )

            self._tl_version = version
            self._tl_dur_audio = audio_dur
            self._tl_dur_offset = self.offset
            self._tl_duration = max(0.0, end)

        return self._tl_duration

    def max_position(self):
        return max(
            0.0,
            self.timeline_duration()
        )

    def play(self):
        if self.playing:
            return

        self.playing = True
        self.paused = False

        self._silence_preview()

        self._start_position = self.position
        self._started_at = time.perf_counter()
        self._latency = 0.0
        self._render_progress = 0.0

        self._play_gen += 1
        gen = self._play_gen

        self._thread = threading.Thread(
            target=self._play_worker,
            args=(gen,),
            daemon=True
        )

        self._thread.start()

    def _play_worker(self, gen):
        if gen != self._play_gen:
            return

        self._play_with_midi(
            self._start_position,
            gen
        )

    def _refresh_midi_cache(self):
        self.midi_phase.clear()
        self.midi_active.clear()

        note_list = (
            list(self.midi.visible_extended_notes())
            if self.midi is not None
            else []
        )

        order = sorted(
            range(len(note_list)),
            key=lambda i: note_list[i].start
        )

        self._r_notes = note_list
        self._r_order = order
        self._r_starts = [
            note_list[i].start
            for i in order
        ]

        self._r_ends = [
            note_list[i].start +
            note_list[i].duration
            for i in order
        ]

        self._r_max_dur = max(
            (
                self._r_ends[i] -
                self._r_starts[i]
                for i in range(len(order))
            ),
            default=0.0
        )

        self._midi_cache_dirty = False

    def _play_with_midi(self, start_position, gen):
        sample_rate = self.sr
        block_size = 1024

        if self.output_device != "internal":
            if not self._open_midi_out():
                self.output_device = "internal"

        use_midi_out = self._midi_out is not None

        if use_midi_out:
            try:
                self._midi_out.all_notes_off()
            except Exception:
                pass

        total_time = self.max_position()

        total_samples = max(
            0,
            int(
                (total_time - start_position) *
                sample_rate
            )
        )

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size
            )
        except Exception:
            if gen == self._play_gen:
                self.playing = False

            self._silence_midi_out()

            return

        self._latency = float(
            getattr(
                stream,
                "latency",
                0.0
            ) or 0.0
        )

        self._refresh_midi_cache()

        midi_worker = None

        if use_midi_out:
            midi_worker = threading.Thread(
                target=self._midi_out_worker,
                args=(
                    start_position,
                    total_time,
                    gen,
                    self._latency
                ),
                daemon=True
            )

            midi_worker.start()

        sample_position = 0

        try:
            stream.start()

            while (
                self.playing and
                sample_position < total_samples
            ):
                if self._midi_cache_dirty:
                    self._refresh_midi_cache()

                current_time = (
                    start_position +
                    sample_position / sample_rate
                )

                audio_index = int(
                    (current_time - self.offset) *
                    sample_rate
                )

                if (
                    self.y is not None and
                    0 <= audio_index < len(self.y)
                ):
                    end_index = min(
                        len(self.y),
                        audio_index + block_size
                    )

                    block = self.y[
                        audio_index:end_index
                    ]

                    if block.dtype != np.float32:
                        block = block.astype(
                            np.float32
                        )
                    else:
                        block = block.copy()

                    if len(block) < block_size:
                        block = np.pad(
                            block,
                            (
                                0,
                                block_size - len(block)
                            )
                        )
                else:
                    block = np.zeros(
                        block_size,
                        dtype=np.float32
                    )

                block *= self.volume

                midi_block = self._render_midi(
                    current_time,
                    block_size,
                    sample_rate
                )

                block += midi_block * 0.35

                peak = np.max(np.abs(block))

                if peak > 0.98:
                    block /= peak

                stream.write(block)

                sample_position += block_size

                self._render_progress = (
                    sample_position /
                    sample_rate
                )

        finally:
            try:
                stream.stop()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

            if gen == self._play_gen:
                if sample_position >= total_samples:
                    self.position = self.max_position()

                self.playing = False

                if (
                    midi_worker is not None and
                    midi_worker.is_alive()
                ):
                    midi_worker.join(timeout=0.2)

                self._silence_midi_out()

            self.midi_phase.clear()
            self.midi_active.clear()
            self.preview_pitch = None
            self.preview_until = 0.0
            self._midi_out_preview_pitch = None

    def _build_midi_events(self, notes, start_position, total_time):
        events = []

        for note in notes:
            note_start = note.start
            note_end = note.start + note.duration

            if note_end <= start_position:
                continue

            if note_start >= total_time:
                continue

            channel = getattr(note, 'channel', 0)

            events.append((
                max(
                    note_start,
                    start_position
                ) - start_position,
                "on",
                note.pitch,
                note.velocity,
                channel
            ))

            events.append((
                max(
                    note_end,
                    start_position
                ) - start_position,
                "off",
                note.pitch,
                0,
                channel
            ))

        if self.midi is not None:
            if self.midi.filter_track is not None:
                index = self.midi.filter_track

                if 0 <= index < len(self.midi.tracks):
                    tracks = [
                        self.midi.tracks[index]
                    ]
                else:
                    tracks = []
            else:
                tracks = self.midi.tracks

            pedals = []

            for track in tracks:
                for pedal in track.pedals:
                    pedals.append((pedal, track.channel))

            pedals.sort(
                key=lambda e: e[0].time
            )

            state = False

            for pedal, ch in pedals:
                if pedal.time <= start_position:
                    state = pedal.down
                else:
                    break

            if state:
                events.append((
                    0.0,
                    "pedal",
                    True,
                    0,
                    0
                ))

            for pedal, ch in pedals:
                if pedal.time <= start_position + 1e-6:
                    continue

                if pedal.time >= total_time:
                    continue

                events.append((
                    pedal.time - start_position,
                    "pedal",
                    pedal.down,
                    0,
                    ch
                ))

        events.sort(
            key=lambda ev: (
                ev[0],
                0
                if ev[1] == "off"
                else (
                    1
                    if ev[1] == "on"
                    else 2
                )
            )
        )

        return events

    def _midi_out_worker(
        self,
        start_position,
        total_time,
        gen,
        latency
    ):
        try:
            sounding = set()

            song_base = start_position

            wall_base = time.perf_counter() + latency

            events = self._build_midi_events(
                self._r_notes,
                song_base,
                total_time
            )

            index = 0

            while True:
                if (
                    not self.playing or
                    gen != self._play_gen
                ):
                    return

                if self._midi_cache_dirty:
                    now = time.perf_counter()

                    elapsed = now - wall_base

                    if elapsed > 0.0:
                        new_song = song_base + elapsed

                        if new_song < start_position:
                            new_song = start_position
                        else:
                            wall_base = now

                        song_base = new_song

                    self._refresh_midi_cache()

                    device = self._midi_out

                    if device is not None and sounding:
                        try:
                            device.all_notes_off()
                        except Exception:
                            pass

                        sounding.clear()

                    events = self._build_midi_events(
                        self._r_notes,
                        song_base,
                        total_time
                    )

                    index = 0

                    continue

                if index >= len(events):
                    return

                relative, kind, pitch, velocity, channel = events[index]

                target = relative - (
                    time.perf_counter() - wall_base
                )

                if target > 0.0:
                    time.sleep(min(
                        0.002,
                        target
                    ))

                    continue

                device = self._midi_out

                if device is None:
                    return

                try:
                    if kind == "on":
                        device.note_on(pitch, velocity, channel)
                        sounding.add((pitch, channel))
                    elif kind == "pedal":
                        device.control_change(
                            64,
                            127 if velocity else 0,
                            channel
                        )
                    else:
                        device.note_off(pitch, channel)
                        sounding.discard((pitch, channel))
                except Exception:
                    pass

                index += 1
        finally:
            if self._midi_out is not None:
                try:
                    self._midi_out.all_notes_off()
                except Exception:
                    pass

    def _render_midi(self, start_time, length, sample_rate):
        output = np.zeros(
            length,
            dtype=np.float32
        )

        if self._midi_out is not None:
            self._midi_out_preview(
                start_time,
                length,
                sample_rate
            )

            return output

        if (
            self.midi is None and
            self.preview_pitch is None
        ):
            return output

        notes = self._r_notes
        order = self._r_order

        end_time = start_time + length / sample_rate

        if (
            self._render_t is None or
            self._render_t.shape[0] != length
        ):
            self._render_t = (
                np.arange(length, dtype=np.float32) /
                sample_rate
            )

            self._render_phase = np.empty(
                length,
                dtype=np.float32
            )

            self._render_sin = np.empty(
                length,
                dtype=np.float32
            )

            self._render_wave = np.empty(
                length,
                dtype=np.float32
            )

            self._render_env = np.empty(
                length,
                dtype=np.float32
            )

        t = self._render_t
        phase_buf = self._render_phase
        sin_buf = self._render_sin
        wave_buf = self._render_wave
        env_buf = self._render_env

        if order and self._r_starts:
            win_start = start_time - self._r_max_dur - 0.001

            i0 = bisect.bisect_left(
                self._r_starts,
                win_start
            )

            i1 = bisect.bisect_right(
                self._r_starts,
                end_time
            )

            iterable = (
                (
                    order[k],
                    notes[order[k]]
                )
                for k in range(i0, i1)
            )
        else:
            iterable = enumerate(notes)

        attack_length = max(
            1,
            int(sample_rate * 0.001)
        )

        release_length = max(
            1,
            int(sample_rate * 0.04)
        )

        for index, note in iterable:
            note_start = note.start
            note_end = note.start + note.duration

            if note_end <= start_time:
                continue

            if note_start >= end_time:
                continue

            frequency = 440.0 * (
                2.0 ** ((note.pitch - 69) / 12.0)
            )

            begin = max(
                0,
                int((note_start - start_time) * sample_rate)
            )

            finish = min(
                length,
                int((note_end - start_time) * sample_rate)
            )

            if finish <= begin:
                continue

            phase = self.midi_phase.get(index, 0.0)

            count = finish - begin

            np.multiply(
                t[:count],
                2.0 * np.pi * frequency,
                out=phase_buf[:count]
            )

            phase_buf[:count] += phase

            np.sin(
                phase_buf[:count],
                out=sin_buf[:count]
            )

            np.sign(
                sin_buf[:count],
                out=wave_buf[:count]
            )

            wave_buf[:count] *= 0.6

            env_buf[:count] = 1.0

            if note_start >= start_time:
                attack = min(
                    attack_length,
                    count
                )

                env_buf[:attack] = np.linspace(
                    0.0,
                    1.0,
                    attack
                )

            if note_end <= end_time:
                release = min(
                    release_length,
                    count
                )

                env_buf[count - release:count] *= np.linspace(
                    1.0,
                    0.0,
                    release
                )

            wave_buf[:count] *= env_buf[:count]

            velocity = max(
                0.0,
                min(
                    1.0,
                    note.velocity / 127.0
                )
            )

            wave_buf[:count] *= velocity * 0.28

            output[begin:finish] += wave_buf[:count]

            self.midi_phase[index] = (
                phase +
                2.0 *
                np.pi *
                frequency *
                count /
                sample_rate
            ) % (2.0 * np.pi)

        if (
            self.preview_pitch is not None and
            time.perf_counter() < self.preview_until
        ):
            frequency = 440.0 * (
                2.0 ** ((self.preview_pitch - 69) / 12.0)
            )

            t = (
                np.arange(length, dtype=np.float32) /
                sample_rate
            )

            phase_array = (
                2.0 *
                np.pi *
                frequency *
                t
            )

            wave = (
                np.sign(
                    np.sin(
                        phase_array
                    )
                ) *
                0.6
            )

            attack_length = max(
                1,
                int(sample_rate * 0.008)
            )

            release_length = max(
                1,
                int(sample_rate * 0.04)
            )

            envelope = np.ones(
                length,
                dtype=np.float32
            )

            attack = min(
                attack_length,
                length
            )

            envelope[:attack] = np.linspace(
                0.0,
                1.0,
                attack
            )

            release = min(
                release_length,
                length
            )

            envelope[-release:] *= np.linspace(
                1.0,
                0.0,
                release
            )

            wave *= envelope

            output += wave * 0.28

        return output

    def _midi_out_preview(self, start_time, length, sample_rate):
        if (
            self.preview_pitch is None or
            time.perf_counter() >= self.preview_until
        ):
            if self._midi_out_preview_pitch is not None:
                try:
                    self._midi_out.note_off(
                        self._midi_out_preview_pitch
                    )
                except Exception:
                    pass

                self._midi_out_preview_pitch = None

            return

        pitch = self.preview_pitch

        if pitch != self._midi_out_preview_pitch:
            if self._midi_out_preview_pitch is not None:
                try:
                    self._midi_out.note_off(
                        self._midi_out_preview_pitch
                    )
                except Exception:
                    pass

            try:
                self._midi_out.note_on(
                    pitch,
                    100
                )
            except Exception:
                pass

            self._midi_out_preview_pitch = pitch

    def preview_note(self, pitch, duration=0.25):
        if self.playing:
            # Brief preview during playback to confirm key press
            self.preview_pitch = pitch
            self.preview_until = time.perf_counter() + 0.05
            return

        self._ensure_preview_thread()

        self._preview_target["pitch"] = pitch
        self._preview_target["trigger"] = self._preview_target.get("trigger", 0) + 1

    def _ensure_preview_thread(self):
        if (
            self._preview_thread is not None and
            self._preview_thread.is_alive()
        ):
            return

        self._preview_thread = threading.Thread(
            target=self._preview_worker,
            daemon=True
        )

        self._preview_thread.start()

    def _preview_worker(self):
        while True:
            if self._preview_stop_event.is_set():
                return

            if self.playing:
                time.sleep(0.01)
                continue

            if self.output_device != "internal":
                if self.playing:
                    time.sleep(0.01)
                    continue

                if not self._open_midi_out():
                    self.output_device = "internal"
                    continue

                self._preview_worker_midi()
            else:
                self._preview_worker_stream()

    def _preview_worker_stream(self):
        while True:
            if self._preview_stop_event.is_set():
                return

            if self.playing:
                time.sleep(0.01)
                continue

            if self.output_device != "internal":
                return

            current_pitch = None
            current_trigger = 0
            t0 = 0.0
            phase0 = 0.0

            def callback(outdata, frames, time_info, status):
                nonlocal current_pitch, current_trigger, t0, phase0

                pitch = (
                    self._preview_target["pitch"]
                )
                trigger = (
                    self._preview_target.get("trigger", 0)
                )

                if pitch != current_pitch or trigger != current_trigger:
                    current_pitch = pitch
                    current_trigger = trigger
                    t0 = 0.0
                    phase0 = 0.0

                n = frames

                if pitch is None:
                    outdata[:] = 0.0
                    return

                sr = self.sr

                freq = 440.0 * (
                    2.0 ** ((pitch - 69) / 12.0)
                )

                times = (
                    t0 +
                    np.arange(
                        n,
                        dtype=np.float32
                    ) / sr
                )

                ph = (
                    phase0 +
                    2.0 *
                    np.pi *
                    freq *
                    (
                        np.arange(
                            n,
                            dtype=np.float32
                        ) / sr
                    )
                )

                wave = (
                    np.sign(
                        np.sin(
                            ph
                        )
                    ) *
                    0.6 *
                    np.exp(
                        -times * 12.0
                    )
                )

                outdata[:, 0] = wave

                phase0 = (
                    phase0 +
                    2.0 *
                    np.pi *
                    freq *
                    (
                        n / sr
                    )
                ) % (2.0 * np.pi)

                t0 += n / sr

            try:
                stream = sd.OutputStream(
                    samplerate=self.sr,
                    channels=1,
                    dtype="float32",
                    blocksize=512,
                    callback=callback
                )
                stream.start()
            except Exception:
                return

            self._preview_stream = stream

            while not self._preview_stop_event.is_set():
                if self.playing:
                    break

                if self.output_device != "internal":
                    break

                if not stream.active:
                    break

                time.sleep(0.05)

            try:
                stream.close()
            except Exception:
                pass

            self._preview_stream = None

    def _preview_worker_midi(self):
        current_pitch = None
        current_trigger = -1
        sounding_pitch = None
        current_since = 0.0

        while not self._preview_stop_event.is_set():
            device = self._midi_out

            if device is None:
                return

            if self.playing:
                time.sleep(0.01)
                continue

            pitch = (
                self._preview_target["pitch"]
            )
            trigger = (
                self._preview_target.get("trigger", 0)
            )

            if (
                pitch != current_pitch or
                trigger != current_trigger
            ):
                if sounding_pitch is not None:
                    try:
                        device.note_off(sounding_pitch)
                    except Exception:
                        pass

                    sounding_pitch = None

                if pitch is not None:
                    try:
                        device.note_on(pitch, 100)
                    except Exception:
                        pass

                    sounding_pitch = pitch
                    current_since = time.perf_counter()

                current_pitch = pitch
                current_trigger = trigger

            if (
                sounding_pitch is not None and
                time.perf_counter() - current_since > 0.3
            ):
                try:
                    device.note_off(sounding_pitch)
                except Exception:
                    pass

                sounding_pitch = None

            time.sleep(0.01)

    def _silence_preview(self):
        self._preview_target["pitch"] = None

    def pause(self):
        if not self.playing:
            return

        self.update_position()

        self.playing = False
        self.paused = True

        self._silence_midi_out()

        self._silence_preview()

    def stop(self):
        self.playing = False
        self.paused = False

        self._silence_midi_out()

        self._silence_preview()

        self.position = (
            self._start_position
        )
        self.midi_phase.clear()
        self.midi_active.clear()
        self._midi_cache_dirty = True
        self.preview_pitch = None
        self.preview_until = 0.0

    def clear(self):
        self.stop()

        self.y = None
        self.sr = 44100
        self.offset = 0.0
        self._start_position = 0.0
        self.file_path = None

    def seek(self, seconds):
        was_playing = self.playing

        if was_playing:
            self.playing = False
            self._silence_midi_out()

        self.position = max(
            0.0,
            min(
                seconds,
                self.max_position()
            )
        )

        self.preview_pitch = None
        self.preview_until = 0.0
        self._midi_out_preview_pitch = None

        if was_playing:
            self.play()

    def update_position(self):
        if not self.playing:
            return

        elapsed = (
            time.perf_counter() -
            self._started_at
        )

        estimate = (
            self._start_position +
            max(
                0.0,
                elapsed - self._latency
            )
        )

        rendered = (
            self._start_position +
            self._render_progress
        )

        self.position = min(
            self.max_position(),
            min(
                estimate,
                rendered
            )
        )

        if self.position >= self.max_position():
            self.playing = False