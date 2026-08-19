import threading
import time
import bisect
import numpy as np
import sounddevice as sd
import librosa
from . import midiout

class AudioData:
    def __init__(self):
        self.y = None
        self.sr = 44100
        self.position = 0.0
        self.playing = False
        
        self._a4_freq = 440.0

        self._started_at = 0.0
        self._start_position = 0.0
        self._thread = None
        self._play_gen = 0
        self._latency = 0.0
        self._latency = 0.0

        self.midi = None
        self.midi_phase = {}
        self.midi_active = set()

        self._r_notes = []
        
        self._r_starts = []
        
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
        
        self._preview_stop_event = threading.Event()

        self._render_progress = 0.0

        self._tl_version = -1
        self._tl_dur_audio = None
        self._tl_dur_offset = None
        self._tl_duration = 1.0

        self.file_path = None
        self._midi_cache_dirty = False
        self._midi_cache_version = 0

    @property
    def a4_freq(self):
        return self._a4_freq

    @a4_freq.setter
    def a4_freq(self, value):
        self._a4_freq = value
        self._send_tuning_to_midi_out()

    def set_midi(self, midi):
        self.midi = midi
        self._midi_cache_dirty = True

    def invalidate_midi_cache(self):
        self._midi_cache_dirty = True

    def set_output_device(self, device):
        with self._midi_lock:
            if self.output_device == device:
                return
                
            self.output_device = device
            self._close_midi_out()

            if device == "internal":
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
            self._send_tuning_to_midi_out()
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

        old_thread = getattr(self, "_thread", None)

        self.playing = True
        
        self.auto_stopped = False

        self._silence_preview()

        self._start_position = self.position
        self._started_at = time.perf_counter()
        self._latency = 0.0
        self._render_progress = 0.0

        self._play_gen += 1
        gen = self._play_gen

        self._thread = threading.Thread(
            target=self._play_worker,
            args=(gen, old_thread),
            daemon=True
        )

        self._thread.start()

    def _play_worker(self, gen, old_thread=None):
        if old_thread is not None and old_thread.is_alive() and old_thread != threading.current_thread():
            old_thread.join(timeout=0.5)

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

        short_notes = []
        long_notes = []
        max_short_duration = 0.0
        
        for n in note_list:
            if n.duration > 8.0:
                long_notes.append(n)
            else:
                short_notes.append(n)
                if n.duration > max_short_duration:
                    max_short_duration = n.duration

        short_notes.sort(key=lambda n: n.start)

        self._r_notes = short_notes
        self._r_starts = [n.start for n in short_notes]
        self._r_max_dur = max_short_duration
        self._r_long_notes = long_notes

        self._midi_cache_version += 1
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
        local_version = self._midi_cache_version
        
        active_notes = {}
        next_note_idx = 0
        
        def reset_active_notes(time_pos):
            nonlocal next_note_idx
            active_notes.clear()
            self.midi_phase.clear()
            
            i0 = bisect.bisect_left(self._r_starts, time_pos - self._r_max_dur - 0.001)
            next_note_idx = bisect.bisect_right(self._r_starts, time_pos)
            
            for i in range(i0, next_note_idx):
                note = self._r_notes[i]
                if note.start + note.duration > time_pos:
                    active_notes[id(note)] = note
                    
            for note in self._r_long_notes:
                if note.start <= time_pos and note.start + note.duration > time_pos:
                    active_notes[id(note)] = note
        
        reset_active_notes(start_position)

        try:
            stream.start()

            while (
                self.playing and
                sample_position < total_samples
            ):
                if getattr(self, "_seek_request", None) is not None:
                    new_pos = self._seek_request
                    self._seek_request = None
                    start_position = new_pos
                    sample_position = 0
                    self._start_position = new_pos
                    self._started_at = time.perf_counter()
                    self._render_progress = 0.0
                    self._silence_midi_out()
                    reset_active_notes(new_pos)
                    continue

                if self._midi_cache_dirty or local_version != self._midi_cache_version:
                    if self._midi_cache_dirty:
                        self._refresh_midi_cache()
                    local_version = self._midi_cache_version

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

                end_time = current_time + block_size / sample_rate
                
                while next_note_idx < len(self._r_starts) and self._r_starts[next_note_idx] < end_time:
                    note = self._r_notes[next_note_idx]
                    if note.start + note.duration > current_time:
                        active_notes[id(note)] = note
                    next_note_idx += 1
                
                to_remove = []
                for note_id, note in active_notes.items():
                    if note.start + note.duration <= current_time:
                        to_remove.append(note_id)
                        
                for nid in to_remove:
                    del active_notes[nid]

                midi_block = self._render_midi(
                    current_time,
                    block_size,
                    sample_rate,
                    active_notes
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
                    self.auto_stopped = True

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
                max(note_start, start_position) - start_position,
                "on",
                note.pitch,
                note.velocity,
                channel
            ))

            events.append((
                max(0.0, max(note_end, start_position) - start_position - 0.001),
                "off",
                note.pitch,
                0,
                channel
            ))

        if self.midi is not None:
            if self.midi.filter_track is not None:
                index = self.midi.filter_track
                tracks = [self.midi.tracks[index]] if 0 <= index < len(self.midi.tracks) else []
            else:
                tracks = self.midi.tracks

            pedals = []
            for track in tracks:
                for pedal in track.pedals:
                    pedals.append((pedal, track.channel))

            pedals.sort(key=lambda e: e[0].time)
            ch_state = {}
            for pedal, ch in pedals:
                if pedal.time <= start_position:
                    ch_state[ch] = pedal.down
                else:
                    break

            for ch, st in ch_state.items():
                if st:
                    events.append((
                        0.0,
                        "pedal",
                        True,
                        0,
                        ch
                    ))

            for pedal, ch in pedals:
                if pedal.time <= start_position + 1e-6:
                    continue
                if pedal.time >= total_time:
                    continue

                adv = 0.001 if not pedal.down else 0.0
                events.append((
                    max(0.0, pedal.time - start_position - adv),
                    "pedal",
                    pedal.down,
                    0,
                    ch
                ))

        def _sort_key(ev):
            t = round(ev[0], 5)
            kind = ev[1]
            if kind == "off":
                priority = 0
            elif kind == "on":
                priority = 1
            elif kind == "pedal" and not ev[2]:
                priority = 2
            elif kind == "pedal" and ev[2]:
                priority = 3
            else:
                priority = 4
            return (t, priority)

        events.sort(key=_sort_key)

        return events

    def _send_tuning_to_midi_out(self):
        device = self._midi_out
        if device is None:
            return

        cents = 1200.0 * np.log2(self.a4_freq / 440.0)
        coarse = int(round(cents / 100.0))
        fine_cents = cents - coarse * 100.0
        
        fine_val = int(round(8192 + fine_cents * 81.92))
        fine_val = max(0, min(16383, fine_val))
        
        fine_msb = fine_val >> 7
        fine_lsb = fine_val & 127
        
        coarse_val = max(0, min(127, 64 + coarse))

        for ch in range(16):
            try:
                device.control_change(101, 0, ch)
                device.control_change(100, 2, ch)  # RPN 00 02 is Channel Coarse Tuning
                device.control_change(6, coarse_val, ch)
                
                device.control_change(101, 0, ch)
                device.control_change(100, 1, ch)  # RPN 00 01 is Channel Fine Tuning
                device.control_change(6, fine_msb, ch)
                device.control_change(38, fine_lsb, ch)
                
                device.control_change(101, 127, ch)
                device.control_change(100, 127, ch)
            except Exception:
                pass

    def _midi_out_worker(
        self,
        start_position,
        total_time,
        gen,
        latency
    ):
        try:
            self._send_tuning_to_midi_out()
            sounding = set()

            song_base = start_position

            wall_base = time.perf_counter() + latency

            events = self._build_midi_events(
                self._r_notes,
                song_base,
                total_time
            )

            index = 0
            local_version = self._midi_cache_version

            while True:
                if (
                    not self.playing or
                    gen != self._play_gen
                ):
                    return

                if getattr(self, "_midi_out_seek_request", None) is not None:
                    new_pos = self._midi_out_seek_request
                    self._midi_out_seek_request = None
                    start_position = new_pos
                    song_base = new_pos
                    wall_base = time.perf_counter() + latency
                    if self._midi_out is not None and sounding:
                        try:
                            self._midi_out.all_notes_off()
                        except Exception:
                            pass
                        sounding.clear()
                    all_notes = self._r_notes + self._r_long_notes
                    events = self._build_midi_events(all_notes, song_base, total_time)
                    index = 0
                    continue

                if self._midi_cache_dirty or local_version != self._midi_cache_version:
                    now = time.perf_counter()

                    elapsed = now - wall_base

                    if elapsed > 0.0:
                        new_song = song_base + elapsed

                        if new_song < start_position:
                            new_song = start_position
                        else:
                            wall_base = now

                        song_base = new_song

                    if self._midi_cache_dirty:
                        self._refresh_midi_cache()
                    
                    local_version = self._midi_cache_version

                    device = self._midi_out

                    if device is not None and sounding:
                        try:
                            device.all_notes_off()
                        except Exception:
                            pass

                        sounding.clear()

                    events = self._build_midi_events(
                        self._r_notes + self._r_long_notes,
                        song_base,
                        total_time
                    )

                    index = 0

                    continue

                if index >= len(events):
                    time.sleep(0.005)
                    continue

                now = time.perf_counter()
                
                # 蜷後§繧ｿ繧､繝溘Φ繧ｰ・医∪縺溘・驕主悉・峨・繧､繝吶Φ繝医ｒ荳豌励↓騾∽ｿ｡縺吶ｋ
                while index < len(events):
                    relative, kind, pitch, velocity, channel = events[index]
                    target = relative - (now - wall_base)
                    
                    if target > 0.0:
                        break

                    device = self._midi_out
                    if device is None:
                        return

                    try:
                        if kind == "on":
                            device.note_on(pitch, velocity, channel)
                            sounding.add((pitch, channel))
                        elif kind == "pedal":
                            device.control_change(64, 127 if pitch else 0, channel)
                        else:
                            device.note_off(pitch, channel)
                            sounding.discard((pitch, channel))
                    except Exception:
                        pass

                    index += 1

                if index < len(events):
                    relative = events[index][0]
                    target = relative - (time.perf_counter() - wall_base)
                    if target > 0.0:
                        time.sleep(min(0.002, target))
        finally:
            if self._midi_out is not None:
                try:
                    self._midi_out.all_notes_off()
                except Exception:
                    pass

    def _render_midi(self, start_time, length, sample_rate, active_notes=None):
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

        if active_notes is not None:
            iterable = active_notes.items()
        else:
            # fallback for external calls if any
            iterable = []

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

            frequency = self.a4_freq * (
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
            frequency = self.a4_freq * (
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

                freq = self.a4_freq * (
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
        

        self._silence_midi_out()
        self._silence_preview()

    def stop(self):
        self.playing = False
        
        self._silence_midi_out()
        self._silence_preview()
        self.position = self._start_position
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
        new_pos = max(0.0, min(seconds, self.max_position()))

        self.preview_pitch = None
        self.preview_until = 0.0
        self._midi_out_preview_pitch = None

        if self.playing:
            self._seek_request = new_pos
            self._midi_out_seek_request = new_pos
        else:
            self.position = new_pos

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

    def export_wav(self, path):
        import scipy.io.wavfile as wavfile
        import numpy as np

        total_time = self.midi.max_extended_end()

        sample_rate = self.sr
        total_samples = int(total_time * sample_rate)

        out_wave = np.zeros(total_samples, dtype=np.float32)

        # 險ｭ螳壹・騾驕ｿ・亥・驛ｨ髻ｳ貅舌ｒ蠑ｷ蛻ｶ・・・繝医Λ繝・け蜃ｺ蜉幢ｼ・        old_midi_out = self._midi_out
        old_filter = self.midi.filter_track
        
        try:
            self._midi_out = None
            self.midi.filter_track = None
            self._refresh_midi_cache()
            
            block_size = 1024
            active_notes = {}
            next_note_idx = 0
            
            for i in range(0, total_samples, block_size):
                current_time = i / sample_rate
                n = min(block_size, total_samples - i)
                end_time = current_time + n / sample_rate
                
                while next_note_idx < len(self._r_starts) and self._r_starts[next_note_idx] < end_time:
                    note = self._r_notes[next_note_idx]
                    if note.start + note.duration > current_time:
                        active_notes[id(note)] = note
                    next_note_idx += 1
                
                to_remove = []
                for note_id, note in active_notes.items():
                    if note.start + note.duration <= current_time:
                        to_remove.append(note_id)
                for nid in to_remove:
                    del active_notes[nid]

                midi_block = self._render_midi(current_time, n, sample_rate, active_notes)

                out_wave[i:i+n] = midi_block * 0.35
        finally:
            self._midi_out = old_midi_out
            self.midi.filter_track = old_filter
            self._refresh_midi_cache()

        peak = np.max(np.abs(out_wave))
        if peak > 0.0:
            if peak > 0.98:
                out_wave /= (peak / 0.98)
        
        out_wave_16 = np.int16(out_wave * 32767)
        wavfile.write(path, sample_rate, out_wave_16)
