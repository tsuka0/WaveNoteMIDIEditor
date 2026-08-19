import librosa
import numpy as np

class SpectrumData:
    def __init__(self):
        self.data = None
        self.times = None
        self.sr = None
        self.hop_length = 512

    def clear(self):
        self.data = None
        self.times = None
        self.sr = None

    def analyze(self, y, sr, min_note=36, max_note=96, a4_freq=440.0):
        self.sr = sr


        fmin = a4_freq * (2.0 ** ((min_note - 69) / 12.0))

        bins_per_octave = 36
        sub_bins = bins_per_octave // 12

        n_bins = (
            max_note -
            min_note +
            1
        ) * sub_bins

        cqt = librosa.cqt(
            y,
            sr=sr,
            hop_length=self.hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=0.0
        )

        magnitude = np.abs(cqt)

        frames = magnitude.shape[1]

        magnitude = magnitude.reshape(
            n_bins // sub_bins,
            sub_bins,
            frames
        ).max(axis=1)

        db = librosa.amplitude_to_db(
            magnitude,
            ref=np.max
        )

        self.data = db.astype(
            np.float32
        )

        self.times = librosa.frames_to_time(
            np.arange(
                self.data.shape[1]
            ),
            sr=sr,
            hop_length=self.hop_length
        )

        return self.data

    def analyze_tempo(self, y, sr):
        """Return the estimated BPM and the time of the first detected beat."""
        if y is None or sr is None or len(y) == 0:
            return None

        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=self.hop_length
        )

        if len(beat_frames) == 0:
            return None

        bpm = float(np.asarray(tempo).reshape(-1)[0])

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=self.hop_length
        )

        # The beat tracker reports tempo at frame resolution.  Re-estimating
        # from all detected beat positions avoids a one-BPM quantisation error.
        seconds_per_beat = None

        if len(beat_times) >= 2:
            beat_numbers = np.arange(len(beat_times), dtype=float)
            centered_beats = beat_numbers - beat_numbers.mean()
            denominator = float(np.dot(centered_beats, centered_beats))

            if denominator > 0:
                seconds_per_beat = float(
                    np.dot(
                        centered_beats,
                        beat_times - beat_times.mean()
                    ) / denominator
                )

                if seconds_per_beat > 0:
                    bpm = 60.0 / seconds_per_beat

        if not np.isfinite(bpm) or bpm <= 0:
            return None

        bpm = float(
            np.floor(
                max(30.0, min(300.0, bpm)) + 0.5
            )
        )

        # Fit the grid origin after choosing the display BPM.  This uses every
        # detected beat, rather than trusting a possibly weak first onset.
        grid_seconds_per_beat = 60.0 / bpm
        beat_numbers = np.arange(len(beat_times), dtype=float)
        beat_origin = float(
            np.mean(
                beat_times -
                beat_numbers * grid_seconds_per_beat
            )
        )

        return bpm, beat_origin
