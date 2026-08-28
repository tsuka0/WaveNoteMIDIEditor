import math

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


        # Shift fmin down by 1 sub-bin (1/36 octave) so that the middle of the 3 sub-bins is exactly the note frequency
        fmin = a4_freq * (2.0 ** ((min_note - 69) / 12.0 - 1.0 / 36.0))

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
        """Return the estimated BPM and the time of the first detected beat.

        手順:
        1. オンセット強度包絡を計算(タップ計測と同じ22.05kHz/256hopに統一)
        2. 自己相関(co-comb)で大まかなBPMを推定
        3. その±8%をDFTグリッド+三分探索で精密に固定(タップ解析と同じ方式)
        4. 精密BPMを事前情報としてビートトラッカーを走らせ、
           検出した全ビートから回帰で最終BPMとグリッド原点を決める
        """
        if y is None or sr is None or len(y) == 0:
            return None

        y = np.asarray(y, dtype=np.float32)

        # --- オンセット包絡(タップ計測 compute_onset_envelope と同条件) ---
        onset_sr = sr
        onset_hop = 256
        if onset_sr > 22050:
            y_onset = librosa.resample(
                y,
                orig_sr=sr,
                target_sr=22050
            )
            onset_sr = 22050
        else:
            y_onset = y

        env = librosa.onset.onset_strength(
            y=y_onset,
            sr=onset_sr,
            hop_length=onset_hop
        )

        env = np.asarray(env, dtype=np.float64)

        if len(env) < 8:
            return None

        # --- BPM推定(自己相関 + DFT精密固定) ---
        try:
            import numpy as _np
            from numpy import median as _median

            raw_tempo_arr = librosa.feature.tempo(
                onset_envelope=env,
                sr=onset_sr,
                hop_length=onset_hop,
                start_bpm=120.0,
                std_bpm=24.0,
                aggregate=_median
            )
            raw_tempo = float(
                _np.asarray(raw_tempo_arr).reshape(-1)[0]
            )
        except Exception:
            raw_tempo = 120.0

        if not np.isfinite(raw_tempo) or not (30.0 <= raw_tempo <= 300.0):
            raw_tempo = 120.0

        times = librosa.frames_to_time(
            np.arange(len(env)),
            sr=onset_sr,
            hop_length=onset_hop
        )
        strengths = env / (float(env.max()) + 1e-12)

        # 解析点数を抑えるためにオンセット列を上限まで間引く
        max_points = 8000
        stride = max(1, len(times) // max_points)
        if stride > 1:
            times = times[::stride]
            strengths = strengths[::stride]

        def dft_score(periods):
            omega = 2.0 * np.pi / periods
            ang = omega[:, None] * times[None, :]
            re = np.sum(strengths[None, :] * np.cos(ang), axis=1)
            im = np.sum(strengths[None, :] * np.sin(ang), axis=1)
            return np.hypot(re, im)

        if raw_tempo >= 30.0:
            t_lo = raw_tempo * 0.92
            t_hi = raw_tempo * 1.08
            n_steps = 320
            period_grid = 60.0 / np.linspace(t_lo, t_hi, n_steps)

            scores = dft_score(period_grid)
            idx = int(np.argmax(scores))

            if 0 < idx < n_steps - 1:
                p_lo = period_grid[idx - 1]
                p_hi = period_grid[idx + 1]
            else:
                p_lo = period_grid[0]
                p_hi = period_grid[-1]

            for _ in range(50):
                m1 = (2.0 * p_lo + p_hi) / 3.0
                m2 = (p_lo + 2.0 * p_hi) / 3.0
                if dft_score(np.array([m1]))[0] < dft_score(np.array([m2]))[0]:
                    p_lo = m1
                else:
                    p_hi = m2

            best_period = (p_lo + p_hi) / 2.0

            if 0.1 <= best_period <= 2.3:
                refined_bpm = 60.0 / best_period
            else:
                refined_bpm = raw_tempo
        else:
            refined_bpm = raw_tempo

        # --- ビートトラッキング(推定BPMを事前情報に使用) ---
        _tracked_tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=env,
            sr=onset_sr,
            hop_length=onset_hop,
            start_bpm=refined_bpm,
            trim=False
        )

        if len(beat_frames) == 0:
            return None

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=onset_sr,
            hop_length=onset_hop
        )

        bpm = refined_bpm

        # The beat tracker reports tempo at frame resolution.  Re-estimating
        # from all detected beat positions avoids a one-BPM quantisation error.
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

        # オンセット列のDFT位相でグリッドを最も強い打点時刻に引き寄せる。
        # 位相は周期不明(同一時刻 mod 1拍)なので、最小二乗原点に最も
        # 近い整数拍のコピーを選ぶ(tap_engine と同じ位相スナップ方式)。
        omega = 2.0 * math.pi * bpm / 60.0
        re = float(
            np.sum(
                strengths * np.cos(omega * times)
            )
        )
        im = float(
            np.sum(
                strengths * np.sin(omega * times)
            )
        )

        if re != 0.0 or im != 0.0:
            phi = math.atan2(im, re) / omega
            beat_origin = (
                phi +
                grid_seconds_per_beat *
                round(
                    (beat_origin - phi) /
                    grid_seconds_per_beat
                )
            )

        return bpm, beat_origin
