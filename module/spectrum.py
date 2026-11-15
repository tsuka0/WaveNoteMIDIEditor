import math
import threading

import librosa
import numpy as np
from scipy.signal import resample_poly

# madmomの拍解析モデルは44.1kHzモノラルを前提としている。
_MADMOM_SAMPLE_RATE = 44100
_MADMOM_FPS = 100
_madmom_processors = None
_madmom_lock = threading.Lock()


def _get_madmom_processors():
    """madmomの拍解析プロセッサを遅延生成して再利用する。"""
    global _madmom_processors
    if _madmom_processors is None:
        from madmom.features.beats import (
            DBNBeatTrackingProcessor,
            RNNBeatProcessor,
        )

        _madmom_processors = (
            RNNBeatProcessor(),
            DBNBeatTrackingProcessor(fps=_MADMOM_FPS),
        )
    return _madmom_processors


def _analyze_tempo_librosa(y, sr):
    """madmomが利用できない場合のフォールバック。

    madmomを優先するが、実行環境にmadmomが無い場合でもテンポが
    反映されるようにlibrosaのビートトラッカーで最小限の解析を行う。
    """
    try:
        if sr > 22050:
            y = librosa.resample(
                y,
                orig_sr=sr,
                target_sr=22050
            )
            sr = 22050

        hop = 256
        env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop
        )
        if len(env) < 8:
            return None

        _tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=env,
            sr=sr,
            hop_length=hop,
            trim=False
        )

        bpm = float(np.asarray(_tempo).reshape(-1)[0])
        if not np.isfinite(bpm) or not (30.0 <= bpm <= 300.0):
            return None

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=hop
        )
        if len(beat_times) == 0:
            return None

        return float(round(bpm)), float(beat_times[0])
    except Exception:
        return None


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
        """Return the estimated BPM and the time of the first detected beat."""
        result = self._analyze_tempo_madmom(y, sr)
        if result is not None:
            return result
        return _analyze_tempo_librosa(y, sr)

    def _analyze_tempo_madmom(self, y, sr):
        """madmomでBPMと拍位置を解析する。

        手順:
        1. madmom用に44.1kHzモノラルへ変換する。
        2. RNNBeatProcessorで拍活性度(beat activation)を計算する。
        3. DBNBeatTrackingProcessorで拍位置(秒)を検出する。
        4. 検出した拍間隔の中央値からBPMを求める。
        5. 拍活性度の固定グリッド評価で整数BPMへ精密化する。
        6. 拍活性度のDFT位相からグリッド原点を決める。
        """
        if y is None or sr is None or len(y) == 0:
            return None

        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)

        if sr <= 0:
            return None

        # madmomのRNNモデルは44.1kHzモノラルを前提としている。
        if sr != _MADMOM_SAMPLE_RATE:
            gcd = math.gcd(int(sr), _MADMOM_SAMPLE_RATE)
            y = resample_poly(
                y,
                _MADMOM_SAMPLE_RATE // gcd,
                int(sr) // gcd
            ).astype(np.float32)

        try:
            rnn, dbn = _get_madmom_processors()
            with _madmom_lock:
                env = np.asarray(rnn(y), dtype=np.float64)
                beat_times = np.asarray(dbn(env), dtype=np.float64)
        except Exception:
            return None

        if env.size < 8:
            return None

        env_max = float(env.max())
        if env_max <= 1e-9:
            return None

        beat_times = beat_times[np.isfinite(beat_times)]
        if beat_times.size < 2:
            return None

        intervals = np.diff(beat_times)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return None

        # DBNが検出した拍間隔の中央値からテンポを求める。
        bpm = 60.0 / float(np.median(intervals))
        if not np.isfinite(bpm) or not (30.0 <= bpm <= 300.0):
            return None

        hop_s = 1.0 / float(_MADMOM_FPS)
        times = np.arange(env.size, dtype=np.float64) * hop_s
        strengths = env / env_max

        # 固定グリッド(位相スイープ付き)での拍上/拍裏の分離度。
        # 整数テンポの精密化に使う。numpyのみで高速。
        def _fixed_grid_ratio(candidate_bpm):
            _per = 60.0 / candidate_bpm
            _half = max(1, int(round(_per / 2.0 / hop_s)))
            _n = len(env)
            _t_end = (_n - 1) * hop_s
            _best = -1.0
            _phases = np.linspace(0.0, _per, 16, endpoint=False)
            for _phi in _phases:
                _grid = _phi + np.arange(0.0, _t_end - _phi + _per, _per)
                if _grid.size == 0:
                    continue
                _bk = np.clip(
                    np.round(_grid / hop_s).astype(int), 0, _n - 1
                )
                _on = float(env[_bk].mean())
                _ix = np.clip(_bk + _half, 0, _n - 1)
                _off = float(env[_ix].mean())
                _r = (_on - _off) / (_on + _off + 1e-9)
                if _r > _best:
                    _best = _r
            return _best

        # 固定グリッド整合で整数BPMを精密化する。
        # madmomの拍間隔中央値はドリフトで小数になることがあるため、
        # 近傍±5 BPMの各整数で分離度を測り、最大を採用する。
        _lo = int(max(30.0, bpm - 5.0))
        _hi = int(min(300.0, bpm + 5.0))
        _best_bpm = _lo
        _best_score = -1.0
        for _cand in range(_lo, _hi + 1):
            _sc = _fixed_grid_ratio(float(_cand))
            if _sc > _best_score:
                _best_score = _sc
                _best_bpm = _cand
        bpm = float(_best_bpm)

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

        # 拍活性度のDFT位相でグリッドを最も強い打点時刻に引き寄せる。
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
