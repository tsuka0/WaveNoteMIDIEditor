import math
import threading

import librosa
import numpy as np
from scipy.signal import resample_poly
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from PySide6.QtWidgets import QWidget
from .i18n import tr

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
        tempo_estimator = None
        try:
            from madmom.features.tempo import (
                CombFilterTempoHistogramProcessor,
                TempoEstimationProcessor,
            )

            tempo_histogram = CombFilterTempoHistogramProcessor(
                min_bpm=30.0,
                max_bpm=300.0,
                fps=_MADMOM_FPS,
            )
            tempo_estimator = TempoEstimationProcessor(
                method=None,
                min_bpm=30.0,
                max_bpm=300.0,
                fps=_MADMOM_FPS,
                histogram_processor=tempo_histogram,
                interpolate=True,
            )
        except Exception:
            # DBN beat intervals remain a madmom-only tempo source if this
            # optional histogram processor cannot be initialized.
            pass
        _madmom_processors = (
            RNNBeatProcessor(),
            tempo_estimator,
            DBNBeatTrackingProcessor(
                min_bpm=30.0,
                max_bpm=300.0,
                fps=_MADMOM_FPS,
            ),
        )
    return _madmom_processors


# テンポ候補グラフのBPM範囲と刻み。
# 「解析結果が違ったときの次の候補」を探しやすいように、
# 一般的な楽曲テンポをほぼ覆う範囲を0.5 BPM刻みで評価する。
_CANDIDATE_MIN_BPM = 30.0
_CANDIDATE_MAX_BPM = 300.0
_CANDIDATE_STEP = 0.5
_GRID_PHASES = 16


class TempoAnalysis:
    """テンポ解析の結果と、グラフ表示用の付随データ。

    env は拍活性度(0..1)、beat_times はmadmomが検出した拍位置(秒)、
    candidates は (BPM, スコア, 最適位相) の配列。
    """

    def __init__(
        self,
        bpm,
        beat_origin,
        beat_times=None,
        env=None,
        env_fps=_MADMOM_FPS,
        candidates=None,
        source="madmom"
    ):
        self.bpm = float(bpm)
        self.beat_origin = float(beat_origin)
        self.beat_times = np.asarray(
            [] if beat_times is None else beat_times,
            dtype=np.float64
        )
        self.env = np.asarray(
            [] if env is None else env,
            dtype=np.float32
        )
        self.env_fps = float(env_fps) if env_fps else 0.0
        if candidates is None:
            candidates = np.zeros((0, 3), dtype=np.float64)
        self.candidates = np.asarray(candidates, dtype=np.float64)
        self.source = source

    def __iter__(self):
        yield self.bpm
        yield self.beat_origin

    @property
    def duration(self):
        if self.env.size == 0 or self.env_fps <= 0:
            return 0.0
        return float(self.env.size) / self.env_fps


class TempoCandidateGraph(QWidget):
    """Interactive BPM versus rhythmic grid-alignment score plot."""
    def __init__(self, analysis, on_select=None, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.on_select = on_select
        self._hover_row = None
        self.setMinimumHeight(165)
        self.setMouseTracking(True)

    def _rect(self):
        return self.rect().adjusted(42, 10, -12, -27)

    def _row_at(self, pos):
        rows = self.analysis.candidates
        r = self._rect()
        if len(rows) == 0 or not r.contains(pos):
            return None
        bpm = rows[0, 0] + (pos.x() - r.left()) / max(1, r.width()) * (rows[-1, 0] - rows[0, 0])
        return rows[int(np.argmin(np.abs(rows[:, 0] - bpm)))]

    def mouseMoveEvent(self, event):
        self._hover_row = self._row_at(event.position().toPoint())
        row = self._hover_row
        self.setToolTip("" if row is None else f"{row[0]:.1f} BPM | score {row[1] * 100:.1f}%\nClick to use this tempo")
        self.update()

    def leaveEvent(self, event):
        self._hover_row = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        row = self._row_at(event.position().toPoint())
        if row is not None and self.on_select:
            self.on_select(float(row[0]))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#20242b"))
        r, rows = self._rect(), self.analysis.candidates
        p.setPen(QPen(QColor("#59616d"), 1))
        p.drawRect(r)
        if len(rows) == 0:
            p.setPen(QColor("#d0d5dd"))
            p.drawText(r, Qt.AlignCenter, tr("候補データがありません", "No candidate data available"))
            return
        for level in (0.0, 0.5, 1.0):
            y = r.bottom() - level * r.height()
            p.setPen(QPen(QColor("#39414c"), 1))
            p.drawLine(int(r.left()), int(y), int(r.right()), int(y))
            p.setPen(QColor("#b8c0cc"))
            p.drawText(3, int(y + 4), f"{level:.1f}")
        lo, hi = float(rows[0, 0]), float(rows[-1, 0])
        path = QPainterPath()
        for i, row in enumerate(rows):
            x = r.left() + (row[0] - lo) / max(1e-9, hi - lo) * r.width()
            y = r.bottom() - np.clip(row[1], 0, 1) * r.height()
            (path.moveTo if i == 0 else path.lineTo)(x, y)
        p.setPen(QPen(QColor("#54c8a8"), 2))
        p.drawPath(path)
        peak = rows[int(np.argmax(rows[:, 1]))]
        peak_x = r.left() + (peak[0] - lo) / max(1e-9, hi - lo) * r.width()
        peak_y = r.bottom() - np.clip(peak[1], 0, 1) * r.height()
        p.setPen(QPen(QColor("#f7f0a8"), 2))
        p.setBrush(QColor("#f7f0a8"))
        p.drawEllipse(int(peak_x) - 4, int(peak_y) - 4, 8, 8)
        p.setPen(QColor("#f7f0a8"))
        label_x = min(max(int(peak_x) + 7, int(r.left())), int(r.right()) - 105)
        label_y = max(int(r.top()) + 15, int(peak_y) - 7)
        p.drawText(label_x, label_y, f"Score peak {peak[0]:.1f} BPM")
        x = r.left() + (self.analysis.bpm - lo) / max(1e-9, hi - lo) * r.width()
        p.setPen(QPen(QColor("#ffbd59"), 2, Qt.DashLine))
        p.drawLine(int(x), int(r.top()), int(x), int(r.bottom()))
        p.setPen(QColor("#d0d5dd"))
        p.drawText(int(r.left()), self.height() - 6, f"{lo:g} BPM")
        p.drawText(int(r.right() - 50), self.height() - 6, f"{hi:g} BPM")
        if self._hover_row is not None:
            row = self._hover_row
            p.setPen(QPen(QColor("#f4f6f8"), 1))
            p.setBrush(QColor(30, 34, 41, 235))
            box = p.boundingRect(0, 0, 160, 28, Qt.AlignCenter, f"{row[0]:.1f} BPM | {row[1] * 100:.1f}%")
            box.moveTo(r.left() + 8, r.top() + 8)
            p.drawRoundedRect(box, 4, 4)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, Qt.AlignCenter, f"{row[0]:.1f} BPM | {row[1] * 100:.1f}%")

def _grid_alignment(env, fps, bpm):
    """固定グリッド上と拍裏の活性度差からBPMの当てはまりを測る。

    戻り値は (スコア, 最適位相[秒])。位相を複数試し、
    拍上と拍裏の分離が最大になる位置を採用する。
    """
    if env is None or len(env) < 8 or fps <= 0 or bpm <= 0:
        return -1.0, 0.0

    hop_s = 1.0 / float(fps)
    period = 60.0 / float(bpm)
    half = max(1, int(round(period / 2.0 / hop_s)))
    n = len(env)
    t_end = (n - 1) * hop_s
    best = -1.0
    best_phase = 0.0

    for phi in np.linspace(0.0, period, _GRID_PHASES, endpoint=False):
        grid = phi + np.arange(0.0, t_end - phi + period, period)
        if grid.size == 0:
            continue

        on_idx = np.clip(
            np.round(grid / hop_s).astype(int), 0, n - 1
        )
        off_idx = np.clip(on_idx + half, 0, n - 1)
        on = float(env[on_idx].mean())
        off = float(env[off_idx].mean())
        ratio = (on - off) / (on + off + 1e-9)
        if ratio > best:
            best = ratio
            best_phase = float(phi)

    return best, best_phase


def _compute_tempo_candidates(env, fps):
    """BPM範囲全体でグリッド整合スコアを計算する。"""
    if env is None or len(env) < 8 or fps <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Candidate scoring samples only beat-grid neighborhoods, so reducing the
    # envelope to 25 Hz keeps long-track analysis responsive without changing
    # the BPM resolution.
    if fps > 25:
        stride = max(1, int(round(fps / 25.0)))
        env = np.asarray(env)[::stride]
        fps = float(fps) / stride

    bpms = np.arange(
        _CANDIDATE_MIN_BPM,
        _CANDIDATE_MAX_BPM + 1e-9,
        _CANDIDATE_STEP
    )
    rows = np.empty((len(bpms), 3), dtype=np.float64)
    for i, candidate in enumerate(bpms):
        score, phase = _grid_alignment(env, fps, float(candidate))
        rows[i] = (float(candidate), score, phase)

    max_score = float(np.nanmax(rows[:, 1])) if len(rows) else 0.0
    if max_score > 1e-9:
        rows[:, 1] = np.clip(rows[:, 1] / max_score, 0.0, 1.0)
    else:
        rows[:, 1] = 0.0

    return rows


def _normalize_envelope(env):
    env = np.asarray(env, dtype=np.float64)
    peak = float(env.max()) if env.size else 0.0
    if peak <= 1e-9:
        return np.zeros_like(env, dtype=np.float32)
    return (env / peak).astype(np.float32)


class SpectrumData:
    def __init__(self):
        self.data = None
        self.times = None
        self.sr = None
        self.hop_length = 512
        self.last_tempo_analysis_error = None

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
        self.last_tempo_analysis_error = None
        try:
            result = self._analyze_tempo_madmom(y, sr)
        except Exception as exc:
            self.last_tempo_analysis_error = f"{type(exc).__name__}: {exc}"
            return None
        if result is None and not self.last_tempo_analysis_error:
            self.last_tempo_analysis_error = (
                "madmom did not produce a usable tempo and beat track."
            )
        return result

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
            rnn, tempo_estimator, dbn = _get_madmom_processors()
            with _madmom_lock:
                env = np.asarray(rnn(y), dtype=np.float64)
                try:
                    if tempo_estimator is None:
                        raise RuntimeError("madmom tempo histogram unavailable")
                    tempo_estimates = np.asarray(tempo_estimator(env), dtype=np.float64)
                except Exception:
                    tempo_estimates = np.zeros((0, 2), dtype=np.float64)
                beat_times = np.asarray(dbn(env), dtype=np.float64)
        except Exception as exc:
            self.last_tempo_analysis_error = f"{type(exc).__name__}: {exc}"
            return None

        if env.size < 8:
            return None

        env_max = float(env.max())
        if env_max <= 1e-9:
            return None

        beat_times = beat_times[np.isfinite(beat_times)]
        # Prefer madmom's RNN activation + comb-filter tempo estimate. If the
        # histogram has no usable peak, use the DBN beat intervals, still from
        # madmom, so the analysis and graph remain available.
        tempo_estimates = np.asarray(tempo_estimates, dtype=np.float64)
        if tempo_estimates.ndim == 2 and tempo_estimates.shape[1] >= 2:
            valid = np.isfinite(tempo_estimates[:, 0]) & np.isfinite(tempo_estimates[:, 1])
            tempo_estimates = tempo_estimates[valid]
        else:
            tempo_estimates = np.zeros((0, 2), dtype=np.float64)

        if len(tempo_estimates):
            bpm = float(tempo_estimates[0, 0])
        else:
            intervals = np.diff(beat_times)
            intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
            if intervals.size == 0:
                return None
            bpm = 60.0 / float(np.median(intervals))
        if not np.isfinite(bpm) or not (30.0 <= bpm <= 300.0):
            return None
        # Use conventional half-up rounding for the applied tempo, while the
        # optional candidate graph can still show its fractional BPM values.
        bpm = float(math.floor(bpm + 0.5))

        hop_s = 1.0 / float(_MADMOM_FPS)
        times = np.arange(env.size, dtype=np.float64) * hop_s
        strengths = env / env_max

        # 固定グリッド(位相スイープ付き)での拍上/拍裏の分離度。
        # 整数テンポの精密化に使う。numpyのみで高速。
        # Candidate scoring is retained only for the optional dialog graph.
        normalized = _normalize_envelope(strengths)
        candidates = _compute_tempo_candidates(normalized, _MADMOM_FPS)

        # DBN beat positions choose the absolute cycle for the comb-estimated BPM.
        grid_seconds_per_beat = 60.0 / bpm
        beat_origin = float(beat_times[0]) if beat_times.size else 0.0
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

        return TempoAnalysis(
            bpm=bpm,
            beat_origin=beat_origin,
            beat_times=beat_times,
            env=normalized,
            env_fps=_MADMOM_FPS,
            candidates=candidates,
            source="madmom"
        )
