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
        2. ビートトラッカーを既定のオンセットベース事前情報で走らせる。
           自己相関(co-comb)を直接`start_bpm`に渡すと半テンポ(オクターブ下)に
           ロックしやすいため、タクタス(正しいオクターブ)はビートトラッカーの
           自動判定に任せる(WaveToneの全体自動解析と同じ考え方)。
        3. 検出した全ビートから回帰で最終BPM(小数保持)とグリッド原点を決める。
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

        hop_s = onset_hop / float(onset_sr)

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

        # --- ビートトラッキング(初期BPMは渡さず、既定の自動テンポ推定を使用) ---
        # `start_bpm`へ自己相関の半テンポ値を渡すとビートトラッカーがそのオクターブに
        # 固定され、たとえば147 BPMが75 BPMと解析される。既定だと正しいタクタスに
        # (147 BPM, 全拍)を見つける。ビートから回帰で精密化するため拍位置が本質。
        _tracked_tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=env,
            sr=onset_sr,
            hop_length=onset_hop,
            trim=False
        )

        if len(beat_frames) == 0:
            return None

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=onset_sr,
            hop_length=onset_hop
        )

        base_bpm = float(
            np.asarray(_tracked_tempo).reshape(-1)[0]
        )

        # --- オクターブ(半テンポ/倍テンポ)補正 ---
        # 拍トラッカーの結果は時々、半テンポ(オクターブ下)や倍テンポにズレる。
        # {base/2, base, base*2} の各オクターブでビートを再検出し、拍上と拍裏の
        # オンセット強度の分離度(beat_ratio)とテンポ事前分布(タクタス域を優先)の
        # 積が最大のオクターブを選ぶ。これでたとえば147 BPMが75 BPMと解析される
        # 問題を防ぐ。事前分布により倍テンポ(非現実的な速さ)も退ける。
        tempo_prior_mu = 127.0
        tempo_prior_sigma = 0.42

        def _log_prior(bpm):
            return math.exp(
                -0.5 * (
                    (math.log(max(1.0, bpm)) - math.log(tempo_prior_mu))
                    / tempo_prior_sigma
                ) ** 2
            )

        def _octave_ratio(start_bpm):
            _t, _bf = librosa.beat.beat_track(
                onset_envelope=env,
                sr=onset_sr,
                hop_length=onset_hop,
                start_bpm=start_bpm,
                trim=False
            )
            if len(_bf) == 0:
                return None
            _bt = librosa.frames_to_time(
                _bf, sr=onset_sr, hop_length=onset_hop
            )
            if len(_bt) < 4:
                return None
            _tb = float(np.asarray(_t).reshape(-1)[0])
            if not (30.0 <= _tb <= 300.0):
                return None
            _per = 60.0 / _tb
            half = max(1, int(round(_per / 2.0 / hop_s)))
            bk = np.clip(np.round(_bt / hop_s).astype(int), 0, len(env) - 1)
            on = float(env[bk].mean())
            ix = np.clip(bk + half, 0, len(env) - 1)
            off = float(env[ix].mean())
            ratio = (on - off) / (on + off + 1e-9)
            return ratio, _tb, _bt

        # 固定グリッド(位相スイープ付き)での拍上/拍裏の分離度。整数テンポの
        # 精密化と付点(1.5倍)拍の判定に使う。numpyのみで高速。
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

        if np.isfinite(base_bpm) and base_bpm > 0:
            # 複合(付点)拍の補正: 主拍の1.5倍(数え拍)が実在し、テンポ事前分布の
            # 妥当範囲にあれば主拍を1.5倍へ引き上げる。例: test.mp3 は110に支配
            # されるが165が数え拍。事前分布で test2.wav の220(=147×1.5)は退ける。
            _dotted = base_bpm * 1.5
            if (
                30.0 <= _dotted <= 300.0
                and _log_prior(_dotted) >= 0.15
                and _fixed_grid_ratio(_dotted) >= 0.10
            ):
                base_bpm = _dotted

            candidates = []
            for mult in (0.5, 1.0, 2.0):
                cb = base_bpm * mult
                if not (30.0 <= cb <= 300.0):
                    continue
                r = _octave_ratio(cb)
                if r is None:
                    continue
                ratio, _tb, _bt = r
                candidates.append(
                    (ratio * _log_prior(_tb), _tb, _bt, ratio)
                )
            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                bpm = candidates[0][1]
                beat_times = candidates[0][2]
            else:
                bpm = base_bpm
        else:
            bpm = base_bpm

        if not np.isfinite(bpm) or bpm <= 0:
            return None

        # 固定グリッド整合で整数BPMを精密化する。
        # ビートトラッカーの小数BPM(例:166.7)はドリフトで整数からズレる一方、
        # 固定グリッド(位相スイープ付き)の拍上/拍裏の分離度は正しい整数テンポで
        # 鋭く尖る(例:test.mp3は165、test2.wavは147で最大)。選んだ拍の近傍
        # ±5 BPMの各整数で分離度を測り、最大を取ることでドリフトに左右されず、
        # 正確な整数テンポに確定する。
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
