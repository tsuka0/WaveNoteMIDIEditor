import math
import time

# タップの重複検出(チャタリング)防止
TAP_DEBOUNCE_S = 0.09

# この間タップが無ければセッションをリセット
TAP_RESET_GAP_S = 2.5

# 再生中にシーク/停止を検出したとみなすズレ量
JUMP_TOLERANCE_S = 0.15

# 直近のタップほど重みを与える減衰係数。
# 0.95は実効的に約20タップ分の記憶を持たせ、均等重みに近い
# 高精度を保ちつつ、テンポドリフトにも追従できる値。
WEIGHT_DECAY = 0.95

# 適用に必要な最小タップ数
MIN_TAPS_FOR_APPLY = 4

# 保持する最大タップ数
MAX_TAPS = 128

# --- WaveTone方式(音量変化による周期の精密固定)の設定 ---
# タップ回帰の結果から探す範囲(±1.2%)。狭い帯域なので
# 誤った山にロックする余地がほとんどない。
ONSET_BAND = 0.03

# 回帰結果との最大乖離(1%)。超える場合は採用しない。
ONSET_MAX_DEV = 0.010

# タップが少ない初期段階での許容幅(2.5%)。回帰がまだ不安定な
# 頃はオンセット解析を優先して素早く正しいテンポへ寄せる。
ONSET_MAX_DEV_EARLY = 0.025

# 厳しいゲート(ONSET_MAX_DEV)へ切り替えるタップ数
ONSET_STRICT_TAPS = 8

# ロック済み目標周期との許容乖離。回帰がノイズでずれたタップでも
# 目標周期と一致する候補なら採用してアンカーを維持する
ONSET_EMA_DEV = 0.010

# ロック直後の移行期(ONSET_MIGRATE_FITS以内)の許容乖離。
# 広めに許して初期の仮ロックがより強い山へ移動できるようにする
ONSET_EMA_DEV_LOOSE = 0.03

# 移行期とみなすフィット数
ONSET_MIGRATE_FITS = 6

# 初回ロックの確定に必要な候補同士の一致幅
# 窓が移動して多少ピークがブレてもロックできるよう1.2%に緩和
ONSET_PENDING_DEV = 0.012

# EMA追従時の探索帯域(広め)。回帰ノイズで真の山を外さないため
ONSET_BAND_TRACK = 0.055

# 初回ロック時の探索帯域。人間のタップ揺れによる回帰の偏り
# が大きくても真のビート山を捉えられるよう15%に拡大
ONSET_BAND_FIRST = 0.15

# 初回ロックの採用に必要な顕著性(帯域内平均スコア比)。
# 真のビート山は平均より際立つため、ノイズの小さな起伏は除外される
ONSET_LOCK_PROMINENCE = 1.4

# 回帰がEMAから連続で離れたとみなすフィット数の上限。
# 短いノイズの連続で正確なロックを捨てないよう大きめの値にし、
# 超えた場合(持続的なテンポ変化)はEMAを捨てて張り直す
ONSET_FAR_LIMIT = 8

# EMAを捨てた後の再ロック猶予フィット数。この間は広い許容幅で
# オンセット候補を採用して素早く再安定させる
ONSET_RELOCK_GRACE = 6

# EMAへのアンカーを維持できる連続不採用回数。これを超えると
# 従来どおり回帰に従う
ONSET_MISS_LIMIT = 2

# 1回のfitで動かす最大幅(10%)。WaveToneのように
# 表示の急変を許容してでも素早く正確なBPMにロックさせる。
ONSET_MAX_STEP = 0.10

# 周期候補の移動平均係数(大きいほど速く追従)。
# 0.9ならほぼ1〜2タップで目標周期に収束する
ONSET_EMA_ALPHA = 0.9

# 滑らかな収束をリセットする(前回出力vs回帰の)乖離幅
ONSET_GLIDE_RESET_DEV = 0.03

# 精密固定を開始するタップ数。適用可能(MIN_TAPS_FOR_APPLY)に
# なった瞬間からオンセット解析を効かせて早期安定化を図る
ONSET_MIN_TAPS = 4

# オンセット強度を見る窓(タップ範囲中心からの半径・秒)と距離減衰
ONSET_WINDOW_S = 30.0
ONSET_DECAY_S = 15.0

# 採用に必要な最小オンセット数。少なめでも解析を開始し
# スカスカな曲でも早期ロックできるように8に緩和
ONSET_MIN_COUNT = 8


def compute_onset_envelope(y, sr):
    """楽曲全体のオンセット強度(音量変化)を計算する。

    戻り値は (オンセット時刻列, 正規化した強度列)。
    失敗時は ([], []) を返す。
    """
    try:
        import numpy as np
        import librosa

        if y is None or len(y) == 0:
            return [], []

        y = np.asarray(y, dtype=np.float32)

        if sr > 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        hop = 256
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

        if len(env) < 8:
            return [], []

        scale = float(np.max(env))
        if scale <= 1e-9:
            return [], []

        peaks = librosa.util.peak_pick(
            env,
            pre_max=int(0.05 * sr / hop),
            post_max=int(0.05 * sr / hop),
            pre_avg=int(0.10 * sr / hop),
            post_avg=int(0.10 * sr / hop),
            delta=0.25 * scale,
            wait=max(2, int(0.05 * sr / hop))
        )

        if len(peaks) == 0:
            return [], []

        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
        strengths = [float(env[i]) / scale for i in peaks]

        return [float(t) for t in times], strengths
    except Exception:
        return [], []


def _median(values):
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) * 0.5


class TapTempoEngine:
    """高精度タップテンポ推定。
    
    完全にステートレスな設計に変更:
    1. ユーザーの最近のタップ(最大16回)から素の回帰BPM(raw_slope)を計算
    2. オンセット情報がある場合、raw_slope の ±15% の範囲で最も強い
       オンセット周期(DTFTのピーク)を探し、そこにピタッと吸着させる。
    3. ガウス窓(Prior)を使って人間のタップに近いピークを優先するため、
       別のテンポ(裏拍など)の強いピークに吸着して固定されるバグを防止。
    4. ステート(ロック状態など)を一切持たないため、途中でテンポが変わっても
       人間のタップの変化に合わせて自動で新しいピークへ即座に追従する。
    """
    def __init__(self):
        self.taps = []
        self._onset_times = None
        self._onset_strengths = None

    def reset(self):
        self.taps = []

    def set_onsets(self, times, strengths=None):
        if not times:
            self._onset_times = None
            self._onset_strengths = None
            return
        self._onset_times = [float(t) for t in times]
        if strengths is not None and len(strengths) == len(times):
            self._onset_strengths = [float(s) for s in strengths]
        else:
            self._onset_strengths = None

    def clear_onsets(self):
        self._onset_times = None
        self._onset_strengths = None

    def __len__(self):
        return len(self.taps)

    def add_tap(self, wall, audio_pos):
        if self.taps:
            last_wall = self.taps[-1][0]
            delta = wall - last_wall
            if delta < TAP_DEBOUNCE_S:
                return False
            # took.jp方式: 3秒(3000ms)でリセット
            if delta > 3.0:
                self.taps = []

        self.taps.append((float(wall), float(audio_pos)))
        # took.jp方式: 直近8回のタップのみを保持して回帰分析する
        if len(self.taps) > 8:
            self.taps.pop(0)
        return True

    def split_on_jump(self, wall, audio_pos, playing):
        if not (playing and self.taps):
            return
        last_wall, last_wall_pos = self.taps[-1]
        dw = wall - last_wall
        dp = audio_pos - last_wall_pos
        if dw > 1e-6 and abs(dp - dw) > JUMP_TOLERANCE_S:
            self.taps = []

    def fit(self):
        n = len(self.taps)
        if n < 2:
            return None

        walls = [p[0] for p in self.taps]
        apos = [p[1] for p in self.taps]

        # 1. 人間のタップから「大体のBPMと位相」を回帰分析で求める (took.jpの手法)
        t_mean = (n - 1) / 2.0
        wall_mean = sum(walls) / n
        apos_mean = sum(apos) / n

        cov_wall = 0.0
        cov_apos = 0.0
        var_x = 0.0

        for i in range(n):
            dx = i - t_mean
            cov_wall += dx * (walls[i] - wall_mean)
            cov_apos += dx * (apos[i] - apos_mean)
            var_x += dx * dx

        if var_x <= 0.0:
            return None

        slope_wall = cov_wall / var_x
        slope_apos = cov_apos / var_x
        if slope_wall <= 0 or slope_apos <= 0:
            return None

        # 人間のタップベースのBPMと位相
        raw_bpm = 60.0 / slope_wall
        raw_intercept_wall = wall_mean - slope_wall * t_mean
        raw_intercept_apos = apos_mean - slope_apos * t_mean

        counts = list(range(n))

        # 2. 自動解析データ（オンセット）があり、タップが安定(4回以上)している場合、波形ピークへスナップする
        # WaveToneの「全体自動解析のBPM」と「手動タップの位相」を組み合わせる最強のロジック
        if n >= 4 and self._onset_times and len(self._onset_times) >= 10:
            strengths = self._onset_strengths if self._onset_strengths else [1.0] * len(self._onset_times)
            
            # 【BPMの決定】: 曲全体のオンセットを用いて、極めてシャープで正確な「曲全体の真のBPM」を割り出す
            def global_score(period, use_prior=True):
                omega = 2.0 * math.pi / period
                re, im = 0.0, 0.0
                for t, s in zip(self._onset_times, strengths):
                    ang = omega * t
                    re += s * math.cos(ang)
                    im += s * math.sin(ang)
                mag = math.hypot(re, im)
                if use_prior:
                    # 5%の緩やかなガウス窓で、倍テンポ等の暴走を防ぎつつ真のピークを尊重する
                    prior = math.exp(-((period - slope_apos) / (0.05 * slope_apos))**2 / 2.0)
                    return mag * prior
                return mag

            # 2-A. グリッド探索（曲全体を使うためピークが非常に鋭くなる。解像度を300に上げて取りこぼしを防ぐ）
            span = 0.08 * slope_apos 
            steps = 300
            lo = slope_apos - span
            hi = slope_apos + span
            
            best_score = -1.0
            best_period = slope_apos
            
            for i in range(steps):
                period = lo + (hi - lo) * (i / (steps - 1))
                s = global_score(period, use_prior=True)
                if s > best_score:
                    best_score = s
                    best_period = period
            
            # 2-B. 三分探索で数学的に完璧なピークへミリ秒単位で完全着地させる
            lo = max(lo, best_period - span / steps)
            hi = min(hi, best_period + span / steps)
            
            for _ in range(30):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                if global_score(m1, use_prior=False) < global_score(m2, use_prior=False):
                    lo = m1
                else:
                    hi = m2
                    
            best_period = (lo + hi) / 2.0
            final_bpm = 60.0 / best_period
            
            # 【位相（オフセット）の決定】: 曲全体のBPMが確定したら、ユーザーがタップした「局所的な位置」の波形に位相を合わせる
            center_t = apos_mean
            re_loc, im_loc = 0.0, 0.0
            omega = 2.0 * math.pi / best_period
            local_count = 0
            
            for t, s in zip(self._onset_times, strengths):
                if abs(t - center_t) <= 15.0: # タップ前後15秒の波形だけを見る
                    # タップ中心に近い波形ほど強く引き寄せられるように距離減衰をかける
                    w = s * math.exp(-((t - center_t)**2) / (2.0 * 10.0 ** 2))
                    ang = omega * t
                    re_loc += w * math.cos(ang)
                    im_loc += w * math.sin(ang)
                    local_count += 1
            
            if local_count >= 4:
                # 局所的な波形のピーク位相を逆算
                phi_onset = math.atan2(im_loc, re_loc) / (2.0 * math.pi / best_period)
            else:
                # 局所的なオンセットが無い場合は人間のタップ位相をそのまま採用
                phi_onset = raw_intercept_apos % best_period
            
            # 人間の手動タップの位相（raw_intercept_apos）を、局所的な波形の真の位相にスナップさせる
            diff = raw_intercept_apos - phi_onset
            k_beats = round(diff / best_period)
            snapped_audio_intercept = phi_onset + k_beats * best_period
            
            # 壁時計（実時間）側の切片も同等に補正
            intercept_shift = snapped_audio_intercept - raw_intercept_apos
            snapped_wall_intercept = raw_intercept_wall + intercept_shift

            self._phase_ref = (apos, counts, [1.0] * n)
            
            return {
                "bpm": final_bpm,
                "phi_audio": snapped_audio_intercept,
                "phi_wall": snapped_wall_intercept,
                "n": n,
                "rms_ms": 0.0
            }

        # オンセット情報がない、またはタップが少ない場合は手動タップの回帰分析結果をそのまま返す
        self._phase_ref = (apos, counts, [1.0] * n)
        return {
            "bpm": raw_bpm,
            "phi_audio": raw_intercept_apos,
            "phi_wall": raw_intercept_wall,
            "n": n,
            "rms_ms": 0.0
        }

    def fit_phase(self, bpm):
        """BPMを固定した場合の最適なグリッド原点(秒)を返す。
        確定時にBPMを整数へ丸める際、打鍵位置への整合を保った
        まま位相だけを最良一致させるために使う。
        """
        ref = getattr(self, "_phase_ref", None)
        if ref is None or bpm <= 0: return None
        apos_list, count_list, weights = ref
        spb = 60.0 / bpm
        num = 0.0
        den = 0.0
        for a, c, w in zip(apos_list, count_list, weights):
            num += w * (a - c * spb)
            den += w
        if den <= 0.0: return None
        return num / den

    @staticmethod
    def beat_pulse(fit, now=None):
        """現在のビート位相に応じたパルス強度(0..1)を返す。"""
        if fit is None or fit["n"] < 3:
            return 0.0
        if now is None:
            now = time.perf_counter()
        spb = 60.0 / fit["bpm"]
        frac = ((now - fit["phi_wall"]) / spb) % 1.0
        return max(0.0, min(1.0, pow(2.718281828, -6.0 * frac)))
