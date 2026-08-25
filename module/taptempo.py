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

    タップ時刻列を「累積拍数 vs 経過時間」の回帰直線で近似する。
    隣接間隔の平均と比べ、人間の打鍵ジッタ(±10〜30ms)の影響が
    O(1/N^3)で減衰するため、少ないタップ数でも極めて高精度に
    BPMを推定できる。外れ値(空打ち・遅れ)はMADベースで除外し、
    直近のタップに指数的重みを付けてテンポドリフトにも追従する。
    """

    def __init__(self):
        # (perf_counter秒, その瞬間のオーディオ位置秒)
        self.taps = []
        self._phase_ref = None

    def reset(self):
        self.taps = []
        self._phase_ref = None

    def __len__(self):
        return len(self.taps)

    def add_tap(self, wall, audio_pos):
        if self.taps:
            last_wall = self.taps[-1][0]
            delta = wall - last_wall
            if delta < TAP_DEBOUNCE_S:
                return False
            if delta > TAP_RESET_GAP_S:
                self.taps = []

        self.taps.append((float(wall), float(audio_pos)))

        if len(self.taps) > MAX_TAPS:
            self.taps.pop(0)

        return True

    def split_on_jump(self, wall, audio_pos, playing):
        """再生中にシーク等でオーディオ位置が不連続になったら
        位相の信頼できない過去タップを捨てる。"""
        if not (playing and self.taps):
            return

        last_wall, last_pos = self.taps[-1]
        dw = wall - last_wall
        dp = audio_pos - last_pos

        if dw > 1e-6 and abs(dp - dw) > JUMP_TOLERANCE_S:
            self.taps = []

    @staticmethod
    def _weights(m):
        return [
            WEIGHT_DECAY ** (m - 1 - j)
            for j in range(m)
        ]

    def _classify(self):
        """タップ列を (walls, apos, counts, use) に整理する。

        counts は累積拍数(間隔の比から1拍/2拍/半拍を判定)。
        use は信頼できるタップのインデックス。"""
        pts = self.taps
        n = len(pts)

        if n < 2:
            return None

        walls = [p[0] for p in pts]
        apos = [p[1] for p in pts]

        if n == 2:
            dw = walls[1] - walls[0]
            if dw <= 0.0:
                return None
            return walls, apos, [0.0, 1.0], [0, 1]

        gaps = [
            walls[i + 1] - walls[i]
            for i in range(n - 1)
        ]

        med = _median(gaps)
        if med <= 0.0:
            return None

        # 間隔の比から各タップ間の拍数を分類する。
        # 1拍おき/2拍抜かし/裏拍タップにも対応。
        counts = [0.0]
        valid = [True] * n

        for g in gaps:
            ratio = g / med
            k = None

            if 0.72 <= ratio <= 1.35:
                k = 1.0
            elif 1.55 <= ratio <= 2.7:
                k = 2.0
            elif 0.40 <= ratio <= 0.68:
                k = 0.5

            if k is None:
                valid[len(counts)] = False
                counts.append(counts[-1])
            else:
                counts.append(counts[-1] + k)

        use = [i for i in range(n) if valid[i]]
        if len(use) < 3:
            use = list(range(n))

        return walls, apos, counts, use

    @staticmethod
    def _weighted_lstsq(xs, ts, weights):
        sw = sum(weights)
        if sw <= 0.0:
            return None

        mx = sum(w * x for w, x in zip(weights, xs)) / sw
        mt = sum(w * t for w, t in zip(weights, ts)) / sw

        sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
        sxt = sum(
            w * (x - mx) * (t - mt)
            for w, x, t in zip(weights, xs, ts)
        )

        if sxx <= 1e-12:
            return None

        slope = sxt / sxx

        return slope, mt - slope * mx

    def _fit_indices(self, idxs, walls, counts):
        """idxsで選ばれたタップに重み付き最小二乗回帰を適用。
        (slope, intercept, residuals, rms) を返す。"""
        m = len(idxs)
        base = counts[idxs[0]]

        xs = [counts[i] - base for i in idxs]
        ts = [walls[i] for i in idxs]

        weights = self._weights(m)

        fit = self._weighted_lstsq(xs, ts, weights)
        if fit is None:
            return None

        slope, intercept = fit

        residuals = [
            t - (intercept + slope * x)
            for x, t in zip(xs, ts)
        ]

        rms = (
            sum(r * r for r in residuals) / m
        ) ** 0.5

        return slope, intercept, residuals, rms

    def fit(self):
        """現在のタップから (bpm, phi_audio, n, rms_ms) を推定する。

        phi_audio は「拍番号0の拍がオーディオ上の何秒に相当するか」。
        戻り値Noneは推定に十分なデータがないことを示す。
        """
        prep = self._classify()
        if prep is None:
            return None

        walls, apos, counts, use = prep
        n = len(pts := self.taps)

        if n == 2:
            dw = walls[1] - walls[0]
            return {
                "bpm": 60.0 / dw,
                "phi_audio": apos[0],
                "phi_wall": walls[0],
                "n": 2,
                "rms_ms": 0.0,
            }

        result = self._fit_indices(use, walls, counts)
        if result is None:
            return None

        slope, intercept, residuals, rms = result

        # 残差の大きな外れ値(空打ち等)を除外して再フィット
        threshold = max(0.045, 3.0 * rms)
        use2 = [
            i
            for i, r in zip(use, residuals)
            if abs(r) <= threshold
        ]

        if MIN_TAPS_FOR_APPLY <= len(use2) < len(use):
            refit = self._fit_indices(use2, walls, counts)
            if refit is not None:
                slope, intercept, _, rms = refit
                use = use2

        if slope <= 0.05 or slope > 4.0:
            return None

        i0 = use[0]

        # 確定時にBPMを整数へ丸えた場合のグリッド原点再計算に
        # 使うデータ(使用タップの音声位置・累積拍数・重み)を保持
        m = len(use)
        base = counts[use[0]]

        self._phase_ref = (
            [apos[i] for i in use],
            [counts[i] - base for i in use],
            self._weights(m),
        )

        return {
            "bpm": 60.0 / slope,
            "phi_audio": apos[i0] + (intercept - walls[i0]),
            "phi_wall": intercept,
            "n": m,
            "rms_ms": rms * 1000.0,
        }

    def fit_phase(self, bpm):
        """BPMを固定した場合の最適なグリッド原点(秒)を返す。

        確定時にBPMを整数へ丸める際、打鍵位置への整合を保った
        まま位相だけを最良一致させるために使う。
        fit() が呼ばれている必要がある。
        """
        ref = getattr(self, "_phase_ref", None)

        if ref is None or bpm <= 0:
            return None

        apos_list, count_list, weights = ref

        spb = 60.0 / bpm

        num = 0.0
        den = 0.0

        for a, c, w in zip(apos_list, count_list, weights):
            num += w * (a - c * spb)
            den += w

        if den <= 0.0:
            return None

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
