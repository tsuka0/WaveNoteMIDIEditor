"""
Voice Library & WORLD Synthesizer Module for WaveNote MIDI Editor.

原音ライブラリ (△_○○.wav) の管理および WORLD ボコーダー (pyworld) による
ピッチシフト合成・多層キャッシュシステムを提供します。
"""

import os
import re
import sys
import time
import hashlib
import threading
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import pyworld as pw
import scipy.signal
import scipy.interpolate

from .settings import load_value, save_value

NOTE_TO_SEMITONE = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

SEMITONE_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ひらがな・カタカナ相互変換用
_HIRA_START, _HIRA_END = ord('ぁ'), ord('ゖ')
_KATA_START, _KATA_END = ord('ァ'), ord('ヶ')

def to_katakana(text: str) -> str:
    return "".join(chr(ord(c) + 96) if _HIRA_START <= ord(c) <= _HIRA_END else c for c in text)

def to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 96) if _KATA_START <= ord(c) <= _KATA_END else c for c in text)

def normalize_text(text: str) -> str:
    """全角記号や全角英数を半角に正規化する。"""
    text = text.strip()
    text = text.replace('＃', '#').replace('♯', '#').replace('♭', 'b').replace('＿', '_')
    return text.translate(str.maketrans(
        '０１２３４５６７８９ＡＢＣＤＥＦＧａｂｃｄｅｆｇ',
        '0123456789ABCDEFGabcdefg'
    ))


def note_name_to_midi(name: str) -> int:
    """音名 (例: 'D#4', 'C4', 'A3') または MIDI 番号文字列 (例: '60') を MIDI ノート番号に変換する。"""
    name = normalize_text(name)
    if name.isdigit():
        return max(0, min(127, int(name)))

    m = re.match(r"^([A-Ga-g][#b]?)(-?\d+)$", name)
    if not m:
        raise ValueError(f"Invalid note name: {name}")
    note = m.group(1).upper()
    octave = int(m.group(2))
    semitone = NOTE_TO_SEMITONE.get(note)
    if semitone is None:
        raise ValueError(f"Unknown note name: {note}")
    return (octave + 1) * 12 + semitone


def midi_to_note_name(pitch: int) -> str:
    """MIDI ノート番号 (例: 63, 60) を音名 (例: 'D#4', 'C4') に変換する。"""
    octave = (pitch // 12) - 1
    semitone = pitch % 12
    return f"{SEMITONE_TO_NOTE[semitone]}{octave}"


class VoiceLibrary:
    """原音ライブラリ管理クラス。

    △_○○.wav (△=発音、○○=音程) のスキャン、
    WORLD による分析 (f0, sp, ap) のキャッシュ、
    ピッチシフト合成波形のキャッシュ、
    およびバックグラウンドでの事前レンダリング (Pre-warming) を行います。
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "VoiceLibrary":
        if cls._instance is None:
            cls._instance = VoiceLibrary()
        return cls._instance

    def __init__(self, folder_path: Optional[str] = None):
        VoiceLibrary._instance = self
        self.folder_path = folder_path or load_value("voice_library_dir", "")
        # samples[phoneme][midi_pitch] = file_path
        self.samples: Dict[str, Dict[int, str]] = {}
        # キャッシュ: file_path -> (raw_audio, fs, f0, sp, ap, t)
        self._feature_cache: Dict[str, Tuple] = {}
        # キャッシュ: (file_path, shift_semitones, target_sr) -> np.ndarray (float32)
        self._wave_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._prewarm_thread: Optional[threading.Thread] = None
        self._prewarm_cancel = False
        self._disk_cache_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "WaveNoteMIDIEditor",
            "world_cache"
        )
        try:
            os.makedirs(self._disk_cache_dir, exist_ok=True)
        except Exception:
            pass

        if self.folder_path and os.path.isdir(self.folder_path):
            self.scan_library()

    def set_folder(self, folder_path: str):
        """音声ライブラリフォルダを設定し、スキャンする。"""
        self.folder_path = folder_path
        save_value("voice_library_dir", folder_path)
        self.clear_cache()
        self.scan_library()

    def clear_cache(self):
        """メモリキャッシュをクリアする。"""
        with self._cache_lock:
            self._feature_cache.clear()
            self._wave_cache.clear()

    def scan_library(self):
        """指定フォルダ内の音声ファイル (△_○○.wav 等) をスキャンしてインデックス化する。"""
        self.samples.clear()
        if not self.folder_path or not os.path.isdir(self.folder_path):
            return

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext != ".wav":
                    continue

                stem = os.path.splitext(file)[0]
                norm_stem = normalize_text(stem)

                # パターン1: △_○○ (区切り文字: _ または -)
                m = re.match(r"^(.+)[_\-]([A-Ga-g][#b]?-?\d+|\d+)$", norm_stem)
                if m:
                    phoneme = m.group(1).strip()
                    pitch_str = m.group(2).strip()
                    try:
                        pitch = note_name_to_midi(pitch_str)
                    except ValueError:
                        pitch = 60
                else:
                    # パターン2: 音程指定なしの単一ファイル (例: あ.wav) -> デフォルト C4 (60)
                    phoneme = stem.strip()
                    pitch = 60

                if not phoneme:
                    continue

                full_path = os.path.join(root, file)
                if phoneme not in self.samples:
                    self.samples[phoneme] = {}
                self.samples[phoneme][pitch] = full_path

    def get_default_phoneme(self) -> Optional[str]:
        """ライブラリ内のデフォルト発音 (「あ」または最初の登録発音) を返す。"""
        if not self.samples:
            return None
        for cand in ("あ", "ア", "a", "A"):
            if cand in self.samples:
                return cand
        return next(iter(self.samples.keys()))

    def resolve_phoneme(self, phoneme: str) -> Optional[str]:
        """発音表記 (ひらがな/カタカナ/空文字) をライブラリ内の登録キーに解決する。"""
        if not self.samples:
            return None

        phoneme = (phoneme or "").strip()
        if not phoneme:
            return self.get_default_phoneme()

        # 1. 完全一致
        if phoneme in self.samples:
            return phoneme

        # 2. ひらがな -> カタカナ
        kata = to_katakana(phoneme)
        if kata in self.samples:
            return kata

        # 3. カタカナ -> ひらがな
        hira = to_hiragana(phoneme)
        if hira in self.samples:
            return hira

        # 4. デフォルト発音へフォールバック
        return self.get_default_phoneme()

    def has_phoneme(self, phoneme: str) -> bool:
        """指定した発音 (または解決可能な発音) の原音が登録されているかを返す。"""
        return self.resolve_phoneme(phoneme) is not None

    def find_best_sample(self, phoneme: str, target_pitch: int) -> Optional[Tuple[str, int, int]]:
        """指定した発音とターゲット音高に最も近い原音ファイルを検索する。

        Returns:
            (file_path, base_pitch, shift_semitones) または None
        """
        resolved_phoneme = self.resolve_phoneme(phoneme)
        if not resolved_phoneme or resolved_phoneme not in self.samples or not self.samples[resolved_phoneme]:
            return None

        available_pitches = list(self.samples[resolved_phoneme].keys())
        best_pitch = min(available_pitches, key=lambda p: abs(p - target_pitch))
        file_path = self.samples[resolved_phoneme][best_pitch]
        shift_semitones = target_pitch - best_pitch
        return file_path, best_pitch, shift_semitones

    def get_sample_duration(self, phoneme: str, target_pitch: int) -> float:
        """指定した発音とピッチに最適な原音サンプルの元ファイルの長さ（秒）を返す。"""
        match = self.find_best_sample(phoneme, target_pitch)
        if not match:
            return 0.0
        file_path = match[0]
        feat = self.extract_world_features(file_path)
        if feat is not None:
            raw_audio, fs, _, _, _, _ = feat
            return len(raw_audio) / float(fs)
        return 0.0

    _CACHE_VERSION = "v8_silky_clear"

    def _get_file_hash(self, file_path: str) -> str:
        try:
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            return hashlib.md5(f"{self._CACHE_VERSION}_{file_path}_{mtime}_{size}".encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.md5(f"{self._CACHE_VERSION}_{file_path}".encode("utf-8")).hexdigest()

    def _load_disk_features(self, file_path: str) -> Optional[Tuple]:
        file_hash = self._get_file_hash(file_path)
        cache_file = os.path.join(self._disk_cache_dir, f"{file_hash}.npz")
        if not os.path.isfile(cache_file):
            return None
        try:
            with np.load(cache_file) as data:
                raw_audio = data["raw_audio"]
                fs = int(data["fs"])
                f0 = data["f0"]
                sp = data["sp"]
                ap = data["ap"]
                t = data["t"]
                return raw_audio, fs, f0, sp, ap, t
        except Exception:
            return None

    def _save_disk_features(self, file_path: str, raw_audio: np.ndarray, fs: int, f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, t: np.ndarray):
        file_hash = self._get_file_hash(file_path)
        cache_file = os.path.join(self._disk_cache_dir, f"{file_hash}.npz")
        try:
            np.savez_compressed(cache_file, raw_audio=raw_audio, fs=fs, f0=f0, sp=sp, ap=ap, t=t)
        except Exception:
            pass

    @staticmethod
    def ultra_clean_audio(audio: np.ndarray, fs: int = 44100, max_slope: Optional[float] = None) -> np.ndarray:
        """歌声の自然な倍音・エア感・抜け（8kHz〜16kHz）を100%保持しながら、
        単発デジタルのスパイク・クリック・超音波高周波・DCオフセットを完全に除去するマスタリングクオリティDSP。

        1. 外れ値インパルス（単発クリックスパイク）の検出 & 線形補間（正常な声帯振動や子音は一切削らない）
        2. 40Hz 2次ハイパスフィルター（サブソニック・DCバイアス除去）
        3. 16.5kHz 2次ローパスフィルター（歌声の高域倍音・空気感を損なわず、DACエイリアシング・超高周波ノイズのみ除去）
        """
        if len(audio) < 4:
            return audio
        y = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        try:
            # 1. 単発デジタルクリックスパイクの精密検出 & 補間
            # 正常な母音の声帯閉鎖では連続波形としての相関があるが、
            # 孤立した1サンプルのグリッチ・クリッピング飛びは局所ラプラシアンで極端な外れ値となる
            residual = y[1:-1] - 0.5 * (y[:-2] + y[2:])
            std_res = np.std(residual)
            if std_res > 1e-6:
                # 10シグマかつ絶対値0.40以上の明らかな外れ値のみピンポイント修復
                threshold = max(0.40, 10.0 * float(std_res))
                spike_idx = np.where(np.abs(residual) > threshold)[0] + 1
                for idx in spike_idx:
                    if 1 <= idx < len(y) - 1:
                        y[idx] = 0.5 * (y[idx - 1] + y[idx + 1])

            nyq = 0.5 * fs
            # 2. 40Hz 2次ハイパスフィルター (DCバイアス & サブソニック除去)
            if nyq > 50.0:
                sos_hp = scipy.signal.butter(2, 40.0 / nyq, btype="highpass", output="sos")
                y = scipy.signal.sosfilt(sos_hp, y).astype(np.float32)

            # 3. 16.5kHz 2次ローパスフィルター (歌声の空気感・子音を完全保存し超音波ノイズのみ遮断)
            cutoff = min(16500.0, nyq * 0.95)
            if cutoff < nyq:
                sos_lp = scipy.signal.butter(2, cutoff / nyq, btype="lowpass", output="sos")
                y = scipy.signal.sosfilt(sos_lp, y).astype(np.float32)
        except Exception:
            pass

        return y

    @staticmethod
    def apply_spectral_clarity(y: np.ndarray, fs: int, shift_semitones: int) -> np.ndarray:
        """ピッチシフト合成後の「ごもごも音・濁り・籠もり感」および「中高音域のからから音・擦れ音」を消去するスペクトルプロセッサ。

        1. 40Hz サブソニックハイパス（超低周波ゴロゴロ音・DCバイアス除去）
        2. 適応型ディマッド（De-Mudding: ピッチ降下時に肥大化する350Hz前後の籠もり帯域を適応低減）
        3. 適応型ディハーシュ（De-Harshing: ピッチ上昇・中高音域で過剰共鳴する3.4kHz付近の乾いたからから音・擦れ感を滑らかに補正）
        """
        if len(y) == 0:
            return y
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        try:
            # 1. 40Hz 2次ハイパスフィルター
            sos_hp = scipy.signal.butter(2, 40.0, btype="highpass", fs=fs, output="sos")
            y = scipy.signal.sosfilt(sos_hp, y).astype(np.float32)

            # 2. 適応型ディマッド（De-mudding: ピッチが下がるほど 350Hz 前後の濁りを適応低減）
            if shift_semitones < 0:
                gain_db = max(-4.0, shift_semitones * 0.8)
                fc = 350.0
                w0 = 2.0 * np.pi * fc / fs
                alpha = np.sin(w0) / (2.0 * 1.0)
                A = 10.0 ** (gain_db / 40.0)
                b0 = 1.0 + alpha * A
                b1 = -2.0 * np.cos(w0)
                b2 = 1.0 - alpha * A
                a0 = 1.0 + alpha / A
                a1 = -2.0 * np.cos(w0)
                a2 = 1.0 - alpha / A
                b = np.array([b0, b1, b2], dtype=np.float64) / a0
                a = np.array([a0, a1, a2], dtype=np.float64) / a0
                y = scipy.signal.lfilter(b, a, y).astype(np.float32)

            # 3. 適応型ディハーシュ（De-harshing: 中高音域へピッチが上がる際の3.4kHz「からから音・耳に刺さる乾いた擦れ」を適応除去）
            elif shift_semitones > 0:
                gain_db = max(-3.2, -shift_semitones * 0.40)
                fc = 3400.0
                w0 = 2.0 * np.pi * fc / fs
                alpha = np.sin(w0) / (2.0 * 1.2)
                A = 10.0 ** (gain_db / 40.0)
                b0 = 1.0 + alpha * A
                b1 = -2.0 * np.cos(w0)
                b2 = 1.0 - alpha * A
                a0 = 1.0 + alpha / A
                a1 = -2.0 * np.cos(w0)
                a2 = 1.0 - alpha / A
                b = np.array([b0, b1, b2], dtype=np.float64) / a0
                a = np.array([a0, a1, a2], dtype=np.float64) / a0
                y = scipy.signal.lfilter(b, a, y).astype(np.float32)
        except Exception:
            pass

        return y

    @staticmethod
    def normalize_and_maximize_audio(
        audio: np.ndarray,
        target_rms: float = 0.22,
        target_peak: float = 0.95
    ) -> np.ndarray:
        """音声を均一なパワー感（実効音量）に揃え、歪みのない最大音量（ピーク0.95）に最大化する。"""
        if len(audio) == 0:
            return audio

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak <= 1e-6:
            return audio

        abs_audio = np.abs(audio)
        sorted_abs = np.sort(abs_audio)
        active_portion = sorted_abs[len(sorted_abs) // 2:]
        active_rms = np.sqrt(np.mean(active_portion ** 2))

        if active_rms > 1e-4:
            rms_gain = target_rms / active_rms
            gain = min(rms_gain, target_peak / peak)
            audio = audio * gain

        # ピークを target_peak (0.95) まで最大化
        current_peak = np.max(np.abs(audio))
        if current_peak > 1e-6:
            audio = audio * (target_peak / current_peak)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _synthesize_psola(
        self,
        file_path: str,
        shift_semitones: int,
        target_sr: int = 44100,
        target_duration: Optional[float] = None,
        base_pitch: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """élastique Soloist と同原理の TD-PSOLA (Pitch-Synchronous Overlap-Add) による超低ノイズ合成。

        原音の生波形を保ちながらピッチ周期に同期して重ね合わせるため、
        ボコーダー特有の再合成ノイズ（カサカサ感・金属バズ）が原理的に皆無になります。
        """
        try:
            import parselmouth
            from parselmouth.praat import call
        except ImportError:
            return None

        try:
            x, fs = sf.read(file_path, dtype="float64")
            if x.ndim > 1:
                x = np.mean(x, axis=1)
            x = np.ascontiguousarray(x, dtype=np.float64)

            # 前処理1: DC除去 + 端部微小フェード (3ms)
            x = x - np.mean(x)
            fade_len = min(len(x) // 4, max(4, int(fs * 0.003)))
            if fade_len > 1:
                fade_curve = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float64)))
                x[:fade_len] *= fade_curve
                x[-fade_len:] *= fade_curve[::-1]

            # 前処理2: 原音収録スパイク・クリックの事前クリーン
            x = self.ultra_clean_audio(x, int(fs)).astype(np.float64)

            # 前処理3: 原音末尾の減衰・フライ・カスレ区間トリミング（未有声化によるピッチ誤判定＆からから音の根本遮断）
            block = max(32, int(fs * 0.01))
            rms_blocks = [np.sqrt(np.mean(x[i:i+block]**2)) for i in range(0, len(x)-block, block)]
            if rms_blocks:
                max_rms = np.max(rms_blocks)
                active_blocks = [i for i, r in enumerate(rms_blocks) if r > 0.12 * max_rms]
                if active_blocks and (active_blocks[-1] + 1) * block < len(x):
                    last_active = min(len(x), (active_blocks[-1] + 1) * block)
                    x = x[:last_active].copy()
                    tail_fade = min(len(x) // 4, max(4, int(fs * 0.005)))
                    x[-tail_fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tail_fade, dtype=np.float64)))

            # numpy 配列から直接 Sound オブジェクト作成（日本語パスでも安全）
            sound = parselmouth.Sound(values=x, sampling_frequency=float(fs))
            orig_dur = sound.duration
            stretch_needed = target_duration is not None and target_duration > (orig_dur + 0.02)

            # ピッチ探索範囲の動的計算 (原音F0の[0.65, 1.55]範囲に限定し、サブハーモニクス跳躍によるガラガラ・からから音を完全防止)
            if base_pitch is not None:
                orig_f0 = 440.0 * (2.0 ** ((base_pitch - 69) / 12.0))
            else:
                try:
                    pitch_obj = call(sound, "To Pitch", 0.0, 50.0, 1000.0)
                    orig_f0 = call(pitch_obj, "Get mean", 0, 0, "Hertz")
                    if np.isnan(orig_f0) or orig_f0 < 40.0:
                        orig_f0 = 220.0
                except Exception:
                    orig_f0 = 220.0

            pitch_ratio = 2.0 ** (shift_semitones / 12.0)
            f0_min = max(55.0, orig_f0 * 0.65)
            f0_max = min(1600.0, orig_f0 * 1.55)

            manipulation = call(sound, "To Manipulation", 0.005, f0_min, f0_max)

            # ピッチシフト
            if shift_semitones != 0:
                pitch_tier = call(manipulation, "Extract pitch tier")
                call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, pitch_ratio)
                call([manipulation, pitch_tier], "Replace pitch tier")

            # 子音保護型サステイン時間伸長 (安定有声母音のみを伸長)
            if stretch_needed and target_duration is not None:
                duration_tier = call(manipulation, "Extract duration tier")
                call(duration_tier, "Remove points between", sound.xmin, sound.xmax)

                t1 = min(0.04, orig_dur * 0.15)
                t2 = max(t1 + 0.03, orig_dur * 0.85)
                ramp_len = min(0.02, max(0.005, (t2 - t1) * 0.15))
                denom = (t2 - t1) - 0.5 * ramp_len
                target_dur_with_margin = target_duration * 1.08
                R = (target_dur_with_margin - (orig_dur - (t2 - t1) + 0.5 * ramp_len)) / max(0.001, denom)
                R = max(1.0, R)

                call(duration_tier, "Add point", sound.xmin, 1.0)
                call(duration_tier, "Add point", sound.xmin + t1, 1.0)
                call(duration_tier, "Add point", sound.xmin + t1 + ramp_len, R)
                call(duration_tier, "Add point", sound.xmin + t2 - ramp_len, R)
                call(duration_tier, "Add point", sound.xmin + t2, 1.0)
                call(duration_tier, "Add point", sound.xmax, 1.0)
                call([manipulation, duration_tier], "Replace duration tier")

            resynth = call(manipulation, "Get resynthesis (overlap-add)")
            y = resynth.values[0].astype(np.float32)

            # 高品質ポリフェーズリサンプリング
            if fs != target_sr:
                from math import gcd
                g = gcd(int(fs), int(target_sr))
                up = int(target_sr // g)
                down = int(fs // g)
                y = scipy.signal.resample_poly(y, up, down).astype(np.float32)

            # 目標時間への確実な到達保証（整数ピッチ周期クロスフェードループ）
            if target_duration is not None:
                req_samples = int(target_duration * target_sr)
                if len(y) < req_samples:
                    shortage = req_samples - len(y)
                    target_p = 60 if base_pitch is None else (base_pitch + shift_semitones)
                    f_target = max(50.0, 440.0 * (2.0 ** ((target_p - 69) / 12.0)))
                    period_samp = max(16, int(round(target_sr / f_target)))
                    num_periods = max(4, int(0.08 * target_sr / period_samp))
                    loop_len = num_periods * period_samp
                    loop_start = max(0, len(y) - loop_len - int(0.03 * target_sr))
                    loop_end = loop_start + loop_len
                    if loop_end <= len(y):
                        loop_seg = y[loop_start:loop_end]
                        reps = (shortage // loop_len) + 2
                        extended_loop = np.tile(loop_seg, reps)
                        xfade_len = min(len(loop_seg) // 2, period_samp * 2)
                        xfade_out = np.cos(np.linspace(0, np.pi / 2, xfade_len, dtype=np.float32)) ** 2
                        xfade_in = np.sin(np.linspace(0, np.pi / 2, xfade_len, dtype=np.float32)) ** 2
                        y[-xfade_len:] = y[-xfade_len:] * xfade_out + extended_loop[:xfade_len] * xfade_in
                        y = np.concatenate([y, extended_loop[xfade_len:shortage + xfade_len]])

            # 終端フェードアウト (15ms) による波形末尾のクリックノイズ完全防止
            fade_out_samples = min(len(y) // 4, max(8, int(target_sr * 0.015)))
            if fade_out_samples > 1:
                fade_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out_samples, dtype=np.float32)))
                y[-fade_out_samples:] *= fade_out

            # スペクトル明瞭度プロセッサ（ごもごも感・籠もりの完全消去）
            y = self.apply_spectral_clarity(y, target_sr, shift_semitones)

            # 合成後マスタリングクリーナー（超音波ノイズのみ除去、ボーカルの抜け・高域を完全保持）
            y = self.ultra_clean_audio(y, target_sr)

            # 音量均一化 ＆ ピーク最大化 (0.95)
            y = self.normalize_and_maximize_audio(y, target_peak=0.95)
            return y
        except Exception as e:
            return None

    def extract_world_features(self, file_path: str) -> Optional[Tuple]:
        """WORLD分析を実行（またはキャッシュからロード）して (raw_audio, fs, f0, sp, ap, t) を返す。

        超低ノイズ化のため：
        1. 直流バイアス (DCオフセット) 除去
        2. 原音端部のステップ不連続を防ぐ微小コサインフェード (3ms)
        3. 高精度 Harvest + Stonemask によるピッチ跳躍・誤判定防止
        4. 有声区間内 F0 ドロップアウト補間
        5. D4C 非周期性指標 (AP) の過剰ノイズ抑制
        """
        with self._cache_lock:
            if file_path in self._feature_cache:
                return self._feature_cache[file_path]

        # ディスクキャッシュ確認
        disk_feat = self._load_disk_features(file_path)
        if disk_feat is not None:
            with self._cache_lock:
                self._feature_cache[file_path] = disk_feat
            return disk_feat

        if not os.path.isfile(file_path):
            return None

        try:
            x, fs = sf.read(file_path, dtype="float64")
            if x.ndim > 1:
                x = np.mean(x, axis=1)
            x = np.ascontiguousarray(x, dtype=np.float64)

            # 前処理1: DCオフセット除去
            x = x - np.mean(x)

            # 前処理2: サンプルの先頭・末尾の不連続破裂音・クリックを防ぐ微小フェード (3ms)
            fade_len = min(len(x) // 4, max(4, int(fs * 0.003)))
            if fade_len > 1:
                fade_curve = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float64)))
                x[:fade_len] *= fade_curve
                x[-fade_len:] *= fade_curve[::-1]

            # 前処理3: 原音収録スパイク・クリック・超高周波バズの完全消去
            x = VoiceLibrary.ultra_clean_audio(x, int(fs)).astype(np.float64)

            # WORLD 特徴量抽出: DIOより圧倒的に高精度な Harvest を使用
            frame_period = 5.0
            _f0, t = pw.harvest(x, fs, frame_period=frame_period, f0_floor=70.0, f0_ceil=800.0)
            f0 = pw.stonemask(x, _f0, t, fs)

            # 有声区間内部のF0ドロップアウト（1〜2フレームの途切れによるジリジリ音）を有声区間補間
            v_indices = np.where(f0 > 0)[0]
            if len(v_indices) > 2:
                first_v, last_v = v_indices[0], v_indices[-1]
                seg = f0[first_v:last_v + 1].copy()
                zero_mask = (seg <= 0)
                pos_mask = (seg > 0)
                if np.any(zero_mask) and np.sum(pos_mask) > 1:
                    x_all = np.arange(len(seg))
                    seg[zero_mask] = np.interp(x_all[zero_mask], x_all[pos_mask], seg[pos_mask])
                    f0[first_v:last_v + 1] = seg

            sp = pw.cheaptrick(x, f0, t, fs)
            ap = pw.d4c(x, f0, t, fs, threshold=0.75)

            # 有声フレームにおける過剰な非周期ノイズ（金属的なカサカサ音）の抑制
            for i in range(len(f0)):
                if f0[i] > 0:
                    ap[i] = np.clip(ap[i], 0.0, 0.95)

            result = (x.astype(np.float32), fs, f0, sp, ap, t)

            with self._cache_lock:
                self._feature_cache[file_path] = result

            self._save_disk_features(file_path, result[0], fs, f0, sp, ap, t)
            return result
        except Exception as e:
            print(f"Error extracting WORLD features for {file_path}: {e}")
            return None

    def get_shifted_waveform(
        self,
        phoneme: str,
        target_pitch: int,
        target_sr: int = 44100,
        target_duration: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """指定した発音をターゲット音程にピッチシフトした音声波形 (float32, 1D) を取得する。

        - élastique Soloist と同系列の TD-PSOLA (Pitch-Synchronous Overlap-Add) を最優先で使用。
          ボコーダー特有の再合成ノイズを極限まで排除し、クリアな原音波形を維持。
        - エラー時は Harvest 版 WORLD に自動フォールバック。
        - 全合成音声に対して音量均一化（Loudness Equalization）およびピーク最大化（0.95）を適用。
        """
        match = self.find_best_sample(phoneme, target_pitch)
        if not match:
            return None
        file_path, base_pitch, shift_semitones = match

        raw_dur = self.get_sample_duration(phoneme, target_pitch)
        stretch_needed = target_duration is not None and target_duration > (raw_dur + 0.02)
        dur_key = self._calc_dur_key(phoneme, target_pitch, target_duration)

        cache_key = (file_path, shift_semitones, target_sr, dur_key)
        with self._cache_lock:
            if cache_key in self._wave_cache:
                return self._wave_cache[cache_key]

        # 1. 最優先: élastique Soloist 並みの TD-PSOLA による超低ノイズ合成
        psola_wave = self._synthesize_psola(
            file_path=file_path,
            shift_semitones=shift_semitones,
            target_sr=target_sr,
            target_duration=target_duration if stretch_needed else None,
            base_pitch=base_pitch
        )
        if psola_wave is not None and len(psola_wave) > 0:
            with self._cache_lock:
                self._wave_cache[cache_key] = psola_wave
            return psola_wave

        # 2. シフトなし・伸長なしの原音フォールバック
        features = self.extract_world_features(file_path)
        if features is None:
            return None

        raw_audio, fs, f0, sp, ap, t = features
        if shift_semitones == 0 and not stretch_needed:
            audio = raw_audio.copy()
            if fs != target_sr:
                from math import gcd
                g = gcd(int(fs), int(target_sr))
                up = int(target_sr // g)
                down = int(fs // g)
                audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)
            else:
                audio = audio.astype(np.float32)

            audio = self.ultra_clean_audio(audio, target_sr)
            audio = self.normalize_and_maximize_audio(audio, target_peak=0.95)
            with self._cache_lock:
                self._wave_cache[cache_key] = audio
            return audio

        # 3. WORLD 合成フォールバック (Harvest + 子音保護型サステイン時間伸長)
        try:
            ratio = 2.0 ** (shift_semitones / 12.0)
            f0_shifted = np.ascontiguousarray(f0 * ratio, dtype=np.float64)
            f0_shifted[f0 <= 0] = 0.0

            orig_frames = len(f0)
            if stretch_needed and target_duration is not None:
                target_frames = max(orig_frames, int(round(target_duration / 0.005)))

                # 子音・アタック部 (先頭約50ms) およびリリース部 (末尾約50ms) を固定
                n_head = min(10, orig_frames // 4)
                n_tail = min(10, orig_frames // 4)
                n_sustain_target = target_frames - n_head - n_tail

                if n_sustain_target > 0 and orig_frames > (n_head + n_tail):
                    t_orig = np.concatenate([
                        np.arange(n_head, dtype=np.float64),
                        np.linspace(n_head, orig_frames - n_tail - 1, n_sustain_target, dtype=np.float64),
                        np.arange(orig_frames - n_tail, orig_frames, dtype=np.float64)
                    ])
                else:
                    t_orig = np.linspace(0.0, orig_frames - 1, target_frames, dtype=np.float64)

                x_frames = np.arange(orig_frames, dtype=np.float64)
                f0_shifted = np.interp(t_orig, x_frames, f0_shifted)
                f0_shifted[f0_shifted <= 10.0] = 0.0

                sp_c = np.empty((target_frames, sp.shape[1]), dtype=np.float64)
                ap_c = np.empty((target_frames, ap.shape[1]), dtype=np.float64)
                for col in range(sp.shape[1]):
                    sp_c[:, col] = np.interp(t_orig, x_frames, sp[:, col])
                    ap_c[:, col] = np.interp(t_orig, x_frames, ap[:, col])
            else:
                sp_c = np.ascontiguousarray(sp, dtype=np.float64)
                ap_c = np.ascontiguousarray(ap, dtype=np.float64)

            y = pw.synthesize(f0_shifted, sp_c, ap_c, fs, frame_period=5.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # サブソニック・ハイパスフィルター (45Hz)
            try:
                sos = scipy.signal.butter(2, 45.0, btype='highpass', fs=fs, output='sos')
                y = scipy.signal.sosfilt(sos, y).astype(np.float32)
            except Exception:
                y = (y - np.mean(y)).astype(np.float32)

            # 高品質ポリフェーズリサンプリング
            if fs != target_sr:
                from math import gcd
                g = gcd(int(fs), int(target_sr))
                up = int(target_sr // g)
                down = int(fs // g)
                y = scipy.signal.resample_poly(y, up, down).astype(np.float32)

            y = self.ultra_clean_audio(y, target_sr)
            # 音量均一化 ＆ 最大化 (0.95)
            y = self.normalize_and_maximize_audio(y, target_peak=0.95)

            with self._cache_lock:
                self._wave_cache[cache_key] = y
            return y
        except Exception as e:
            print(f"Error synthesizing WORLD audio: {e}")
            return None

    def _calc_dur_key(self, phoneme: str, target_pitch: int, target_duration: Optional[float]) -> float:
        """指定したターゲット時間に対応する正規化されたキャッシュキー（dur_key）を計算する。"""
        if not target_duration or target_duration <= 0.05:
            return 0.0
        raw_dur = self.get_sample_duration(phoneme, target_pitch)
        if raw_dur > 0.0 and target_duration <= (raw_dur + 0.02):
            return 0.0
        return round(target_duration, 2)

    def has_exact_cached_waveform(
        self,
        phoneme: str,
        target_pitch: int,
        target_sr: int = 44100,
        target_duration: Optional[float] = None
    ) -> bool:
        """指定した発音・ピッチ・長さの波形が厳密にキャッシュに存在するか（フォールバックなし）を返す。"""
        match = self.find_best_sample(phoneme, target_pitch)
        if not match:
            return False
        file_path, _, shift_semitones = match
        dur_key = self._calc_dur_key(phoneme, target_pitch, target_duration)
        cache_key = (file_path, shift_semitones, target_sr, dur_key)
        with self._cache_lock:
            return cache_key in self._wave_cache

    def get_cached_waveform(
        self,
        phoneme: str,
        target_pitch: int,
        target_sr: int = 44100,
        target_duration: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """キャッシュ済みの波形を取得する（厳密一致を優先し、未完成時は基本波形へフォールバック）。"""
        match = self.find_best_sample(phoneme, target_pitch)
        if not match:
            return None
        file_path, _, shift_semitones = match
        dur_key = self._calc_dur_key(phoneme, target_pitch, target_duration)
        cache_key = (file_path, shift_semitones, target_sr, dur_key)
        with self._cache_lock:
            if cache_key in self._wave_cache:
                return self._wave_cache[cache_key]
            # 伸長版が未完成でも、基本版があればリアルタイム再生用にフォールバック
            base_key = (file_path, shift_semitones, target_sr, 0.0)
            return self._wave_cache.get(base_key)

    def prewarm_notes(self, notes: list, target_sr: int = 44100, release_time: float = 0.06):
        """ノーツリストに含まれる発音・ピッチ・長さの組み合わせをバックグラウンドで事前合成する。

        厳密なキャッシュチェックを行い、基本波形だけでなく各ノーツの伸長波形（note.duration + release_time）
        も確実にスキップせず事前合成します。
        """
        needed_requests = set()
        for note in notes:
            lyric = getattr(note, "lyric", "").strip()
            resolved = self.resolve_phoneme(lyric)
            if resolved:
                dur = round(getattr(note, "duration", 0.0) + release_time, 2)
                needed_requests.add((resolved, note.pitch, dur))

        if not needed_requests:
            return

        def _worker(requests):
            for phoneme, pitch, dur in requests:
                if self._prewarm_cancel:
                    break
                # 基本波形を事前合成 (未作成の場合のみ)
                if not self.has_exact_cached_waveform(phoneme, pitch, target_sr, None):
                    self.get_shifted_waveform(phoneme, pitch, target_sr, None)
                # 伸長波形を事前合成 (未作成の場合のみ確実に合成)
                if dur > 0.0 and not self.has_exact_cached_waveform(phoneme, pitch, target_sr, dur):
                    self.get_shifted_waveform(phoneme, pitch, target_sr, dur)

        self._prewarm_cancel = False
        self._prewarm_thread = threading.Thread(
            target=_worker,
            args=(list(needed_requests),),
            daemon=True
        )
        self._prewarm_thread.start()

    def cancel_prewarm(self):
        """実行中のバックグラウンド事前合成スレッドを確実に停止する。"""
        self._prewarm_cancel = True
        t = getattr(self, "_prewarm_thread", None)
        if t is not None and t.is_alive() and t != threading.current_thread():
            t.join(timeout=0.3)
        self._prewarm_thread = None

    def prewarm_all_sync(
        self,
        notes: list,
        target_sr: int = 44100,
        release_time: float = 0.06
    ):
        """再生開始前に全ノーツの音声計算を同期的に100%完了させる。

        未計算のノーツを漏れなく全て計算・キャッシュし、再生中の計算処理・スレッド競合をゼロにする。
        """
        self.cancel_prewarm()
        for note in notes:
            lyric = getattr(note, "lyric", "").strip()
            resolved = self.resolve_phoneme(lyric)
            if not resolved:
                continue
            dur = round(getattr(note, "duration", 0.0) + release_time, 2)
            if not self.has_exact_cached_waveform(resolved, note.pitch, target_sr, dur):
                self.get_shifted_waveform(resolved, note.pitch, target_sr, dur)

    def prewarm_notes_sync(
        self,
        notes: list,
        target_sr: int = 44100,
        release_time: float = 0.06,
        max_notes: int = 30
    ):
        """再生開始直前などに、未キャッシュのノーツを高速に同期プリロードして再生中のアンダーランを根絶する。"""
        self.prewarm_all_sync(notes[:max_notes], target_sr, release_time)

    def open_folder(self) -> bool:
        """エクスプローラー等で音声ライブラリフォルダを開く。"""
        if not self.folder_path or not os.path.isdir(self.folder_path):
            return False

        try:
            if sys.platform == "win32":
                os.startfile(self.folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.folder_path])
            else:
                subprocess.Popen(["xdg-open", self.folder_path])
            return True
        except Exception as e:
            print(f"Failed to open folder: {e}")
            return False
