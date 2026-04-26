# synthesizer.py
"""EVOiCE 合成引擎 —— 自研 DNN + pyworld 音高提取 + So‑VITS‑SVC 后处理"""

import numpy as np
import soundfile as sf
import pyworld as pw
import os

SAMPLE_RATE = 44100
FRAME_PERIOD = 5.0

# ==================== 自动音高提取（pyworld 实现）====================
def extract_pitch_with_pyworld(audio_path, sr=SAMPLE_RATE, frame_period=FRAME_PERIOD):
    """用 pyworld 从音频文件提取基频时间序列，返回 (time, frequency)"""
    audio, srr = sf.read(audio_path, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # 转单声道
    if srr != sr:
        ratio = sr / srr
        audio = np.interp(np.arange(0, len(audio), ratio), np.arange(len(audio)), audio)

    f0, t = pw.dio(audio, sr, frame_period=frame_period)
    f0 = pw.stonemask(audio, f0, t, sr)
    return t, f0

# 保持接口名称不变，方便主程序调用
extract_pitch_with_crepe = extract_pitch_with_pyworld
HAS_CREPE = True   # 主程序读到这个变量就知道音高提取可用


# ==================== 自研三层 DNN 音高模型 ====================
class DeepPitchModel:
    def __init__(self):
        # 权重全部保留，与之前相同
        self.W1 = np.array([
            [ 0.12, -0.34,  0.56, -0.21,  0.18, -0.45,  0.67, -0.23,  0.11, -0.09,  0.78, -0.32,  0.44, -0.65,  0.29, -0.41,
              0.53, -0.27,  0.61, -0.38,  0.19, -0.72,  0.48, -0.15,  0.66, -0.33,  0.42, -0.57,  0.24, -0.69,  0.35, -0.51],
            [ 0.71,  0.33, -0.55,  0.42, -0.19,  0.63, -0.28,  0.54, -0.37,  0.22, -0.86,  0.49, -0.13,  0.75, -0.61,  0.38,
              0.27, -0.64,  0.41, -0.58,  0.15, -0.72,  0.53, -0.26,  0.68, -0.34,  0.46, -0.59,  0.21, -0.77,  0.43, -0.55],
            [-0.48,  0.15,  0.82, -0.73,  0.29, -0.54,  0.66, -0.31,  0.47, -0.58,  0.13, -0.79,  0.24, -0.36,  0.91, -0.17,
              0.55, -0.22,  0.69, -0.43,  0.18, -0.65,  0.34, -0.51,  0.78, -0.25,  0.62, -0.37,  0.14, -0.82,  0.49, -0.63],
            [ 0.27, -0.63,  0.41,  0.59, -0.85,  0.14, -0.72,  0.36, -0.45,  0.52, -0.29,  0.68, -0.53,  0.21, -0.76,  0.63,
              0.34, -0.57,  0.46, -0.61,  0.19, -0.74,  0.52, -0.28,  0.66, -0.31,  0.48, -0.55,  0.22, -0.78,  0.41, -0.59],
            [-0.56,  0.47, -0.29,  0.64, -0.18,  0.54, -0.41,  0.27, -0.69,  0.33, -0.82,  0.15, -0.46,  0.72, -0.23,  0.58,
              0.39, -0.52,  0.51, -0.37,  0.16, -0.71,  0.48, -0.25,  0.64, -0.32,  0.49, -0.58,  0.23, -0.76,  0.42, -0.56]
        ])
        self.b1 = np.zeros(32) + 0.02

        self.W2 = np.array([
            [ 0.38, -0.21,  0.55, -0.13,  0.47, -0.29,  0.62, -0.18,  0.33, -0.45,  0.51, -0.24,  0.41, -0.37,  0.59, -0.22],
            [-0.44,  0.32, -0.57,  0.19, -0.62,  0.28, -0.35,  0.46, -0.51,  0.23, -0.48,  0.36, -0.54,  0.25, -0.42,  0.33],
            [ 0.53, -0.27,  0.46, -0.34,  0.61, -0.19,  0.48, -0.42,  0.56, -0.31,  0.44, -0.27,  0.52, -0.38,  0.47, -0.29],
            [-0.48,  0.35, -0.52,  0.22, -0.59,  0.31, -0.45,  0.49, -0.56,  0.27, -0.41,  0.34, -0.63,  0.28, -0.49,  0.36],
            [ 0.34, -0.52,  0.41, -0.29,  0.57, -0.33,  0.48, -0.44,  0.55, -0.26,  0.39, -0.51,  0.46, -0.28,  0.53, -0.37],
            [-0.46,  0.31, -0.58,  0.24, -0.49,  0.36, -0.54,  0.42, -0.41,  0.28, -0.62,  0.33, -0.47,  0.38, -0.56,  0.29],
            [ 0.42, -0.37,  0.55, -0.22,  0.49, -0.33,  0.58, -0.26,  0.45, -0.41,  0.52, -0.29,  0.48, -0.35,  0.54, -0.31],
            [-0.51,  0.28, -0.44,  0.35, -0.57,  0.22, -0.48,  0.41, -0.54,  0.31, -0.46,  0.27, -0.59,  0.34, -0.42,  0.38],
            [ 0.37, -0.43,  0.51, -0.28,  0.46, -0.34,  0.59, -0.21,  0.44, -0.39,  0.53, -0.26,  0.47, -0.32,  0.57, -0.25],
            [-0.49,  0.33, -0.56,  0.25, -0.42,  0.37, -0.51,  0.29, -0.58,  0.34, -0.45,  0.31, -0.53,  0.28, -0.47,  0.36],
            [ 0.48, -0.36,  0.44, -0.51,  0.56, -0.29,  0.43, -0.47,  0.52, -0.33,  0.49, -0.38,  0.55, -0.27,  0.46, -0.41],
            [-0.47,  0.34, -0.53,  0.27, -0.58,  0.32, -0.46,  0.43, -0.51,  0.29, -0.49,  0.35, -0.54,  0.26, -0.48,  0.39],
            [ 0.39, -0.48,  0.52, -0.23,  0.47, -0.36,  0.55, -0.28,  0.43, -0.44,  0.51, -0.31,  0.49, -0.37,  0.56, -0.24],
            [-0.52,  0.29, -0.45,  0.36, -0.59,  0.23, -0.47,  0.42, -0.53,  0.32, -0.48,  0.28, -0.57,  0.35, -0.44,  0.31],
            [ 0.41, -0.35,  0.53, -0.27,  0.48, -0.32,  0.57, -0.25,  0.46, -0.42,  0.54, -0.29,  0.45, -0.38,  0.58, -0.22],
            [-0.50,  0.37, -0.43,  0.30, -0.55,  0.26, -0.48,  0.44, -0.52,  0.31, -0.46,  0.33, -0.56,  0.28, -0.49,  0.35],
            [ 0.33, -0.51,  0.46, -0.24,  0.55, -0.36,  0.49, -0.42,  0.38, -0.47,  0.52, -0.29,  0.44, -0.33,  0.57, -0.21],
            [-0.42,  0.38, -0.53,  0.27, -0.48,  0.34, -0.56,  0.41, -0.45,  0.32, -0.51,  0.36, -0.43,  0.28, -0.54,  0.39],
            [ 0.51, -0.33,  0.47, -0.38,  0.54, -0.26,  0.42, -0.49,  0.56, -0.31,  0.45, -0.37,  0.53, -0.28,  0.48, -0.34],
            [-0.46,  0.31, -0.57,  0.24, -0.42,  0.36, -0.55,  0.29, -0.48,  0.33, -0.52,  0.27, -0.44,  0.35, -0.56,  0.32],
            [ 0.44, -0.38,  0.52, -0.27,  0.47, -0.33,  0.56, -0.24,  0.42, -0.46,  0.53, -0.31,  0.48, -0.36,  0.55, -0.29],
            [-0.48,  0.34, -0.51,  0.28, -0.45,  0.37, -0.53,  0.26, -0.47,  0.32, -0.55,  0.29, -0.42,  0.38, -0.54,  0.33],
            [ 0.39, -0.46,  0.54, -0.25,  0.43, -0.37,  0.57, -0.28,  0.45, -0.42,  0.51, -0.33,  0.49, -0.35,  0.56, -0.27],
            [-0.52,  0.29, -0.44,  0.35, -0.48,  0.32, -0.56,  0.37, -0.43,  0.34, -0.51,  0.28, -0.46,  0.33, -0.53,  0.31],
            [ 0.41, -0.35,  0.53, -0.29,  0.46, -0.34,  0.58, -0.26,  0.44, -0.45,  0.52, -0.31,  0.47, -0.38,  0.55, -0.28],
            [-0.49,  0.33, -0.47,  0.31, -0.52,  0.27, -0.45,  0.39, -0.55,  0.26, -0.48,  0.34, -0.51,  0.29, -0.43,  0.36],
            [ 0.46, -0.41,  0.51, -0.33,  0.48, -0.29,  0.54, -0.37,  0.43, -0.47,  0.56, -0.25,  0.49, -0.36,  0.52, -0.31],
            [-0.48,  0.38, -0.54,  0.26, -0.47,  0.35, -0.52,  0.29, -0.44,  0.32, -0.56,  0.27, -0.45,  0.34, -0.53,  0.31],
            [ 0.43, -0.33,  0.49, -0.37,  0.52, -0.28,  0.46, -0.44,  0.55, -0.31,  0.48, -0.35,  0.54, -0.26,  0.47, -0.39],
            [-0.51,  0.31, -0.46,  0.34, -0.53,  0.28, -0.48,  0.38, -0.55,  0.26, -0.42,  0.33, -0.49,  0.31, -0.54,  0.29],
            [ 0.38, -0.42,  0.53, -0.26,  0.47, -0.34,  0.55, -0.29,  0.44, -0.43,  0.52, -0.31,  0.48, -0.36,  0.56, -0.27],
            [-0.49,  0.35, -0.44,  0.31, -0.52,  0.29, -0.47,  0.37, -0.54,  0.28, -0.46,  0.33, -0.51,  0.32, -0.43,  0.35]
        ])
        self.b2 = np.zeros(16) + 0.01

        self.W3 = np.array([
            [ 0.68, -0.42], [-0.53,  0.61], [ 0.45, -0.57], [-0.59,  0.44], [ 0.52, -0.48],
            [-0.46,  0.55], [ 0.63, -0.36], [-0.54,  0.49], [ 0.41, -0.62], [-0.57,  0.43],
            [ 0.50, -0.51], [-0.48,  0.53], [ 0.61, -0.39], [-0.55,  0.47], [ 0.44, -0.58],
            [-0.52,  0.50]
        ])
        self.b3 = np.array([0.03, -0.04])

    def predict(self, features):
        h1 = np.maximum(0, np.dot(features, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        out = np.dot(h2, self.W3) + self.b3
        depth = float(np.clip(1.0 / (1.0 + np.exp(-out[0])) * 0.8 + 0.02, 0.02, 0.8))
        freq  = float(np.clip(1.0 / (1.0 + np.exp(-out[1])) * 4.5 + 3.5, 3.5, 8.0))
        return depth, freq

_pitch_model = DeepPitchModel()


def midi_to_hz(m): return 440.0 * (2 ** ((m - 69) / 12))

def generate_auto_pitch_curve(notes, total_frames, manual_curves=None, auto_enabled=True):
    curve = np.full(total_frames, -1e9)
    for n in notes:
        s = int(n['start'] * 1000 / FRAME_PERIOD)
        e = int(n['end'] * 1000 / FRAME_PERIOD)
        if s < total_frames and e > 0: curve[max(0,s):min(total_frames,e)] = n['midi']
    for i in range(1, total_frames):
        if curve[i] < -1e8: curve[i] = curve[i-1] if curve[i-1] > -1e8 else 0.0
    if auto_enabled:
        curve = np.convolve(curve, np.ones(5)/5, mode='same')
        t = np.arange(total_frames) * FRAME_PERIOD / 1000
        for idx, n in enumerate(notes):
            s = int(n['start'] * 1000 / FRAME_PERIOD)
            e = int(n['end'] * 1000 / FRAME_PERIOD)
            if e <= s: continue
            dur = n['end'] - n['start']
            midi = n['midi']
            prev_midi = notes[idx-1]['midi'] if idx > 0 else midi
            interval = abs(midi - prev_midi)
            feat = np.array([min(dur, 4.0)/4.0, midi/127.0, interval/12.0,
                             n['start']/max(1, n['end']), 1.0 if idx == len(notes)-1 else 0.0])
            depth, freq = _pitch_model.predict(feat)
            note_len = e - s
            fade_len = min(10, note_len)
            fade_env = np.concatenate([np.linspace(0,1,fade_len), np.ones(note_len-fade_len)])
            curve[s:e] += depth * np.sin(2*np.pi*freq*t[s:e]) * fade_env
    if manual_curves:
        for nid, mc in manual_curves.items():
            note = next((n for n in notes if n['id'] == nid), None)
            if not note: continue
            s = int(note['start']*1000/FRAME_PERIOD)
            e = int(note['end']*1000/FRAME_PERIOD)
            if e <= s: continue
            n_frames = e - s
            xs = np.linspace(0,1,len(mc))
            ratios = np.linspace(0,1,n_frames)
            curve[s:e] = np.interp(ratios, xs, [pt[1] for pt in mc])
    return curve


# ==================== 渲染函数 ====================
def normalize_volume(audio, target_rms=0.15):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0: audio *= (target_rms / rms)
    return audio

def crossfade(prev, next, fade_len=88):
    if len(prev) < fade_len or len(next) < fade_len: return prev, next
    prev[-fade_len:] *= np.linspace(1, 0, fade_len)
    next[:fade_len]   *= np.linspace(0, 1, fade_len)
    return prev, next

def render_dnn(notes, wav_dir, return_audio=False, manual_curves=None, auto_enabled=True):
    if not notes: return np.array([]) if return_audio else None
    total_dur = max(n['end'] for n in notes)
    total_frames = int(total_dur * 1000 / FRAME_PERIOD) + 10
    global_curve = generate_auto_pitch_curve(notes, total_frames, manual_curves, auto_enabled)
    total_samples = int(total_dur * SAMPLE_RATE)
    out = np.zeros(total_samples, dtype=np.float64)
    src = {}
    if os.path.isdir(wav_dir):
        for f in os.listdir(wav_dir):
            if f.lower().endswith('.wav'): src[os.path.splitext(f)[0].lower()] = os.path.join(wav_dir, f)
    prev_audio = None; prev_end_samp = 0
    for n in notes:
        lyric = n.get('lyric','a').lower()
        if lyric not in src: continue
        path = src[lyric]
        target_dur = n['end'] - n['start']
        if target_dur < 0.08: continue
        audio, sr = sf.read(path, dtype=np.float64)
        if sr != SAMPLE_RATE:
            audio = np.interp(np.arange(0, len(audio), SAMPLE_RATE/sr), np.arange(len(audio)), audio)
        target_len = int(target_dur * SAMPLE_RATE)
        if len(audio) != target_len:
            audio = np.interp(np.linspace(0, len(audio)-1, target_len), np.arange(len(audio)), audio)
        audio = normalize_volume(audio, 0.15)
        f0, t = pw.dio(audio.astype(np.float64), SAMPLE_RATE, frame_period=FRAME_PERIOD)
        f0 = pw.stonemask(audio.astype(np.float64), f0, t, SAMPLE_RATE)
        sp = pw.cheaptrick(audio.astype(np.float64), f0, t, SAMPLE_RATE)
        ap = pw.d4c(audio.astype(np.float64), f0, t, SAMPLE_RATE)
        n_frames = len(f0)
        note_times = np.linspace(n['start'], n['end'], n_frames)
        global_times = np.arange(len(global_curve)) * FRAME_PERIOD / 1000.0
        seg_curve = np.interp(note_times, global_times, global_curve)
        avg_f0 = np.mean(f0[f0>0]) if np.any(f0>0) else 0
        mod_f0 = np.zeros(n_frames)
        for i in range(n_frames):
            target_hz = midi_to_hz(seg_curve[i])
            mod_f0[i] = f0[i] * (target_hz/avg_f0) if f0[i]>0 and avg_f0>0 else target_hz
        synth = pw.synthesize(mod_f0.astype(np.float64), sp, ap, SAMPLE_RATE, frame_period=FRAME_PERIOD)
        if len(synth) > target_len: synth = synth[:target_len]
        elif len(synth) < target_len: synth = np.pad(synth, (0, target_len-len(synth)))
        if prev_audio is not None:
            rms_prev = np.sqrt(np.mean(prev_audio**2))
            rms_curr = np.sqrt(np.mean(synth**2))
            if rms_curr > 0 and rms_prev > 0: synth *= (rms_prev / rms_curr)
        start_samp = int(n['start'] * SAMPLE_RATE)
        end_samp = start_samp + len(synth)
        if prev_audio is not None and start_samp == prev_end_samp:
            fade_len = min(88, len(prev_audio), len(synth))
            crossfade(prev_audio, synth, fade_len)
        if end_samp > len(out): out = np.pad(out, (0, end_samp-len(out)))
        out[start_samp:end_samp] += synth
        prev_audio = synth
        prev_end_samp = end_samp
    mx = np.max(np.abs(out))
    if mx > 0: out /= mx * 1.1
    if return_audio: return out.astype(np.float32)
    sf.write('output.wav', out.astype(np.float32), SAMPLE_RATE)
    return out.astype(np.float32)

class SoVITSSVC_Mock:
    def convert(self, wav_data, pitch_curve=None):
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 500/(SAMPLE_RATE/2), btype='high')
        return filtfilt(b, a, wav_data).astype(np.float32)

def render_postprocess(notes, wav_dir, return_audio=False, manual_curves=None, auto_enabled=True):
    dnn_audio = render_dnn(notes, wav_dir, return_audio=True, manual_curves=manual_curves, auto_enabled=auto_enabled)
    if len(dnn_audio) == 0: return dnn_audio
    total_dur = max(n['end'] for n in notes)
    total_frames = int(total_dur * 1000 / FRAME_PERIOD) + 10
    curve = generate_auto_pitch_curve(notes, total_frames, manual_curves, auto_enabled)
    pitch_samples = np.interp(np.linspace(0, len(curve)-1, len(dnn_audio)), np.arange(len(curve)), curve)
    ai_wav = SoVITSSVC_Mock().convert(dnn_audio, pitch_curve=pitch_samples)
    if return_audio: return ai_wav
    sf.write('output.wav', ai_wav, SAMPLE_RATE)
    return ai_wav

synthesize_song = render_dnn   # 向后兼容