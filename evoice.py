# EVOiCE.py
"""
EVOiCE 完整版 —— 独立合成引擎、保留头像、无缝音符、自然颤音
依赖：numpy, pyworld, soundfile, scipy, sounddevice (可选), Pillow (可选)
"""

import sys, os, math, threading, webbrowser, traceback, json, time, tempfile, subprocess, platform

def emergency_alert(t, m):
    try: import ctypes; ctypes.windll.user32.MessageBoxW(0, m, t, 0x30)
    except: pass

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk, Menu
except Exception as e:
    emergency_alert("错误", f"tkinter 导入失败：{e}")
    sys.exit(1)

try:
    import numpy as np
    import soundfile as sf
except Exception as e:
    emergency_alert("错误", f"核心音频库缺失：{e}")
    sys.exit(1)

# 导入独立合成模块
from synthesizer import (
    synthesize_song,
    generate_auto_pitch_curve,
    midi_to_hz,
    SAMPLE_RATE,
    render_dnn,
    render_postprocess,
    extract_pitch_with_crepe   # 新增，用于 AI 提取音高线
)

HAS_SD = False
try:
    import sounddevice as sd
    HAS_SD = 'sounddevice'
except: pass

HAS_CREPE = False
try:
    import crepe
    HAS_CREPE = True
except:
    HAS_CREPE = False

# 尝试导入 Pillow 以显示头像
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except:
    HAS_PIL = False

# ---------- 全局参数 ----------
FRAME_PERIOD = 5.0
START_MIDI = 36
OCTAVES = 5
WHITE_KEYS = [0,2,4,5,7,9,11]
DEFAULT_BPM = 120
DEFAULT_SNAP = '1/4'
WAVEFORM_HEIGHT = 30

STYLE = {
    'bg': '#F0F0F5',
    'menu_bg': '#E0E0E8',
    'canvas_bg': '#FFFFFF',
    'note_fill': '#A8D8EA',
    'note_selected': '#7EC8E3',
    'note_border': '#3E8EAF',
    'note_text': '#1A3A4A',
    'grid_strong': '#D0D0D0',
    'grid_weak': '#EEEEEE',
    'key_white': '#FFFFFF',
    'key_black': '#444444',
    'pitch_curve': '#A0A0A0',
    'pitch_handle': '#F1C40F',
    'waveform': '#B0B0B0',
    'playhead': '#E74C3C',
    'status_bg': '#2C3E50',
    'status_fg': '#ECF0F1',
    'prop_bg': '#F0F0F0',
}

# ================== 播放工具 ==================
def play_audio_blocking(audio_array, sr=SAMPLE_RATE):
    if len(audio_array) == 0: return
    threading.Thread(target=_play_blocking, args=(audio_array, sr), daemon=True).start()

def _play_blocking(audio_array, sr):
    if HAS_SD == 'sounddevice':
        sd.play(audio_array, sr)
        sd.wait()
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, audio_array, sr)
        tmp.close()
        try:
            if platform.system() == 'Windows':
                os.startfile(tmp.name)
            elif platform.system() == 'Darwin':
                subprocess.run(['afplay', tmp.name])
            else:
                subprocess.run(['aplay', tmp.name], stderr=subprocess.DEVNULL)
        except:
            webbrowser.open(tmp.name)
        threading.Thread(target=lambda: (time.sleep(len(audio_array)/sr+1), os.unlink(tmp.name))).start()

def make_piano_tone(midi, dur=0.15):
    hz = midi_to_hz(midi)
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    attack_len = int(SAMPLE_RATE * 0.02)
    release_len = int(SAMPLE_RATE * 0.05)
    if len(t) < attack_len + release_len:
        envelope = np.linspace(1, 0, len(t))
    else:
        envelope = np.ones(len(t))
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-release_len:] = np.linspace(1, 0, release_len)
    return 0.3 * np.sin(2 * np.pi * hz * t) * envelope

# ================== 钢琴卷帘 ==================
class PianoRoll(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=STYLE['bg'], **kw)
        self.notes = []
        self.note_items = {}
        self.next_id = 0
        self.bpm = DEFAULT_BPM
        self.snap = DEFAULT_SNAP
        self.zoom_x = 50
        self.grid_y = 26
        self.offset_x = 60
        self.offset_y = 20
        self.total_seconds = 15
        self.playhead_time = 0.0
        self.auto_pitch_enabled = True
        self.manual_curves = {}
        self.selected_note = None
        self.on_select_callback = None

        self.drag_note = None
        self.drag_mode = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_pos = None
        self.click_tolerance = 5

        self.creating_note = False
        self.create_start = None
        self.create_midi = None
        self.create_rect = None
        self._last_beep = 0

        self.icon_label = None

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg=STYLE['canvas_bg'], highlightthickness=0)
        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.h_scroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        self.v_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.icon_label = tk.Label(self.canvas_frame, bg=STYLE['canvas_bg'])
        self.icon_label.place(x=5, y=5)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<space>", self.toggle_playback)
        self.canvas.focus_set()
        self.draw_grid()

    # --- 公共方法 ---
    def set_select_callback(self, cb): self.on_select_callback = cb
    def set_auto_pitch(self, enabled):
        self.auto_pitch_enabled = enabled
        self.draw_grid()

    # --- 坐标 ---
    def time_to_x(self, t): return self.offset_x + t * self.zoom_x
    def x_to_time(self, x): return max(0.0, (x - self.offset_x) / self.zoom_x)
    def midi_to_y(self, m): return self.offset_y + (START_MIDI + OCTAVES*12 - m) * self.grid_y
    def y_to_midi(self, y):
        rel = self.offset_y + OCTAVES*12*self.grid_y - y
        midi = int(rel / self.grid_y) + START_MIDI
        return max(START_MIDI, min(START_MIDI + OCTAVES*12 - 1, midi))
    def beat_duration(self, beat='1/4'): return (60.0 / self.bpm) * (1.0 / int(beat.split('/')[1]))
    def snap_time(self, t, beat=None):
        if beat is None: beat = self.snap
        dur = self.beat_duration(beat)
        return round(t / dur) * dur

    def play_piano_once(self, midi):
        if time.time() - self._last_beep > 0.12:
            self._last_beep = time.time()
            threading.Thread(target=lambda: play_audio_blocking(make_piano_tone(midi)), daemon=True).start()

    # --- 播放控制 ---
    def toggle_playback(self, event=None):
        self.start_play()

    def start_play(self):
        if not hasattr(self.master, 'master') or not self.master.master.source_dir:
            messagebox.showwarning("提示", "请先选择歌手")
            return
        audio = synthesize_song(self.notes, self.master.master.source_dir, return_audio=True,
                                manual_curves=self.manual_curves, auto_enabled=self.auto_pitch_enabled)
        if len(audio) == 0:
            messagebox.showinfo("提示", "合成结果为空")
            return
        play_audio_blocking(audio)

    # --- 加载头像 ---
    def load_singer_icon(self, singer_dir):
        icon_path = os.path.join(singer_dir, "icon.bmp")
        if os.path.exists(icon_path) and HAS_PIL:
            try:
                img = Image.open(icon_path).resize((48, 48))
                self.icon_img = ImageTk.PhotoImage(img)
                self.icon_label.config(image=self.icon_img)
            except:
                self.icon_label.config(image='')
        else:
            self.icon_label.config(image='')

    # --- 绘制 ---
    def draw_grid(self):
        self.canvas.delete("grid")
        h = self.midi_to_y(START_MIDI) + 50
        w = self.offset_x + self.total_seconds * self.zoom_x + 200
        self.canvas.config(scrollregion=(0, 0, w + 100, h + 100))

        q_len = self.beat_duration('1/4')
        e_len = self.beat_duration('1/8')
        for t in np.arange(0, self.total_seconds + e_len, e_len):
            x = self.time_to_x(t)
            if x > w + 200: break
            fill = STYLE['grid_strong'] if abs(t % q_len) < 0.001 else STYLE['grid_weak']
            self.canvas.create_line(x, 0, x, h, fill=fill, tags='grid')
        for midi in range(START_MIDI, START_MIDI + OCTAVES*12 + 1):
            y = self.midi_to_y(midi)
            color = STYLE['grid_strong'] if midi % 12 in WHITE_KEYS else STYLE['grid_weak']
            self.canvas.create_line(0, y, w, y, fill=color, tags='grid')
        for midi in range(START_MIDI, START_MIDI + OCTAVES*12):
            y = self.midi_to_y(midi)
            fill = STYLE['key_white'] if midi % 12 in WHITE_KEYS else STYLE['key_black']
            self.canvas.create_rectangle(0, y - self.grid_y//2, self.offset_x, y + self.grid_y//2,
                                         fill=fill, outline='#CCCCCC', tags='grid')
        self.redraw_all()

    def redraw_all(self):
        self.draw_playhead()
        self.draw_waveforms()
        self.draw_manual_handles()
        self.update_note_display()
        self.redraw_pitch_preview()

    def redraw_pitch_preview(self):
        self.canvas.delete("pitch_curve")
        if not self.notes: return
        total_frames = int(self.total_seconds * 1000 / FRAME_PERIOD) + 10
        curve = generate_auto_pitch_curve(self.notes, total_frames, self.manual_curves, self.auto_pitch_enabled)
        pts = []
        for i in range(total_frames):
            t = i * FRAME_PERIOD / 1000
            x = self.time_to_x(t)
            midi_val = max(START_MIDI, min(START_MIDI + OCTAVES*12 - 1, curve[i]))
            y = self.midi_to_y(midi_val)
            pts.extend([x, y])
        if len(pts) >= 4: self.canvas.create_line(pts, fill=STYLE['pitch_curve'], width=2, tags='pitch_curve')

    def draw_playhead(self):
        self.canvas.delete("playhead")
        x = self.time_to_x(self.playhead_time)
        self.canvas.create_line(x, 0, x, 9999, fill=STYLE['playhead'], width=2, tags="playhead")

    def draw_waveforms(self):
        self.canvas.delete("waveform")
        if not hasattr(self.master, 'master') or not hasattr(self.master.master, 'source_dir'): return
        sdir = self.master.master.source_dir
        for n in self.notes:
            lyric = n.get('lyric', 'a').lower()
            wav_path = os.path.join(sdir, f"{lyric}.wav")
            if not os.path.exists(wav_path): continue
            try:
                audio, sr = sf.read(wav_path, dtype=np.float32)
                if sr != SAMPLE_RATE:
                    audio = np.interp(np.linspace(0, len(audio)-1, int(len(audio)*SAMPLE_RATE/sr)), np.arange(len(audio)), audio)
                target_len = int((n['end'] - n['start']) * SAMPLE_RATE)
                if len(audio) != target_len:
                    audio = np.interp(np.linspace(0, len(audio)-1, target_len), np.arange(len(audio)), audio)
                y_center = self.midi_to_y(n['midi']) + self.grid_y//2 + WAVEFORM_HEIGHT//2 + 2
                x1 = self.time_to_x(n['start'])
                x2 = self.time_to_x(n['end'])
                max_pts = int((x2 - x1) // 2)
                if max_pts < 2: continue
                step = max(1, len(audio) // max_pts)
                pts = []
                for i in range(0, len(audio), step):
                    t_rel = i / len(audio)
                    x = x1 + (x2 - x1) * t_rel
                    amp = audio[i] * (WAVEFORM_HEIGHT//2 - 1)
                    pts.extend([x, y_center - amp])
                if len(pts) >= 4:
                    self.canvas.create_line(pts, fill=STYLE['waveform'], tags='waveform', smooth=True)
            except: pass

    def draw_manual_handles(self):
        self.canvas.delete("manual_handle")
        for nid, pts in self.manual_curves.items():
            note = next((n for n in self.notes if n['id'] == nid), None)
            if not note: continue
            x1 = self.time_to_x(note['start'])
            x2 = self.time_to_x(note['end'])
            for rx, midi in pts:
                x = x1 + (x2 - x1) * rx
                y = self.midi_to_y(midi)
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=STYLE['pitch_handle'], outline='black', tags='manual_handle')

    def update_note_display(self):
        for nid in list(self.note_items.keys()):
            self.canvas.delete(self.note_items[nid][0])
            self.canvas.delete(self.note_items[nid][1])
        self.note_items.clear()
        for n in self.notes:
            fill = STYLE['note_selected'] if (self.selected_note and self.selected_note['id'] == n['id']) else STYLE['note_fill']
            outline = STYLE['note_border']
            x1 = self.time_to_x(n['start'])
            x2 = self.time_to_x(n['end'])
            y = self.midi_to_y(n['midi'])
            rect = self.canvas.create_rectangle(x1, y - self.grid_y//2, x2, y + self.grid_y//2,
                                                fill=fill, outline=outline, tags='note')
            txt = self.canvas.create_text((x1+x2)/2, y, text=n['lyric'], fill=STYLE['note_text'], font=('Arial', 10, 'bold'))
            self.note_items[n['id']] = (rect, txt)

    def select_note(self, note):
        self.selected_note = note
        self.update_note_display()
        if self.on_select_callback: self.on_select_callback(note)

    def delete_selected(self, event=None):
        if self.selected_note:
            self.notes = [n for n in self.notes if n['id'] != self.selected_note['id']]
            if self.selected_note['id'] in self.manual_curves: del self.manual_curves[self.selected_note['id']]
            self.selected_note = None
            self.update_note_display()
            self.draw_grid()

    # --- 交互逻辑 ---
    def on_motion(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        for nid, pts in self.manual_curves.items():
            note = next((n for n in self.notes if n['id'] == nid), None)
            if not note: continue
            x1 = self.time_to_x(note['start'])
            x2 = self.time_to_x(note['end'])
            for (rx, midi) in pts:
                cx = x1 + (x2 - x1) * rx
                cy = self.midi_to_y(midi)
                if (cx - canvas_x)**2 + (cy - canvas_y)**2 < 36:
                    self.canvas.config(cursor='hand2')
                    return
        if abs(canvas_x - self.time_to_x(self.playhead_time)) < 8:
            self.canvas.config(cursor='sb_h_double_arrow')
            return
        self.canvas.config(cursor='arrow')

    def on_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if canvas_x < self.offset_x:
            midi = self.y_to_midi(canvas_y)
            threading.Thread(target=lambda: play_audio_blocking(make_piano_tone(midi, 0.3)), daemon=True).start()
            return
        if abs(canvas_x - self.time_to_x(self.playhead_time)) < 8:
            self.drag_mode = 'playhead'
            return
        for nid, pts in list(self.manual_curves.items()):
            note = next((n for n in self.notes if n['id'] == nid), None)
            if not note: continue
            x1 = self.time_to_x(note['start'])
            x2 = self.time_to_x(note['end'])
            for i, (rx, midi) in enumerate(pts):
                cx = x1 + (x2 - x1) * rx
                cy = self.midi_to_y(midi)
                if (cx - canvas_x)**2 + (cy - canvas_y)**2 < 36:
                    self.drag_mode = 'manual_point'
                    self.drag_point_info = (nid, i)
                    self.drag_start_x = canvas_x
                    self.drag_start_y = canvas_y
                    return
        for n in reversed(self.notes):
            x1 = self.time_to_x(n['start'])
            x2 = self.time_to_x(n['end'])
            y1 = self.midi_to_y(n['midi']) - self.grid_y//2
            y2 = self.midi_to_y(n['midi']) + self.grid_y//2 + WAVEFORM_HEIGHT
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                self.drag_note = n
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                self.drag_start_pos = (n['start'], n['end'], n['midi'])
                self.drag_mode = 'resize' if abs(canvas_x - x2) < 16 else 'move'
                return
        self.selected_note = None
        self.update_note_display()
        if canvas_x > self.offset_x:
            start = self.snap_time(self.x_to_time(canvas_x))
            if self.notes:
                prev_end = max(n['end'] for n in self.notes)
                if start <= prev_end + self.beat_duration('1/4'):
                    start = prev_end
            self.create_start = start
            self.create_midi = self.y_to_midi(canvas_y)
            self.creating_note = True
            x1 = self.time_to_x(start)
            y = self.midi_to_y(self.create_midi)
            self.create_rect = self.canvas.create_rectangle(x1, y-self.grid_y//2, x1, y+self.grid_y//2,
                                                            fill='', outline='black', dash=(4,2), tags='create')
            self.play_piano_once(self.create_midi)
            self.drag_mode = None

    def on_drag(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if self.drag_mode == 'playhead':
            self.playhead_time = max(0, min(self.total_seconds, self.x_to_time(canvas_x)))
            self.draw_playhead()
            return
        if self.drag_mode == 'manual_point':
            nid, idx = self.drag_point_info
            pts = self.manual_curves[nid]
            note = next((n for n in self.notes if n['id'] == nid), None)
            if not note: return
            x1 = self.time_to_x(note['start'])
            x2 = self.time_to_x(note['end'])
            rx = max(0, min(1, (canvas_x - x1) / (x2 - x1)))
            pts[idx] = (rx, self.y_to_midi(canvas_y))
            self.draw_grid()
            return
        if self.creating_note:
            current_time = self.snap_time(max(self.create_start, self.x_to_time(canvas_x)))
            current_midi = self.y_to_midi(canvas_y)
            x1 = self.time_to_x(self.create_start)
            x2 = self.time_to_x(current_time)
            y = self.midi_to_y(current_midi)
            if self.create_rect:
                self.canvas.coords(self.create_rect, x1, y-self.grid_y//2, x2, y+self.grid_y//2)
            self.play_piano_once(current_midi)
            self.create_midi = current_midi
            return
        if not self.drag_note: return
        dx = canvas_x - self.drag_start_x
        dy = canvas_y - self.drag_start_y
        if abs(dx) < self.click_tolerance and abs(dy) < self.click_tolerance: return
        dx_sec = dx / self.zoom_x
        dy_midi = -dy / self.grid_y
        if self.drag_mode == 'move':
            new_start = max(0, self.drag_note['start'] + dx_sec)
            dur = self.drag_note['end'] - self.drag_note['start']
            new_start = self.snap_time(new_start)
            new_midi = int(self.drag_note['midi'] + dy_midi)
            new_midi = max(START_MIDI, min(START_MIDI + OCTAVES*12 - 1, new_midi))
            self.drag_note['start'] = new_start
            self.drag_note['end'] = new_start + dur
            self.drag_note['midi'] = new_midi
            self.drag_start_x = canvas_x
            self.drag_start_y = canvas_y
            self.update_note_display()
        elif self.drag_mode == 'resize':
            new_end = self.drag_note['start'] + max(0.05, self.drag_note['end'] + dx_sec - self.drag_note['start'])
            new_end = self.snap_time(new_end)
            if new_end > self.drag_note['start'] + 0.05:
                self.drag_note['end'] = new_end
                self.drag_start_x = canvas_x
                self.update_note_display()

    def on_release(self, event):
        if self.drag_note and self.drag_start_pos:
            n = self.drag_note
            if (n['start'], n['end'], n['midi']) == self.drag_start_pos:
                self.select_note(n)
            elif self.selected_note and self.selected_note['id'] == n['id']:
                if self.on_select_callback: self.on_select_callback(n)
        self.drag_note = None
        self.drag_mode = None
        self.drag_start_pos = None
        self.drag_point_info = None

        if self.creating_note:
            self.canvas.delete(self.create_rect)
            self.create_rect = None
            end_time = self.snap_time(max(self.create_start, self.x_to_time(self.canvas.canvasx(event.x))))
            if end_time > self.create_start + 0.05:
                self.add_note(self.create_start, self.create_midi, end_time - self.create_start, 'a', snap=False)
            self.creating_note = False
            self.draw_grid()
            return
        self.canvas.config(cursor='arrow')

    def on_double_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        for n in reversed(self.notes):
            x1 = self.time_to_x(n['start'])
            x2 = self.time_to_x(n['end'])
            y1 = self.midi_to_y(n['midi']) - self.grid_y//2
            y2 = self.midi_to_y(n['midi']) + self.grid_y//2 + WAVEFORM_HEIGHT
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                self.start_manual_edit(n)
                return
        self.editing_note = None
        self.draw_grid()

    def start_manual_edit(self, note):
        self.editing_note = note
        nid = note['id']
        if nid not in self.manual_curves:
            self.manual_curves[nid] = [(0.0, note['midi']), (0.33, note['midi']), (0.66, note['midi']), (1.0, note['midi'])]
        self.draw_grid()

    def delete_manual_curve(self, event=None):
        if self.editing_note and self.editing_note['id'] in self.manual_curves:
            del self.manual_curves[self.editing_note['id']]
            self.editing_note = None
            self.draw_grid()

    def add_note(self, start, midi, duration, lyric, snap=True):
        if snap:
            start = self.snap_time(start)
            end = self.snap_time(start + duration)
        else:
            end = start + duration
        if self.notes:
            prev_end = max(n['end'] for n in self.notes)
            if abs(start - prev_end) < 0.01:
                start = prev_end
        note = {'id': self.next_id, 'start': start, 'end': end, 'midi': midi, 'lyric': lyric}
        self.next_id += 1
        self.notes.append(note)
        if note['end'] + 2 > self.total_seconds: self.total_seconds = round(note['end'] + 2)
        self.draw_grid()
        self.select_note(note)

    def modify_note(self, note_id, **kwargs):
        note = next((n for n in self.notes if n['id'] == note_id), None)
        if note:
            for k, v in kwargs.items():
                if k in note: note[k] = v
            self.update_note_display()
            self.draw_grid()
            if self.selected_note and self.selected_note['id'] == note_id:
                if self.on_select_callback: self.on_select_callback(note)

    def set_zoom(self, factor):
        new = self.zoom_x * factor
        if 15 <= new <= 200:
            self.zoom_x = new
            self.draw_grid()

    def set_bpm(self, bpm):
        try: self.bpm = float(bpm)
        except: self.bpm = DEFAULT_BPM
        self.draw_grid()

    def set_snap(self, snap):
        self.snap = snap
        self.draw_grid()

    def get_project_data(self):
        return {'bpm': self.bpm, 'notes': [{k: v for k, v in n.items() if k != 'id'} for n in self.notes]}

    def load_project_data(self, data):
        self.notes.clear(); self.note_items.clear(); self.manual_curves.clear()
        self.next_id = 0; self.total_seconds = 15
        if 'bpm' in data: self.bpm = data['bpm']
        for n in data.get('notes', []):
            note = {'id': self.next_id, 'start': n['start'], 'end': n['end'], 'midi': n['midi'], 'lyric': n.get('lyric', 'a')}
            self.notes.append(note); self.next_id += 1
            if note['end'] + 2 > self.total_seconds: self.total_seconds = round(note['end'] + 2)
        self.draw_grid()

# ================== 编辑器页面 ==================
class EditorPage(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=STYLE['bg'])
        self.app = app
        self.project_path = None

        main = tk.Frame(self, bg=STYLE['bg'])
        main.pack(fill=tk.BOTH, expand=True)

        self.piano = PianoRoll(main)
        self.piano.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(self, bg=STYLE['menu_bg'], padx=5, pady=5)
        toolbar.pack(fill=tk.X)

        grp1 = tk.LabelFrame(toolbar, text="节拍", bg=STYLE['menu_bg'], padx=5, pady=5)
        grp1.pack(side=tk.LEFT, padx=5)
        tk.Label(grp1, text="BPM:", bg=STYLE['menu_bg']).pack(side=tk.LEFT)
        self.bpm_var = tk.StringVar(value=str(DEFAULT_BPM))
        ttk.Entry(grp1, textvariable=self.bpm_var, width=4).pack(side=tk.LEFT)
        self.bpm_var.trace_add('write', lambda *a: self.piano.set_bpm(self.bpm_var.get()))
        tk.Label(grp1, text="吸附:", bg=STYLE['menu_bg']).pack(side=tk.LEFT, padx=(5,0))
        self.snap_var = tk.StringVar(value=DEFAULT_SNAP)
        ttk.Combobox(grp1, textvariable=self.snap_var, values=['1/4','1/8','1/16'], width=4, state='readonly').pack(side=tk.LEFT)
        self.snap_var.trace_add('write', lambda *a: self.piano.set_snap(self.snap_var.get()))

        grp2 = tk.LabelFrame(toolbar, text="视图", bg=STYLE['menu_bg'], padx=5, pady=5)
        grp2.pack(side=tk.LEFT, padx=5)
        ttk.Button(grp2, text="🔍+", command=lambda: self.piano.set_zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(grp2, text="🔍-", command=lambda: self.piano.set_zoom(0.8)).pack(side=tk.LEFT)

        grp3 = tk.LabelFrame(toolbar, text="控制", bg=STYLE['menu_bg'], padx=5, pady=5)
        grp3.pack(side=tk.LEFT, padx=5)
        self.auto_pitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp3, text="自动音高", variable=self.auto_pitch_var, command=self.toggle_auto_pitch).pack(side=tk.LEFT)
        ttk.Button(grp3, text="▶ 播放", command=self.start_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(grp3, text="📦 导出WAV", command=self.export_wav).pack(side=tk.LEFT, padx=2)

        # 渲染方案下拉
        tk.Label(toolbar, text="渲染方案:", bg=STYLE['menu_bg']).pack(side=tk.LEFT, padx=(20,0))
        self.render_mode = tk.StringVar(value="自研 DNN")
        render_combo = ttk.Combobox(toolbar, textvariable=self.render_mode,
                                    values=["自研 DNN", "So‑VITS‑SVC 后处理"],
                                    width=18, state='readonly')
        render_combo.pack(side=tk.LEFT)

        # 新增“提取音高”按钮（用于 crepe）
        ttk.Button(toolbar, text="🎤 提取音高", command=self.extract_pitch_from_audio).pack(side=tk.LEFT, padx=5)

        ttk.Button(toolbar, text="📝 填充歌词", command=self.batch_lyrics).pack(side=tk.LEFT, padx=10)

        tk.Label(toolbar, text="歌手:", bg=STYLE['menu_bg']).pack(side=tk.LEFT, padx=(20,0))
        self.singer_var = tk.StringVar(value="默认歌手")
        self.singer_combo = ttk.Combobox(toolbar, textvariable=self.singer_var, width=12, state='readonly')
        self.singer_combo.pack(side=tk.LEFT)
        self.singer_combo.bind("<<ComboboxSelected>>", lambda e: self.select_singer())

        ttk.Button(toolbar, text="💾 保存", command=self.save_project).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="🏠 返回", command=self.back_to_home).pack(side=tk.RIGHT, padx=2)

        track_panel = tk.Frame(main, bg=STYLE['menu_bg'], width=150)
        track_panel.pack(side=tk.LEFT, fill=tk.Y)
        track_panel.pack_propagate(False)
        tk.Label(track_panel, text="Track 1", font=('Arial', 12, 'bold'), bg=STYLE['menu_bg']).pack(pady=10)
        tk.Label(track_panel, text="选择歌手:", bg=STYLE['menu_bg']).pack(pady=(20,0))
        self.singer_listbox = tk.Listbox(track_panel, height=8)
        self.singer_listbox.pack(fill=tk.X, padx=10, pady=5)
        self.load_singers()

        prop_frame = tk.LabelFrame(main, text="音符属性", bg=STYLE['prop_bg'], padx=10, pady=10, width=220)
        prop_frame.pack(side=tk.RIGHT, fill=tk.Y)
        prop_frame.pack_propagate(False)
        labels = ["歌词", "音高 (MIDI)", "开始 (秒)", "结束 (秒)"]
        self.prop_entries = []
        for i, txt in enumerate(labels):
            ttk.Label(prop_frame, text=txt, background=STYLE['prop_bg']).pack(anchor=tk.W, pady=(5,0))
            entry = ttk.Entry(prop_frame, width=15)
            entry.pack(fill=tk.X, pady=2)
            entry.bind("<Return>", self.apply_properties)
            self.prop_entries.append(entry)
        ttk.Button(prop_frame, text="应用修改", command=self.apply_properties).pack(pady=10)

        self.piano.set_select_callback(self.on_note_selected)
        self.source_dir = os.path.join(self.app.singers_dir, "默认歌手")
        self.piano.load_singer_icon(self.source_dir)

    def load_singers(self):
        singers_dir = self.app.singers_dir
        if os.path.isdir(singers_dir):
            for d in os.listdir(singers_dir):
                if os.path.isdir(os.path.join(singers_dir, d)):
                    self.singer_listbox.insert(tk.END, d)
            singers = list(self.singer_listbox.get(0, tk.END))
            if singers:
                self.singer_var.set(singers[0])
                self.singer_combo['values'] = singers

    def select_singer(self):
        name = self.singer_var.get()
        if name:
            self.source_dir = os.path.join(self.app.singers_dir, name)
            self.piano.load_singer_icon(self.source_dir)
            self.piano.draw_grid()

    def toggle_auto_pitch(self): self.piano.set_auto_pitch(self.auto_pitch_var.get())

    def on_note_selected(self, note):
        for entry in self.prop_entries: entry.delete(0, tk.END)
        if note:
            self.prop_entries[0].insert(0, note['lyric'])
            self.prop_entries[1].insert(0, str(note['midi']))
            self.prop_entries[2].insert(0, f"{note['start']:.2f}")
            self.prop_entries[3].insert(0, f"{note['end']:.2f}")

    def apply_properties(self, event=None):
        if not self.piano.selected_note:
            messagebox.showwarning("提示", "请先选中一个音符")
            return
        note = self.piano.selected_note
        try:
            lyric = self.prop_entries[0].get().strip()
            midi = int(self.prop_entries[1].get())
            start = float(self.prop_entries[2].get())
            end = float(self.prop_entries[3].get())
        except ValueError:
            messagebox.showerror("输入错误", "数值格式有误")
            return
        if start >= end:
            messagebox.showerror("错误", "开始时间必须小于结束时间")
            return
        self.piano.modify_note(note['id'], lyric=lyric, midi=midi, start=start, end=end)

    def batch_lyrics(self):
        lyrics_str = simpledialog.askstring("批量填词", "输入歌词（空格分隔）:")
        if lyrics_str:
            words = lyrics_str.strip().split()
            sorted_notes = sorted(self.piano.notes, key=lambda n: n['start'])
            for i, note in enumerate(sorted_notes):
                if i < len(words): note['lyric'] = words[i]
            self.piano.update_note_display()
            self.piano.draw_grid()

    def start_play(self):
        if not self.source_dir:
            messagebox.showwarning("提示", "请先选择歌手")
            return
        notes = self.piano.notes
        mode = self.render_mode.get()
        if mode == "自研 DNN":
            audio = render_dnn(notes, self.source_dir, return_audio=True,
                               manual_curves=self.piano.manual_curves,
                               auto_enabled=self.piano.auto_pitch_enabled)
        elif mode == "So‑VITS‑SVC 后处理":
            audio = render_postprocess(notes, self.source_dir, return_audio=True,
                                       manual_curves=self.piano.manual_curves,
                                       auto_enabled=self.piano.auto_pitch_enabled)
        else:
            audio = np.array([])
        if len(audio) == 0:
            messagebox.showinfo("提示", "合成失败")
            return
        play_audio_blocking(audio)

    def export_wav(self):
        if not self.source_dir:
            messagebox.showwarning("提示", "请先选择歌手")
            return
        notes = self.piano.notes
        mode = self.render_mode.get()
        def run_export():
            if mode == "自研 DNN":
                render_dnn(notes, self.source_dir, return_audio=False,
                           manual_curves=self.piano.manual_curves,
                           auto_enabled=self.piano.auto_pitch_enabled)
            elif mode == "So‑VITS‑SVC 后处理":
                render_postprocess(notes, self.source_dir, return_audio=False,
                                   manual_curves=self.piano.manual_curves,
                                   auto_enabled=self.piano.auto_pitch_enabled)
        threading.Thread(target=run_export, daemon=True).start()

    def extract_pitch_from_audio(self):
        """用 crepe 分析干声，将频率转为 MIDI 并存入手动音高曲线（示例）"""
        file_path = filedialog.askopenfilename(filetypes=[("音频文件", "*.wav")])
        if not file_path: return
        try:
            time_arr, frequency = extract_pitch_with_crepe(file_path, sr=SAMPLE_RATE)
            # 将频率转为 MIDI（忽略无声段）
            midi_arr = np.where(frequency > 0, 69 + 12 * np.log2(frequency / 440.0), -1)
            # 这里仅示例：把提取的 MIDI 曲线赋值给当前所有音符（简单粗暴）
            # 更好的做法是让用户选择参考音符，然后将曲线应用上去。
            if self.piano.notes:
                note = self.piano.notes[0]
                nid = note['id']
                # 生成控制点
                pts = []
                for i in range(min(len(time_arr), 100)):  # 取最多100个点
                    t = time_arr[i]
                    rel_x = (t - time_arr[0]) / (time_arr[-1] - time_arr[0]) if time_arr[-1] > time_arr[0] else 0
                    midi = midi_arr[i] if midi_arr[i] > 0 else note['midi']
                    pts.append((rel_x, midi))
                self.piano.manual_curves[nid] = pts
                self.piano.draw_grid()
                messagebox.showinfo("提取完成", "已将音高曲线应用于第一个音符（示例）")
            else:
                messagebox.showwarning("提示", "请先在钢琴卷帘中添加音符")
        except Exception as e:
            messagebox.showerror("错误", f"提取失败：{e}")

    def save_project(self):
        if not self.project_path:
            self.project_path = filedialog.asksaveasfilename(defaultextension=".evgc", filetypes=[("EVOiCE工程", "*.evgc")])
        if self.project_path:
            data = self.piano.get_project_data()
            with open(self.project_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
            self.app.add_recent(self.project_path)

    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("EVOiCE工程", "*.evgc")])
        if path:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            self.piano.load_project_data(data)
            self.project_path = path
            self.app.add_recent(path)

    def back_to_home(self): self.app.show_home()

# ================== 首页 ==================
class HomePage(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=STYLE['bg'])
        self.app = app
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        card = tk.Frame(self, bg='white', padx=40, pady=40, relief=tk.RIDGE, bd=2)
        card.grid(row=0, column=0, sticky=tk.NSEW)
        tk.Label(card, text="🎤 EVOiCE", font=('Helvetica', 24, 'bold'), bg='white').pack(pady=20)
        tk.Label(card, text="自制歌声编辑器", font=('Helvetica', 12), bg='white').pack(pady=5)
        ttk.Button(card, text="📄 新建工程", command=self.new_project).pack(pady=10, ipadx=20)
        ttk.Button(card, text="📂 打开工程 (.evgc)", command=self.open_project).pack(pady=10, ipadx=20)
        tk.Label(card, text="—— 最近工程 ——", bg='white', font=('Helvetica', 10)).pack(pady=(30,5))
        self.recent_list = tk.Listbox(card, width=50, height=4)
        self.recent_list.pack(pady=5)
        self.recent_list.bind("<Double-Button-1>", self.open_recent)
        self.load_recents()

    def new_project(self): self.app.show_editor()
    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("EVOiCE工程", "*.evgc")])
        if path: self.app.show_editor(path)
    def open_recent(self, event):
        sel = self.recent_list.curselection()
        if sel:
            path = self.recent_list.get(sel[0])
            if os.path.exists(path): self.app.show_editor(path)
    def load_recents(self):
        try:
            with open('recent_projects.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line and os.path.exists(line): self.recent_list.insert(tk.END, line)
        except: pass
    def add_recent(self, path):
        if not path: return
        items = list(self.recent_list.get(0, tk.END))
        if path in items: items.remove(path)
        items.insert(0, path)
        self.recent_list.delete(0, tk.END)
        for p in items[:5]: self.recent_list.insert(tk.END, p)
        with open('recent_projects.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(items[:5]))

# ================== 主应用 ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EVOiCE 完整版")
        self.geometry("1200x700")
        self.configure(bg=STYLE['bg'])
        self.report_callback_exception = self._show_tk_error

        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="新建", command=lambda: self.show_editor())
        file_menu.add_command(label="打开 (.evgc)", command=self.open_project_file)
        file_menu.add_command(label="保存", command=self.save_current)
        file_menu.add_command(label="导出 WAV", command=self.export_current_wav)
        file_menu.add_separator()
        file_menu.add_command(label="返回首页", command=self.show_home)
        menubar.add_cascade(label="文件", menu=file_menu)
        menubar.add_cascade(label="编辑", menu=Menu(menubar, tearoff=0))
        menubar.add_cascade(label="工程", menu=Menu(menubar, tearoff=0))
        menubar.add_cascade(label="工具", menu=Menu(menubar, tearoff=0))
        menubar.add_cascade(label="帮助", menu=Menu(menubar, tearoff=0))
        self.config(menu=menubar)

        self.singers_dir = "singers"
        if not os.path.exists(self.singers_dir):
            os.makedirs(os.path.join(self.singers_dir, "默认歌手"))

        self.home = None
        self.editor = None
        self.show_home()

    def _show_tk_error(self, exc, val, tb):
        err = "".join(traceback.format_exception(exc, val, tb))
        messagebox.showerror("异常", err)

    def show_home(self):
        if self.editor: self.editor.pack_forget()
        if not self.home: self.home = HomePage(self, self)
        self.home.pack(fill=tk.BOTH, expand=True)

    def show_editor(self, project_path=None):
        if self.home: self.home.pack_forget()
        if not self.editor: self.editor = EditorPage(self, self)
        self.editor.pack(fill=tk.BOTH, expand=True)
        if project_path:
            try:
                with open(project_path, 'r', encoding='utf-8') as f: data = json.load(f)
                self.editor.piano.load_project_data(data)
                self.editor.project_path = project_path
                self.home.add_recent(project_path)
            except: pass

    def open_project_file(self):
        path = filedialog.askopenfilename(filetypes=[("EVOiCE工程", "*.evgc")])
        if path: self.show_editor(path)

    def save_current(self):
        if self.editor: self.editor.save_project()

    def export_current_wav(self):
        if self.editor: self.editor.export_wav()

    def add_recent(self, path):
        if self.home: self.home.add_recent(path)

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        emergency_alert("启动失败", str(e))