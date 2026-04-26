import numpy as np
import soundfile as sf
import os

def create_sample_voicebank():
    """创建示例音源文件"""
    voicebank_dir = "voicebank"
    
    if not os.path.exists(voicebank_dir):
        os.makedirs(voicebank_dir)
    
    sample_rate = 44100
    
    # 创建几个基本音高的音源文件
    notes = [
        ('c4', 261.63),  # C4
        ('d4', 293.66),  # D4
        ('e4', 329.63),  # E4
        ('f4', 349.23),  # F4
        ('g4', 392.00),  # G4
        ('a4', 440.00),  # A4
        ('b4', 493.88),  # B4
        ('c5', 523.25)   # C5
    ]
    
    for note_name, frequency in notes:
        duration = 1.0  # 1秒
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 生成正弦波
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # 添加谐波使其更自然
        audio += 0.1 * np.sin(2 * np.pi * 2 * frequency * t)
        audio += 0.05 * np.sin(2 * np.pi * 3 * frequency * t)
        
        # 应用包络
        attack = int(sample_rate * 0.05)   # 50ms起音
        decay = int(sample_rate * 0.15)    # 150ms衰减
        sustain = int(sample_rate * 0.7)   # 700ms持续
        release = int(sample_rate * 0.1)   # 100ms释放
        
        envelope = np.zeros_like(audio)
        
        # 起音阶段
        for i in range(attack):
            envelope[i] = i / attack
        
        # 衰减阶段
        for i in range(attack, attack + decay):
            envelope[i] = 1.0 - 0.3 * (i - attack) / decay
        
        # 持续阶段
        envelope[attack + decay : attack + decay + sustain] = 0.7
        
        # 释放阶段
        for i in range(attack + decay + sustain, len(audio)):
            pos = i - (attack + decay + sustain)
            envelope[i] = 0.7 * (1.0 - pos / release)
        
        # 应用包络
        audio *= envelope
        
        # 保存文件
        file_path = os.path.join(voicebank_dir, f"{note_name}.wav")
        sf.write(file_path, audio, sample_rate)
        print(f"已创建音源文件：{file_path}")
    
    # 创建默认音源文件
    default_file = os.path.join(voicebank_dir, "default.wav")
    sf.write(default_file, audio, sample_rate)
    print(f"已创建默认音源文件：{default_file}")

if __name__ == "__main__":
    create_sample_voicebank()
    print("示例音源库创建完成！")