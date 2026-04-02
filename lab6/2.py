import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal

def visualize_spectrogram(audio_path, output_image_path="spectrogram.png", title="Spectrogram", max_duration=30):
    with wave.open(audio_path, 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
    
    max_samples = int(max_duration * 44100)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"  Ограничено до {max_duration} секунд для визуализации")
    
    plt.figure(figsize=(14, 8))
    
    nperseg = 2048
    noverlap = 1536
    
    f, t, Sxx = signal.spectrogram(audio, fs=44100, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
    
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='inferno')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 8000)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Spectrogram saved to: {output_image_path}")

def visualize_comparison(original_audio_path, stego_audio_path, output_path="comparison.png", max_duration=20):
    with wave.open(original_audio_path, 'rb') as wav:
        frames = wav.readframes(wav.getparams().nframes)
        original = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
    
    with wave.open(stego_audio_path, 'rb') as wav:
        frames = wav.readframes(wav.getparams().nframes)
        stego = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
    
    max_samples = int(max_duration * 44100)
    if len(original) > max_samples:
        original = original[:max_samples]
        stego = stego[:max_samples]
        print(f"  Ограничено до {max_duration} секунд для визуализации")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    nperseg = 2048
    noverlap = 1536
    
    f, t, Sxx_orig = signal.spectrogram(original, fs=44100, nperseg=nperseg, noverlap=noverlap)
    axes[0].pcolormesh(t, f, 10 * np.log10(Sxx_orig + 1e-10), shading='auto', cmap='inferno')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Original Audio Spectrogram')
    axes[0].set_ylim(0, 8000)
    
    f, t, Sxx_stego = signal.spectrogram(stego, fs=44100, nperseg=nperseg, noverlap=noverlap)
    im = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_stego + 1e-10), shading='auto', cmap='inferno')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Stego Audio Spectrogram')
    axes[1].set_ylim(0, 8000)
    
    plt.colorbar(im, ax=axes, label='Power (dB)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison saved to: {output_path}")

def visualize_text_region(audio_path, start_time, duration, output_path="text_region.png"):
    with wave.open(audio_path, 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
    
    start_sample = int(start_time * 44100)
    end_sample = int((start_time + duration) * 44100)
    segment = audio[start_sample:end_sample]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    time = np.linspace(0, len(segment)/44100, len(segment))
    plt.plot(time, segment)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Segment ({start_time:.1f}s to {start_time+duration:.1f}s)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    nperseg = 1024
    noverlap = 768
    f, t, Sxx = signal.spectrogram(segment, fs=44100, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t + start_time, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='inferno')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram of Text Region')
    plt.ylim(0, 1000)
    plt.colorbar(label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Text region visualization saved to: {output_path}")

def visualize_image_region(audio_path, output_path="image_spectrogram.png", max_seconds=30):
    with wave.open(audio_path, 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
    
    image_duration = 128 * 128 * 0.02
    max_samples = int(min(max_seconds, image_duration) * 44100)
    audio = audio[:max_samples]
    
    plt.figure(figsize=(14, 8))
    
    nperseg = 4096
    noverlap = 3072
    f, t, Sxx = signal.spectrogram(audio, fs=44100, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
    
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='inferno')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Image Region Spectrogram (500-4000 Hz tones)')
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 5000)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Image region spectrogram saved to: {output_path}")

if __name__ == "__main__":
    visualize_spectrogram("stego_audio.wav", "stego_spectrogram.png", "Stego Audio Spectrogram", max_duration=30)
    
    visualize_image_region("stego_audio.wav", "image_spectrogram.png")
    
    visualize_text_region("stego_audio.wav", start_time=328.0, duration=15.0, output_path="text_spectrogram.png")