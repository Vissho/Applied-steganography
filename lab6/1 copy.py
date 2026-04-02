import numpy as np
import wave
from PIL import Image
import os
from scipy import signal

class AudioSteganography:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def image_to_audio(self, image_path, output_path, pixel_duration_ms=20):
        img = Image.open(image_path).convert('L')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        h, w = img_array.shape
        
        pixel_duration = pixel_duration_ms / 1000
        samples_per_pixel = int(self.sample_rate * pixel_duration)
        
        total_samples = h * w * samples_per_pixel
        audio = np.zeros(total_samples, dtype=np.float32)
        
        freq_min = 500
        freq_max = 4000
        
        for i in range(h):
            for j in range(w):
                pixel_value = img_array[i, j]
                freq = freq_min + (pixel_value / 255.0) * (freq_max - freq_min)
                
                t = np.linspace(0, pixel_duration, samples_per_pixel, endpoint=False)
                tone = 0.5 * np.sin(2 * np.pi * freq * t)
                
                start_idx = (i * w + j) * samples_per_pixel
                audio[start_idx:start_idx + samples_per_pixel] = tone
        
        audio = (audio * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        return img_array.shape, len(audio)

    def text_to_audio_echo_separate(self, image_audio_path, message, output_path,
                                     delay0_ms=80, delay1_ms=160, echo_amp=0.5):
        with wave.open(image_audio_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            image_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
        
        binary = ''.join(format(ord(c), '08b') for c in message)
        binary += '00000000'
        
        print(f"  Сообщение: '{message}'")
        print(f"  Бинарная строка: {binary}")
        
        bit_duration_ms = 300
        bit_samples = int(self.sample_rate * bit_duration_ms / 1000)
        delay0_samples = int(self.sample_rate * delay0_ms / 1000)
        delay1_samples = int(self.sample_rate * delay1_ms / 1000)
        
        silence_duration = 1.0
        silence_samples = int(self.sample_rate * silence_duration)
        
        text_audio = np.zeros(bit_samples * len(binary), dtype=np.float32)
        
        for bit_idx, bit in enumerate(binary):
            start = bit_idx * bit_samples
            end = start + bit_samples
            
            carrier = np.random.randn(bit_samples) * 0.05
            
            if bit == '0':
                delay = delay0_samples
            else:
                delay = delay1_samples
            
            text_audio[start:end] = carrier
            
            echo_start = start + delay
            echo_end = end + delay
            if echo_end <= len(text_audio):
                text_audio[echo_start:echo_end] += carrier * echo_amp
            elif echo_start < len(text_audio):
                available = len(text_audio) - echo_start
                text_audio[echo_start:] += carrier[:available] * echo_amp
        
        max_val = np.max(np.abs(text_audio))
        if max_val > 0:
            text_audio = text_audio / max_val * 0.3
        
        combined_audio = np.concatenate([image_audio, np.zeros(silence_samples, dtype=np.float32), text_audio])
        
        combined_audio = np.clip(combined_audio, -1, 1)
        audio_out = (combined_audio * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_out.tobytes())
        
        print(f"  Встроено бит: {len(binary)}")
        return len(binary)

    def extract_image_from_spectrogram(self, audio_path, output_image_path):
        with wave.open(audio_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
        
        h = 128
        w = 128
        pixel_duration_ms = 20
        samples_per_pixel = int(self.sample_rate * pixel_duration_ms / 1000)
        
        image_samples = h * w * samples_per_pixel
        audio_for_image = audio[:image_samples]
        
        img_array = np.zeros((h, w), dtype=np.uint8)
        
        freq_min = 500
        freq_max = 4000
        freq_range = freq_max - freq_min
        
        fft_size = 4096
        
        for i in range(h):
            for j in range(w):
                pixel_idx = i * w + j
                start = pixel_idx * samples_per_pixel
                end = min(start + samples_per_pixel, len(audio_for_image))
                
                if start >= len(audio_for_image):
                    continue
                
                segment = audio_for_image[start:end]
                
                if len(segment) < fft_size:
                    segment = np.pad(segment, (0, fft_size - len(segment)), mode='constant')
                
                freqs, times, Sxx = signal.spectrogram(
                    segment, fs=self.sample_rate,
                    nperseg=fft_size, noverlap=fft_size//2,
                    mode='magnitude'
                )
                
                if len(freqs) > 0 and Sxx.shape[1] > 0:
                    power = np.mean(Sxx, axis=1)
                    
                    smooth_power = np.convolve(power, np.ones(5)/5, mode='same')
                    
                    peak_idx = np.argmax(smooth_power)
                    freq_detected = freqs[peak_idx]
                    
                    if freq_detected < freq_min:
                        freq_detected = freq_min
                    if freq_detected > freq_max:
                        freq_detected = freq_max
                    
                    pixel_val = int((freq_detected - freq_min) / freq_range * 255)
                    pixel_val = min(255, max(0, pixel_val))
                    
                    if pixel_val < 10:
                        pixel_val = 0
                    if pixel_val > 245:
                        pixel_val = 255
                    
                    img_array[i, j] = pixel_val
        
        img = Image.fromarray(img_array, mode='L')
        img.save(output_image_path)
        
        return img_array.shape

    def extract_text_from_echo_separate(self, audio_path, delay0_ms=80, delay1_ms=160, 
                                          bit_duration_ms=300):
        with wave.open(audio_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
        
        h = 128
        w = 128
        pixel_duration_ms = 20
        samples_per_pixel = int(self.sample_rate * pixel_duration_ms / 1000)
        image_samples = h * w * samples_per_pixel
        
        silence_duration = 1.0
        silence_samples = int(self.sample_rate * silence_duration)
        
        text_start = image_samples + silence_samples
        text_audio = audio[text_start:]
        
        bit_samples = int(self.sample_rate * bit_duration_ms / 1000)
        delay0_samples = int(self.sample_rate * delay0_ms / 1000)
        delay1_samples = int(self.sample_rate * delay1_ms / 1000)
        
        max_bits = len(text_audio) // bit_samples
        print(f"  text_start: {text_start}")
        print(f"  text_audio length: {len(text_audio)}")
        print(f"  bit_samples: {bit_samples}")
        print(f"  max_bits: {max_bits}")
        
        binary = ''
        
        for bit_idx in range(max_bits):
            start = bit_idx * bit_samples
            end = min(start + bit_samples, len(text_audio))
            segment = text_audio[start:end]
            
            if len(segment) < bit_samples:
                break
            
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if autocorr[0] == 0:
                corr0 = 0
                corr1 = 0
            else:
                corr0 = autocorr[delay0_samples] / autocorr[0] if delay0_samples < len(autocorr) else 0
                corr1 = autocorr[delay1_samples] / autocorr[0] if delay1_samples < len(autocorr) else 0
            
            if bit_idx < 30:
                print(f"    Бит {bit_idx}: corr0={corr0:.6f}, corr1={corr1:.6f}")
            
            if corr0 > corr1:
                binary += '0'
            else:
                binary += '1'
            
            if len(binary) >= 16 and binary[-16:] == '0000000000000000':
                binary = binary[:-16]
                print(f"  Найден маркер конца на бите {bit_idx}")
                break
        
        print(f"  Извлечено бит: {len(binary)}")
        print(f"  Бинарная строка: {binary[:64]}...")
        
        message = ''
        for i in range(0, len(binary), 8):
            if i + 8 <= len(binary):
                byte = binary[i:i+8]
                try:
                    char = chr(int(byte, 2))
                    if 32 <= ord(char) <= 126:
                        message += char
                except:
                    pass
        
        return message

def create_composite_audio(image_path, text_message, output_path):
    stego = AudioSteganography()
    
    temp_audio = "temp_image_audio.wav"
    
    print("\n1. Встраивание изображения в аудио:")
    shape, audio_len = stego.image_to_audio(image_path, temp_audio, pixel_duration_ms=20)
    print(f"   Изображение {shape[1]}x{shape[0]} преобразовано в аудио")
    print(f"   Длина аудио: {audio_len} семплов ({audio_len/44100:.1f} сек)")
    
    print("\n2. Встраивание текста методом эхо:")
    bits = stego.text_to_audio_echo_separate(temp_audio, text_message, output_path,
                                              delay0_ms=80, delay1_ms=160, echo_amp=0.5)
    print(f"   Встроено {bits} бит текста")
    
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    print(f"\nРезультат сохранен в {output_path}")
    return output_path

def analyze_and_extract(audio_path):
    print("=" * 70)
    stego = AudioSteganography()
    
    img_output = "extracted_image.png"
    shape = stego.extract_image_from_spectrogram(audio_path, img_output)
    
    print("\nИзвлечение текста методом эхо:")
    text = stego.extract_text_from_echo_separate(audio_path, delay0_ms=80, delay1_ms=160, 
                                                   bit_duration_ms=300)
    print(f"   Извлеченный текст: '{text}'")
    
    return img_output, text

def main():
    output_audio = "stego_audio.wav"
    
    extracted_img, extracted_text = analyze_and_extract(output_audio)
    
    print("=" * 70)
    print(f"Извлеченное изображение: {extracted_img}")
    print(f"Извлеченный текст: '{extracted_text}'")
    

if __name__ == "__main__":
    main()

# Частота дискретизации: 44.1 кГц (44100 Гц)

# Параметры изображения:
# Разрешение: 128×128 пикселей
# Длительность тона на пиксель: 20 мс
# Диапазон частот: 500–4000 Гц
# Окно: Ханна (плавное нарастание/затухание)

# Параметры эхо-встраивания текста:
# Длительность бита: 300 мс
# Задержка эха для бита '0': 80 мс (3528 семплов)
# Задержка эха для бита '1': 160 мс (7056 семплов)
# Амплитуда эха: 0.5 (50% от исходного сигнала)
# Несущий сигнал: белый шум с амплитудой 0.05
# Пауза между изображением и текстом: 1 секунда (тишина)

# Маркер конца сообщения: 16 нулевых бит (0000000000000000)