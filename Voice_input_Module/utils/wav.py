import os
import numpy as np
import librosa
import librosa.display
from glob import glob
from tqdm import tqdm
from PIL import Image

'''
def segment_audio(y, sr, duration= 30):
    print("sr=",sr)
    segsamples = int(duration * sr)
    n_seg = len(y) // segsamples
    segments = []
    for i in range(n_seg):
        start = i * segsamples
        end = start + segsamples
        segments.append(y[start:end])
    
    return segments

def resize_spec(spec, size=(256, 256)):
    spec = np.array(spec, dtype=np.float32)
    finite = spec[np.isfinite(spec)]
    if finite.size == 0:
        spec_min, spec_max = 0.0, 1.0
    else:
        spec_min, spec_max = finite.min(), finite.max()

    spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-10) * 255.0
    spec_img = Image.fromarray(spec_norm.astype(np.uint8))
    spec_img_resized = spec_img.resize(size, Image.LANCZOS)
    spec_resized = np.array(spec_img_resized, dtype=np.float32) / 255.0 * (spec_max - spec_min) + spec_min
    return spec_resized

def process_wav_file(seg, wav_path, spec_path):
    y, sr = librosa.load(wav_path, sr=None) 
    S1 =[]
    for s in seg:
        S = librosa.feature.melspectrogram(y = s, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max, amin=1e-10)
        spec_resized = resize_spec(S_db) 
        S1.append(spec_resized)
    spec_resized = np.array(S1)
    np.save(spec_path, spec_resized)  
'''

def process_wav_file(wav_path, spec_path, target_fps=30):
    y, sr = librosa.load(wav_path, sr=None)
    hop_length = int(sr / target_fps)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec_resized = S_db.T
    print(f"      特征形状 (Frames, Mels): {spec_resized.shape}")
    np.save(spec_path, spec_resized.astype(np.float32))

splits=["trn", "tst", "val"]
roles=["interloctr", "main-agent"]

def process_all(base_dir):
    for split in splits:
        for role in roles:
            print(f"Processing split: {split}, role: {role}")

            wav_dir = os.path.join(base_dir, split, role, 'wav')
            spec_dir = os.path.join(base_dir, split, role, 'spec')
            os.makedirs(spec_dir, exist_ok=True)  
            wav_files = glob(os.path.join(wav_dir, '*.wav'))  
            for wav_file in tqdm(wav_files, desc=f"Processing {split}"):
                fname = os.path.splitext(os.path.basename(wav_file))[0]  
                spec_path = os.path.join(spec_dir, fname + '.npy')
                try:
                    '''
                    y, sr = librosa.load(wav_file, sr=None)
                    print(wav_file,len(y))
                    seg = segment_audio(y, sr)
                    print(len(seg))
                    process_wav_file(seg, wav_file, spec_path)
                    '''
                    process_wav_file(wav_file, spec_path)  
                except Exception as e:
                    print(f"Failed for {wav_file}: {e}")

if __name__ == "__main__":
    process_all("genea2023_dataset")