import librosa
from scipy.io import wavfile
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm



def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn + '_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)

def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def split_wavs(direct, toDirect, dtTime, sr, threshold):
    src_root = direct
    dst_root = toDirect
    dt = dtTime
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            signal, rate = librosa.load(src_fn,sr)
            mask = envelope(signal, rate, threshold=threshold)
            signal = signal[mask]
            delta_sample = int(dt * rate)

            if signal.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.float32)
                sample[:signal.shape[0]] = signal
                save_sample(sample, rate, target_dir, fn, 0)

            else:
                trunc = signal.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, signal.shape[0] - trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = signal[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)

if __name__ == '__main__':
    direct = 'audio_train'  # 'Thư mục train'
    toDirect = 'clean'  # Thư mục tiền xử lý
    dtTime = 1.0  # thời gian lấy mẫu
    sr = 16000  # tần số lấy mẫu
    threshold = 0.0005  # threshold
    split_wavs(direct, toDirect, dtTime, sr, threshold)
