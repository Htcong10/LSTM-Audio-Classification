import librosa
from keras.models import load_model
from sklearn.metrics import accuracy_score
from clean import  envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pandas as pd
def make_prediction(modelFile,logDir,testFile,timeIn,sampleRate,thresHold):

    model = load_model(modelFile,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/*'.format(testFile), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])

    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    results = []
    label_pred = []
    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        wav,rate = librosa.load(wav_fn,sampleRate)
        mask = envelope(wav, rate, threshold=thresHold)
        clean_wav = wav[mask]
        step = int(sampleRate*timeIn)
        batch = []
        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        label_pred.append(classes[y_pred])
        print('Actual class: {}, Predicted class: {}'.format(wav_fn, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', logDir), np.array(results))
    return label_pred

if __name__ == '__main__':
    classes = sorted(os.listdir('clean'))
    modelFile='models/lstm.h5'
    logDir='y-pred'
    testFile='TestPredict'
    timeIn=1.0
    sampleRate=16000
    thresHold=0.0005
    label_pred=make_prediction(modelFile,logDir,testFile,timeIn,sampleRate,thresHold)
    df=pd.read_csv('Test.csv')
    df['y_pred']=label_pred
    df.to_csv('Pred.csv', index=False)
    pre= pd.read_csv('Pred.csv')
    acc_score = accuracy_score(y_true=pre.label, y_pred=pre.y_pred)
    print(acc_score)