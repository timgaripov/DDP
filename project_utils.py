import librosa
import librosa.display
import midi
import numpy as np
import pandas as pd

def parse_midi(fname, instrument_id):
    pattern = midi.read_midifile(fname)
    bpm = [x for x in pattern[0] if isinstance(x, midi.SetTempoEvent)][0].bpm
    tdict = {
        midi.NoteOnEvent: 1,
        midi.NoteOffEvent: -1,
    }
    events = []
    for i in range(1, 5):    
        t = np.cumsum([e.tick for e in pattern[i]], dtype=np.int32)                         
        events.extend([(t, i - 1, tdict[e.__class__], e.pitch) for (t, e) in zip(t, pattern[i]) if e.__class__ in tdict])
    events = list(sorted(events, key=lambda x: (x[0], x[2])))
    
    chord = [0] * 4
    
    chords = []    
        
    for i in range(len(events)):
        if (i > 0) and (events[i - 1][0] != events[i][0]):
            chords.append((events[i - 1][0], tuple(chord)))            
        if events[i][2] == 1:
            chord[events[i][1]] = events[i][3]            
        else:            
            chord[events[i][1]] = 0            
    chords.append((events[-1][0], tuple(chord)))
    
    score = []
    for (t, chord) in chords:
        if score and score[-1][-1] == chord[instrument_id]:
            continue
        score.append((t, t / pattern.resolution, t * 60000.0 / bpm / pattern.resolution, chord[instrument_id]))
            
    return pd.DataFrame(score)

def wav_features(fname):
    sr = 44100
    r = librosa.load(fname, sr=sr, offset=0.023)[0]
    n_fft, hop_length = 1024, 441
    features = np.vstack([
        librosa.feature.rmse(r, frame_length=n_fft, hop_length=hop_length),
        librosa.feature.spectral_centroid(r, sr=sr, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.spectral_bandwidth(r, sr=sr, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.mfcc(r, sr=sr, n_mfcc=5, n_fft=n_fft, hop_length=hop_length)
    ]).T
    return features

def grountruth_matrix(alignment, s):
    b = alignment[2].copy()
    mx = np.ceil(np.max(alignment[2]) * 2) / 2    
    Y = ((b[:, None] >= s[None, :-1]) & (b[:, None] < s[None, 1:])).astype(np.int32)
    return Y

def prepare(scores, alignments):
    S = set(sum(map(lambda x: x[3].tolist(), scores), []))
    K = len(S)    
    id_to_key = {i: k for i, k in enumerate(S)}
    key_to_id = {k: i for i, k in enumerate(S)}
    
    GTs = []
    Bs = []
    
    SYs = []
    
    for alignment, score in zip(alignments, scores):        
        GT = grountruth_matrix(alignment, score[1])
        B = np.array([key_to_id[v] for v in score[3][:-1]], dtype=np.int32)
        GTs.append(GT)
        Bs.append(B)
        
        SY = np.zeros(alignment.shape[0], dtype=np.int32)
        j = 0
        for i in range(alignment.shape[0]):
            while (score[1][j + 1] <= alignment.iloc[i, 2]):
                j += 1            
            SY[i] = key_to_id[score[3][j]]
        SYs.append(SY)
        
    return GTs, Bs, SYs, K