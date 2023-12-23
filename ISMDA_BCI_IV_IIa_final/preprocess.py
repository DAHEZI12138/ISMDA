import numpy as np
from scipy.signal import butter, lfilter
import mne

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_dc_bias(data):
    return data - np.mean(data, axis=0)

def remove_artifacts(data, fs):
    n_channels = 40  # Number of EEG channels
    info = mne.create_info(n_channels, fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)

    # Visualize the components to identify which are artifacts
    ica.plot_components()
    # Identify the artifact components (e.g., components 0 and 1),
    artifact_components = [0, 1]
    ica.exclude = artifact_components

    # Apply the ICA to the raw data, effectively removing the artifacts
    raw_corrected = ica.apply(raw.copy())
    return raw_corrected

def segment_and_baseline_correction(data, fs, segment_length, baseline_period):

    segmented_data = []
    for start in range(0, len(data), segment_length):
        end = start + segment_length
        if end <= len(data):
            segment = data[start:end]
            baseline = np.mean(segment[:baseline_period], axis=0)
            segment -= baseline  # baseline_correction
            segmented_data.append(segment)
    return segmented_data

def remove_outliers(data):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    # Calculate z-scores
    z_scores = (data - mean) / std
    # Choose a threshold (e.g., 3 standard deviations)
    threshold = 3
    # Identify outliers
    outliers = np.where(np.abs(z_scores) > threshold)
    # Remove outliers
    data_cleaned = np.delete(data, outliers)
    return data_cleaned

def preprocess(eeg_data, fs=250):
    eeg_data = remove_dc_bias(eeg_data)

    lowcut = 1.0
    highcut = 50.0
    eeg_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs, order=6)

    eeg_data = remove_artifacts(eeg_data, fs)

    segment_length = 250
    baseline_period = 50
    eeg_data = segment_and_baseline_correction(eeg_data, fs, segment_length, baseline_period)

    eeg_data = remove_outliers(eeg_data)

    return eeg_data