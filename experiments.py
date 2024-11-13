# Creating Synthetic EEG data 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression

sampling_rate= 500 # in Hertz
duration = 10 # in Seconds
time = np.linspace(0, duration, duration * sampling_rate)

def generate_synthetic_eeg(frequency, amplitude=1.0):
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = np.random.normal(0, 0.5, signal.shape)
    return signal + noise

eeg_left_hand = generate_synthetic_eeg(frequency= 10) #in Hertz
eeg_right_hand = generate_synthetic_eeg(frequency = 12) #in Hertz

eeg_data = np.vstack((eeg_left_hand, eeg_right_hand))
labels = np.array([0, 1]) 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut/nyquist
    high = highcut/nyquist
    b, a = butter(order, [low, high], btype='band')
    return b,a 

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=sampling_rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
eeg_left_hand_filtered = bandpass_filter(eeg_left_hand)
eeg_right_hand_filtered = bandpass_filter(eeg_right_hand)



# Function to compute power spectral density in specific bands
def compute_band_power(data, fs):
    freqs, psd = welch(data, fs)
    # Define frequency bands
    bands = {'delta': (1, 4),
             'theta': (4, 8),
             'alpha': (8, 13),
             'beta': (13, 30)}
    band_power = {}
    for band in bands:
        freq_ix = np.where((freqs >= bands[band][0]) & (freqs <= bands[band][1]))
        band_power[band] = np.mean(psd[freq_ix])
    return band_power

features_left = compute_band_power(eeg_left_hand_filtered, sampling_rate)
features_right = compute_band_power(eeg_right_hand_filtered, sampling_rate)

feature_matrix = np.array([[features_left['alpha'], features_left['beta']],
                           [features_right['alpha'], features_right['beta']]])

# Using Logistic Regression to classify EEG signals based on extracted features


# Labels: 0 for left hand, 1 for right hand
labels = np.array([0, 1])

clf = LogisticRegression()
clf.fit(feature_matrix, labels)

new_signal = generate_synthetic_eeg(frequency=10)  # Simulating new left-hand intention
new_signal_filtered = bandpass_filter(new_signal)
new_features = compute_band_power(new_signal_filtered, sampling_rate)
new_feature_vector = np.array([[new_features['alpha'], new_features['beta']]])

# Prediction intention
prediction = clf.predict(new_feature_vector)
intention = 'Left Hand' if prediction[0] == 0 else 'Right Hand'
print(f"Predicted Intention: {intention}")

# Sending the commands to an electrical stimulated device.
def send_control_command(intention):
    if intention == 'Left Hand':
        command = 'Stimulate left hand muscles'
    else:
        command = 'Stimulate right hand muscles'
    print(f"Sending command: {command}")
    # Here, you would integrate with the actual device API or interface
    # For simulation purpose, we would just print the command



# Drawing a plot for better understanding (Visualization of EEG signals and PSD)
# Plot the raw and filtered EEG signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, eeg_left_hand, label='Raw EEG Left Hand')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, eeg_left_hand_filtered, label='Filtered EEG Left Hand', color='orange')
plt.legend()
plt.tight_layout()
plt.savefig('Synthetic EEG_SIGNALS')
plt.show()

# Plotting the Power Spectral Density
def plot_PSD():
    freqs, psd = welch(eeg_left_hand_filtered, sampling_rate)
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs, psd)
    plt.title('Power Spectral Density of Filtered EEG (Left Hand)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.savefig('Synethetic PSD')
    plt.show()
plot_PSD()




