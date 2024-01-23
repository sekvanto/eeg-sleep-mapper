import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from tkinter import Tk, filedialog, Button, Label

##import cProfile

def power_spectral_density(signal, sampling_rate):
    freqs = rfftfreq(len(signal), 1 / sampling_rate)
    psd = np.abs(rfft(signal))**2
    return freqs, psd

# Function to calculate the relative power of each band
def calculate_relative_power(psd, freqs, bands, total_power):
    band_power = {}
    for band in bands.keys():
        idx_band = np.logical_and(freqs >= bands[band][0], freqs <= bands[band][1])
        band_power[band] = np.sum(psd[idx_band])
    relative_power = {band: power / total_power for band, power in band_power.items()}
    return relative_power

# Function to adjust parameters based on the duration of the sleep recording
def adjust_parameters_based_on_duration(data, sampling_rate, base_duration=20*60, 
                                        base_window_size=1, base_step_size=0.1, base_smoothing_window=20):
    total_duration = len(data) / sampling_rate
    scale_factor = total_duration / base_duration

    window_size = int(max(int(base_window_size * sampling_rate * scale_factor), sampling_rate))
    step_size = int(max(int(base_step_size * sampling_rate * scale_factor), int(sampling_rate / 2)))
    smoothing_window = int(max(int(base_smoothing_window * scale_factor), 1))
    
    return window_size, step_size, smoothing_window

# Optimized function to plot relative power over time
def plot_relative_power_over_time(data, bands, sampling_rate):
    window_size, step_size, smoothing_window = adjust_parameters_based_on_duration(data, sampling_rate)

    # Initialize relative power dictionary and frequency array
    relative_powers = {band: np.empty(0) for band in bands}
    freqs = rfftfreq(window_size, 1 / sampling_rate)

    # Loop over the data with optimized array operations
    for start in range(0, len(data) - window_size, step_size):
        segment = data[start:start + window_size]
        psd = np.abs(rfft(segment))**2
        total_power = np.sum(psd)

        for band in bands:
            idx_band = np.logical_and(freqs >= bands[band][0], freqs <= bands[band][1])
            band_power = np.sum(psd[idx_band])
            relative_powers[band] = np.append(relative_powers[band], (band_power / total_power) * 100)

    # Efficient smoothing using cumulative sum
    for band in bands:
        cumsum = np.cumsum(np.insert(relative_powers[band], 0, 0)) 
        relative_powers[band] = (cumsum[smoothing_window:] - cumsum[:-smoothing_window]) / smoothing_window

    # Time vector for plotting in minutes
    time_vector = np.arange(len(relative_powers['Delta'])) * step_size / sampling_rate / 60

    # Plotting
    plt.figure(figsize=(15, 7))
    for band, color in zip(bands, ['b', 'g', 'r', 'c']):
        plt.plot(time_vector, relative_powers[band], label=band, color=color, linewidth=2)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Relative Power (%)')
    plt.title('Smoothed Relative Brainwave Power Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to process EEG file
def process_eeg_file(file_path, channel):
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30)
    }
    
    data = pd.read_csv(file_path, delimiter=';')
    sampling_rate = data['Sampling Rate'].iloc[0]
    
    eeg_data = data[channel].dropna()
    
    plot_relative_power_over_time(eeg_data, bands, sampling_rate)

# Function to handle channel selection and process the file
def handle_channel_selection(file_path, channel):
    root.destroy()
    process_eeg_file(file_path, channel)

# Main function to run the program
def main():
    global root
    root = Tk()
    root.title("EEG Channel Selection")

    # Display a label
    label = Label(root, text="Select the EEG Channel to Process:")
    label.pack(pady=10)

    # Button for Channel 1
    channel1_button = Button(root, text="Channel 1", command=lambda: handle_channel_selection(file_path, 'Channel 1'))
    channel1_button.pack(pady=5)

    # Button for Channel 2
    channel2_button = Button(root, text="Channel 2", command=lambda: handle_channel_selection(file_path, 'Channel 2'))
    channel2_button.pack(pady=5)

    # File selection
    file_path = filedialog.askopenfilename()
    if not file_path:
        root.destroy()
        return

    root.mainloop()

if __name__ == "__main__":
    main()
