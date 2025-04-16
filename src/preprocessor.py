import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch


class CopPreprocessor:
    """
    The `CopPreprocessor` class provides methods for preprocessing center of pressure (CoP) signals, 
    including filtering, Fourier transform operations, and conversion to center of mass (CoM) signals.

    Attributes:
        fs (float): Sampling frequency of the signal.
        cutoff_freq (float): Cutoff frequency for the Butterworth filter.
        order (int): Order of the Butterworth filter.
        omega_0 (float): Natural frequency for the frequency relationship function.

    Methods:
        __init__(self, fs, cutoff_freq, order, omega_0):
            Initializes the preprocessor with the given parameters.

        _design_butterworth_filter(self, f_sample, f_cut, n):
            Designs a low-pass Butterworth filter with the specified parameters.

        _apply_butterworth_filter(self, x, b, a):
            Applies the Butterworth filter to the input signal.

        freq_rel(self, freq):
            Computes the frequency relationship function to convert CoP to CoM.

        compute_fourier_transform(self, y):
            Computes the Fourier transform of the input signal and returns the frequencies and transform values.

        compute_inverse_fourier_transform(self, Y):
            Computes the inverse Fourier transform of the input spectrum and returns the real-valued signal.

        _filter_data(self, x):
            Filters the input signal using a Butterworth filter and returns the filtered signal.

        convert_to_com(self, x):
            Converts the input CoP signal to a CoM signal by filtering, applying the frequency relationship, 
            and performing inverse Fourier transform.
    """

    def __init__(self, fs, cutoff_freq, order, omega_0):
        self.fs = fs
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.omega_0 = omega_0

    def _design_butterworth_filter(self, f_sample, f_cut, n):
        nyq = 0.5 * f_sample  # Nyquist frequency
        normal_cutoff = f_cut / nyq # Normal cut off frequency
        b, a = butter(n, normal_cutoff, btype='low', analog=False)
        return b, a

    def _apply_butterworth_filter(self, x, b, a):
        y = filtfilt(b, a, x)
        return y

    def freq_rel(self, freq):
        # Frequency relationship function which converts CoP (center of pressure) to CoM (center of mass)
        # This is a simple second-order low-pass filter model
        return self.omega_0 ** 2 / (self.omega_0 ** 2 + (2*np.pi*freq) ** 2)

    def compute_fourier_transform(self, y):
        # Compute the Fourier transform of the input signal
        # and return the frequencies and the Fourier transform values
        N = len(y)
        dt = 1 / self.fs
        freqs = np.fft.fftfreq(N, dt)
        Y = np.fft.fft(y)
        return freqs, Y

    def compute_inverse_fourier_transform(self, Y):
        # Compute the inverse Fourier transform of the input spectrum
        # and return the real part of the inverse transform
        y = np.fft.ifft(Y)
        return y.real  # Take the real part to get a real-valued signal

    def _filter_data(self, x):
        # Filter the input signal using a Butterworth filter
        # and return the filtered signal
        b, a = self._design_butterworth_filter(self.fs, self.cutoff_freq, self.order)
        filtered = self._apply_butterworth_filter(x, b, a)
        return filtered

    def convert_to_com(self, x):
        filtered = self._filter_data(x)
        freqs, X = self.compute_fourier_transform(filtered)
        com_omega = X * self.freq_rel(freqs)
        com_t = self.compute_inverse_fourier_transform(com_omega)
        return com_t


class Summarystats:
    """
    A class for calculating summary statistics from time-series data, including histograms 
    and power spectral density (PSD) features based on the Method of two studies:
    [1] https://www.nature.com/articles/s41598-017-02372-1
    [2] https://pubmed.ncbi.nlm.nih.gov/33261318/

    Attributes:
        fs (float): Sampling frequency of the data.
        nbins (int): Number of bins for histogram calculations. Default is 15.
        n_ftr_psd (int): Number of frequency features to extract from PSD. Default is 10.
        n_window (int): Window size (in seconds) for PSD calculation. Default is 40.
        overlap_pr (float): Overlap percentage for PSD calculation windows. Default is 0.75.
    Methods:
        observation_bins(x_array):
            Computes bin edges for histogram calculations based on the maximum absolute 
            value or three times the standard deviation of the input data.
        calculate_psd_values(x, sampling_rate, n_window, overlap_pr):
            Calculates the power spectral density (PSD) of the input signal and selects 
            frequencies up to 1.5 Hz.
        creat_summary_stat_vec(x, x_ddot=None, x_bins=None, x_dot_bins=None, x_ddot_bins=None, normalize_values=True):
            Creates a summary statistics vector from the input signal, including histogram 
            and PSD features. Optionally normalizes the features.
    """

    def __init__(self, fs, nbins=15, n_ftr_psd=10, n_window=40, overlap_pr=0.75):
        self.fs = fs
        self.nbins = nbins
        self.n_ftr_psd = n_ftr_psd
        self.n_window = n_window
        self.overlap_pr = overlap_pr
    
    def observation_bins(self, x_array):
        # Calculating histogram bin edges based on the maximum absolute value or three times the standard deviation
        # of the input data
        abs_val = np.abs(x_array)
        # histogram by taking a smaller value of either the maximum absolute value in the data or
        # three times of standard deviation of the data, and divided it by 15.
        print(f"max : {np.max(abs_val, axis=1).mean()}\n 3std: {3 * np.std(x_array, axis=1).mean()}")
        hist_band = min(np.max(abs_val, axis=1).mean(), 3 * np.std(x_array, axis=1).mean())
        bin_edges = np.linspace(0, hist_band, self.nbins + 1)
        return bin_edges

    def calculate_psd_values(self, x, sampling_rate, n_window, overlap_pr):
        # Calculate the power spectral density (PSD) using Welch's method
        # with a boxcar window and specified overlap percentage
        f, psd = welch(x, fs=sampling_rate, window='boxcar', nperseg=sampling_rate * n_window,
                       noverlap=sampling_rate * n_window * overlap_pr)
        # Select frequencies up to 1.5 Hz
        max_freq_index = np.argmax(f > 1.5)  # Find index corresponding to 1.5 Hz
        f = f[:max_freq_index]
        psd = psd[:max_freq_index]
        return f[1:], psd[1:]

    def creat_summary_stat_vec(self, 
                               x, 
                               x_ddot=None,
                               x_bins=None,
                               x_dot_bins=None,
                               x_ddot_bins=None,
                               normalize_values=True):
        # Creating summary statistics vector from the input signal
        # including histogram and PSD features
        dt = 1 / self.fs
        x_dot = np.gradient(x) / dt
        
        numerical_x_ddot = np.gradient(x_dot) / dt
        if x_ddot is None:
            x_ddot = numerical_x_ddot
        
        # Calculating histogram summary statistics
        x_hist, _ = np.histogram(np.abs(x), bins=x_bins)
        x_dot_hist, _ = np.histogram(np.abs(x_dot), bins=x_dot_bins)
        x_ddot_hist, _ = np.histogram(np.abs(x_ddot), bins=x_ddot_bins)

        # Calculating frequency summary statistics
        f_x, psd_x = self.calculate_psd_values(x, self.fs, self.n_window, self.overlap_pr)
        f_xdot, psd_xdot = self.calculate_psd_values(x_dot, self.fs, self.n_window, self.overlap_pr)
        f_xddot, psd_xddot = self.calculate_psd_values(numerical_x_ddot, self.fs, self.n_window, self.overlap_pr)

        f_selected=np.logspace(np.log10(f_x[0]), np.log10(f_x[-1]), self.n_ftr_psd, endpoint=True)
        
        psd1 = np.interp(f_selected, f_x, psd_x)
        psd2 = np.interp(f_selected, f_xdot, psd_xdot)
        psd3 = np.interp(f_selected, f_xddot, psd_xddot)
        

        n1, n2 = self.nbins, self.n_ftr_psd
        ss_vec = np.zeros((3 * n1 + 3 * n2, ))
        
        if normalize_values:
            # normalization
            # handle the situation where all elements of the histogram are zeros we use a condition 
            ss1 = x_hist / x_hist.sum() if x_hist.sum() else x_hist
            ss2 = x_dot_hist / x_dot_hist.sum() if x_dot_hist.sum() else x_dot_hist
            ss3 = x_ddot_hist / x_ddot_hist.sum() if x_ddot_hist.sum() else x_ddot_hist
            ss4 = psd1 / psd1.sum()
            ss5 = psd2 / psd2.sum()
            ss6 = psd3 / psd3.sum()

            # concatenate the summary statistics into a single vector
            ss_vec[0:n1] = ss1
            ss_vec[n1:2*n1] = ss2
            ss_vec[2*n1:3*n1] = ss3
            ss_vec[3*n1:3*n1 + n2] = ss4
            ss_vec[3*n1 + n2:3*n1 + 2*n2] = ss5
            ss_vec[3*n1 + 2*n2:3*n1 + 3*n2] = ss6
            
        else:
            # concatenate the summary statistics into a single vector without normalization
            ss_vec[0:n1] = x_hist
            ss_vec[n1:2*n1] = x_dot_hist
            ss_vec[2*n1:3*n1] = x_ddot_hist
            ss_vec[3*n1:3*n1 + n2] = psd1
            ss_vec[3*n1 + n2:3*n1 + 2*n2] = psd2
            ss_vec[3*n1 + 2*n2:3*n1 + 3*n2] = psd3
        
        return ss_vec, f_selected

