import sys
import numpy as np
import scipy.signal as sig
import matplotlib

class Wppfft(object):
    """Computes the fast fourier transform of a signal
    How to use:
    1.) Pass the signal into the class,
    2.) Call wfft() to get the transformed data
    3.) Call freq_axis() and time_axis() to get the freq and time values for each index in the array
    """
    def __init__(self, data, fs, win_size, fft_size, overlap_fac=0.75):
        """Computes a bunch of information that will be used in all of the STFT functions"""
        self.data = np.array(data, dtype=np.float32)
        self.fs = np.int32(fs)
        self.win_size = np.int32(win_size)
        self.fft_size = np.int32(fft_size)
        self.overlap_fac = np.float32(1 - overlap_fac)

        self.hop_size = np.int32(np.floor(self.win_size * self.overlap_fac))
        self.pad_end_size = self.fft_size
        self.total_segments = np.int32(np.ceil(len(self.data) / np.float32(self.hop_size)))
        self.t_max = len(self.data) / np.float32(self.fs)

    def wfft(self, scale='log', ref=1, clip=None):
        """Perform the STFT and return the result"""

        # Todo: changing the overlap factor doens't seem to preserve energy, need to fix this
        window = np.hanning(self.win_size) * self.overlap_fac * 2
        inner_pad = np.zeros((self.fft_size * 2) - self.win_size)

        proc = np.concatenate((self.data, np.zeros(self.pad_end_size)))
        result = np.empty((self.total_segments, self.fft_size), dtype=np.float32)

        for i in range(self.total_segments):
            current_hop = self.hop_size * i
            segment = proc[current_hop:current_hop+self.win_size]
            windowed = segment * window
            padded = np.append(windowed, inner_pad)
            spectrum = np.fft.fft(padded) / self.fft_size
            autopower = np.abs(spectrum * np.conj(spectrum))
            result[i, :] = autopower[:self.fft_size]

        if scale == 'log':
            result = self.dB(result, ref)

        if clip is not None:
            np.clip(result, clip[0], clip[1], out=result)

        return result

    def dB(self, data, ref):
        """Return the dB equivelant of the input data"""
        return 20*np.log10(data / ref)

    def freq_axis(self):
        """Returns a list of frequencies which correspond to the bins in the returned data from stft()"""
        return np.arange(self.fft_size) / np.float32(self.fft_size * 2) * self.fs

    def time_axis(self):
        """Returns a list of times which correspond to the bins in the returned data from stft()"""
        return np.arange(self.total_segments) / np.float32(self.total_segments) * self.t_max


def create_ticks_optimum(axis, num_ticks, resolution, return_errors=False):
    """ Try to divide <num_ticks> ticks evenly across the axis, keeping ticks to the nearest <resolution>"""
    max_val = axis[-1]
    hop_size = max_val / np.float32(num_ticks)

    indicies = []
    ideal_vals = []
    errors = []

    for i in range(num_ticks):
        current_hop = resolution * round(float(i*hop_size)/resolution)
        index = np.abs(axis-current_hop).argmin()

        indicies.append(index)
        ideal_vals.append(current_hop)
        errors.append(np.abs(current_hop - axis[index]))

    if return_errors:
        return indicies, ideal_vals, errors
    else:
        return indicies, ideal_vals


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from nptdms import tdms
    import stft

    tdmspath = './TDMS/vib.tdms'
    tdfile = tdms.TdmsFile(tdmspath)
    data = tdfile.channel_data('Vibration', 'DOC X')
    dataProp = tdfile.group_channels('Vibration')[0].properties
    
    print(dataProp)
    
    DOCSensitivity = float(dataProp['Sensor Sensitivity (mV/EU)'])
    y = data / (DOCSensitivity / 1000)
    fs = 8533.33333333

    win_size = 2**13
    fft_size = win_size
    
    print('y shape:', y.shape)
    
    s =  Wppfft(data=y, fs=fs, win_size=win_size, fft_size=fft_size, overlap_fac=0.5)
    result = s.wfft(scale='log', ref=1.0, clip=[-86, None])

    #result = np.clip(result, -60, 200)

    print('result shape:', result.shape)
    print(result)
    img = plt.imshow(result, origin='lower', cmap='jet', interpolation='none', aspect='auto')
    cbar=plt.colorbar(img)
    tick_res_x = result.shape[1] / 10
    tick_res_y = result.shape[0] / 10

    freqs = s.freq_axis()         
    print('num of freqs:', freqs.shape[0])
    print ('max freq:', freqs[-1])
    x_tick_locations, x_tick_vals = create_ticks_optimum(freqs, num_ticks=15, resolution=50)

    print('tick_locations:', x_tick_locations)
    print('tick_values', x_tick_vals)

    plt.xticks(x_tick_locations, x_tick_vals)

    time = s.time_axis()        
    print (time.shape, time[-1])
    y_tick_locations, y_tick_vals = create_ticks_optimum(time, num_ticks=10, resolution=1)

    plt.yticks(y_tick_locations, y_tick_vals)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Time [s]')
    plt.title('Autopower spectrum of "DOC X"')

    plt.show()
    
    #'''
    next_pow_2 = np.power(2, np.int32(np.ceil(np.log2(y.shape[0]))))
    pad = np.zeros(next_pow_2 - y.shape[0])
    y = np.append(y, pad)
    spectrum = np.fft.fft(y)
    print(spectrum.shape)
    autopower = np.empty(int(spectrum.shape[0]/2), dtype=np.float32)
    autopower[:] = np.abs(spectrum * np.conj(spectrum))[:autopower.shape[0]]
    plt.subplot(1,2,1)
    spectrum_freqs = np.arange(spectrum.shape[0]) / np.float32(next_pow_2) * fs
    plt.plot(spectrum_freqs, spectrum)
    plt.subplot(1,2,2)
    autopower_freqs = np.arange(autopower.shape[0]) / np.float32(next_pow_2) * fs 
    plt.plot(autopower_freqs, autopower)
    print ("Highest frequency is:", np.argmax(autopower) * fs / np.float32(next_pow_2), "Hz")
    plt.show()
    #'''
