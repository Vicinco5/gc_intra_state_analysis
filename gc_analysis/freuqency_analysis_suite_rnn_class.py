#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:30:08 2025

THE vernerable freuqency analysis pipeline

@author: vincentcalia-bogan
"""
# further improvments for this: 
    # a text file in the dataset dir that will tell me what's signficant and where
    # batching plots on figs so I don't have a billion files
    # RUNNING THIS ON AN EPOCH-BY-EPOCH BASIS:
        # only do if the periodograms come back with some level of high-power oscillatory significance 
        # as otherwise this would take practically forever
        # figure out quantificaiton of freuqencies as well 
        


import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy.signal import find_peaks, periodogram, spectrogram, lombscargle
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# definitely going to have to break this up by changepoint...not looking forward (the data rates will be unbelieveable)
class FrequencyAnalysisPipeline:
    """
    Perform multi-modal frequency-domain and time-frequency analyses on latent/PC time series,
    and save plots and data outputs into a structured directory tree.

    Parameters
    ----------
    tld : str
        Top-level output directory. Under this, a subfolder per dataset and taste will be created.
    standardized_changepoints_dict : dict
        Mapping `{ dataset_name → { taste_idx → { trial_idx → [cp_times…] } } }`
        used to overlay changepoint lines on time-series plots.
    modified_tastes : list[int]
        List of taste identifiers to process (e.g. `[0, 1, 2, 3]`).
    peak_height_threshold : float, default=0.05
        Minimum FFT magnitude for peak detection.
    spectrogram_nperseg : int, default=32
        Number of samples per segment when computing spectrograms.
    min_freq : float, default=1
        Lowest frequency (Hz) to include in any spectral plot.
    max_freq : float, default=np.inf
        Highest frequency (Hz) to include in any spectral plot.
    start_time : float, default=1500
        Start time (ms) for full-trial PC time-series plots.
    end_time : float, default=4500
        End time (ms) for full-trial PC time-series plots.
    do_fft : bool, default=True
        Compute and plot FFT (and combined spectra).
    do_periodogram : bool, default=True
        Compute and plot periodograms (and combined PSD).
    do_peaks : bool, default=True
        Detect and plot FFT peaks.
    do_lombscargle : bool, default=True
        Compute and plot Lomb–Scargle periodograms.
    do_full_trials : bool, default=True
        Generate full-trial PC time-series plots with changepoints.
    do_spectrogram : bool, default=True
        Compute and save spectrogram images.
    do_individual_plots : bool, default=True
        Save per-component analyses (FFT data, peaks, periodogram, Lomb–Scargle, spectrogram).
    smart_skip : bool, default=False
        If True, skip any analysis type whose output folder already contains files.
    min_amplitude_threshold : float, default=0.0
        Minimum amplitude/power threshold for masking low-power frequencies.

    Plot and data outputs
    ---------------------
    For each `<dataset_name>` and taste `<t>`:
      • Combined FFT (all PCs) per trial:
        `<tld>/<dataset_name>/taste_<t>/fft/trial_<i>_combined_fft.png`
      • Individual-PC FFT:
        - Data: `<...>/fft/individual-pc/trial_<i>_<col>_fft - <dataset>.txt`
        - Peaks plot: `<...>/fft/individual-pc/trial_<i>_<col>_fft_peaks.png`
      • Combined FFT peaks:
        `<...>/peaks/trial_<i>_combined_peaks.png`
      • Individual-PC peaks (text):
        `<...>/peaks/individual/trial_<i>_<col>_peaks - <dataset>.txt`
      • Combined periodogram:
        `<...>/periodogram/trial_<i>_combined_periodogram.png`
      • Individual-PC periodogram:
        `<...>/periodogram/individual-pc/trial_<i>_<col>_periodogram.png`
      • Combined Lomb–Scargle:
        `<...>/lombscargle_periodogram/trial_<i>_combined_lombscargle.png`
      • Individual-PC Lomb–Scargle:
        `<...>/lombscargle_periodogram/individual-pc/trial_<i>_<col>_lombscargle.png`
      • Spectrogram per PC (if peaks found):
        `<...>/spectrogram/trial_<i>_<col>_spectrogram.png`
      • Full-trial PC time-series plots (chunks of 4 trials):
        `<...>/full_trial_plots/rnn_plots_taste_<t>_chunk_<k>.png`

    Usage
    -----
        pipeline = FrequencyAnalysisPipeline(
            tld="output_dir",
            standardized_changepoints_dict=cp_dict,
            modified_tastes=[0,1,2,3],
            peak_height_threshold=0.1,
            spectrogram_nperseg=64,
            min_freq=2,
            max_freq=100,
            start_time=1500,
            end_time=4500,
            do_fft=True, do_periodogram=True, do_peaks=True,
            do_lombscargle=True, do_full_trials=True,
            do_spectrogram=True, do_individual_plots=True,
            smart_skip=False, min_amplitude_threshold=0.01
        )
        pipeline.run({ "dataset1": df1, "dataset2": df2 })
    """
    def __init__(self, tld, standardized_changepoints_dict, modified_tastes, peak_height_threshold=0.05, spectrogram_nperseg=32, min_freq=1, max_freq=np.inf, start_time=1500, end_time=4500,
                 do_fft=True, do_periodogram=True, do_peaks=True, do_lombscargle=True, do_full_trials=True, do_spectrogram=True, do_individual_plots=True, smart_skip=False, min_amplitude_threshold=0.0):
        self.tld = tld
        self.standardized_changepoints_dict = standardized_changepoints_dict
        self.modified_tastes = modified_tastes
        self.peak_height_threshold = peak_height_threshold
        self.spectrogram_nperseg = spectrogram_nperseg
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.start_time = start_time
        self.end_time = end_time

        self.do_fft = do_fft
        self.do_periodogram = do_periodogram
        self.do_peaks = do_peaks
        self.do_lombscargle = do_lombscargle
        self.do_full_trials = do_full_trials
        self.do_spectrogram = do_spectrogram
        self.do_individual_plots = do_individual_plots
        self.smart_skip = smart_skip
        self.min_amplitude_threshold = min_amplitude_threshold

        os.makedirs(tld, exist_ok=True)

        print("Frequency Analysis Pipeline Settings:")
        print(f"FFT: {self.do_fft}, Periodogram: {self.do_periodogram}, Peaks: {self.do_peaks}, Lomb-Scargle: {self.do_lombscargle}")
        print(f"Full Trials: {self.do_full_trials}, Spectrogram: {self.do_spectrogram}, Individual Plots: {self.do_individual_plots}, Smart Skip: {self.smart_skip}")
        print(f"Min Frequency: {self.min_freq} Hz, Max Frequency: {self.max_freq} Hz, Min Amplitude Threshold: {self.min_amplitude_threshold}")

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        return [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]

    def run(self, dataset_dict):
        for dataset_name, df in tqdm(dataset_dict.items(), desc="Processing datasets"):
            self._analyze_dataset(dataset_name, df)
    def _analyze_dataset(self, dataset_name, df):
        print(f"Processing dataset: {dataset_name}")
        data_cols = self.get_data_columns(df)
        tastes = df['taste'].unique().to_list()
        core_dataset_name = dataset_name.split('_repacked_raw_latent_vectors')[0]
    
        dataset_dir = os.path.join(self.tld, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
    
        for taste_idx, taste in enumerate(tastes):
            taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
            analysis_dirs = {
                'fft': os.path.join(taste_dir, 'fft'),
                'peaks': os.path.join(taste_dir, 'peaks'),
                'periodogram': os.path.join(taste_dir, 'periodogram'),
                'spectrogram': os.path.join(taste_dir, 'spectrogram'),
                'lombscargle_periodogram': os.path.join(taste_dir, 'lombscargle_periodogram'),
                'individual_pc_lombscargle': os.path.join(taste_dir, 'lombscargle_periodogram', 'individual-pc'),
                'individual_pc_periodogram': os.path.join(taste_dir, 'periodogram', 'individual-pc'),
                'individual_pc_fft': os.path.join(taste_dir, 'fft', 'individual-pc'),
                'individual_pc_peaks': os.path.join(taste_dir, 'peaks', 'individual'),
                'full_trial_plots': os.path.join(taste_dir, 'full_trial_plots')
            }
    
            # Smart skip check BEFORE creating directories
            skip_flags = {}
            for analysis_type in ['fft', 'periodogram', 'peaks', 'lombscargle_periodogram', 'spectrogram']:
                path = analysis_dirs[analysis_type]
                skip_flags[analysis_type] = self.smart_skip and os.path.exists(path) and os.listdir(path)
    
            for analysis_type, should_skip in skip_flags.items():
                if should_skip:
                    print(f"[Smart Skip] Skipping {analysis_type.replace('_', ' ').title()} analysis for dataset {dataset_name}, taste {taste}.")
    
            # Create the directories if they do not exist
            for path in analysis_dirs.values():
                os.makedirs(path, exist_ok=True)
    
            taste_df = df.filter(pl.col('taste') == taste)
            trials = taste_df['trial'].unique().to_list()
    
            for trial in trials:
                trial_df = taste_df.filter(pl.col('trial') == trial)
                time = trial_df['time'].to_numpy() / 1000.0
    
                if self.do_fft and not skip_flags['fft']:
                    combined_fig_fft, combined_ax_fft = plt.subplots(figsize=(8, 5))
                if self.do_periodogram and not skip_flags['periodogram']:
                    combined_fig_periodogram, combined_ax_periodogram = plt.subplots(figsize=(8, 5))
                if self.do_peaks and not skip_flags['peaks']:
                    combined_fig_peaks, combined_ax_peaks = plt.subplots(figsize=(8, 5))
                if self.do_lombscargle and not skip_flags['lombscargle_periodogram']:
                    combined_fig_lombscargle, combined_ax_lombscargle = plt.subplots(figsize=(8, 5))
    
                for col in data_cols:
                    signal = trial_df[col].to_numpy()
                    trial_prefix = f"trial_{trial}_{col}"
    
                    if (self.do_fft or self.do_peaks) and not skip_flags['fft']:
                        freqs, fft_magnitude = self.compute_fft(signal, time * 1000)
                        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq) & (fft_magnitude >= self.min_amplitude_threshold)
    
                    if self.do_fft and not skip_flags['fft']:
                        if self.do_individual_plots:
                            fft_txt_path = os.path.join(analysis_dirs['individual_pc_fft'], f"{trial_prefix}_fft - {core_dataset_name}.txt")
                            fft_plot_path = os.path.join(analysis_dirs['individual_pc_fft'], f"{trial_prefix}_fft_peaks.png")
                            if not (self.smart_skip and os.path.exists(fft_txt_path)):
                                self.save_fft(freqs[mask], fft_magnitude[mask], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
                            if not (self.smart_skip and os.path.exists(fft_plot_path)):
                                self.plot_individual_fft(freqs[mask], fft_magnitude[mask], [], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
                        combined_ax_fft.plot(freqs[mask], fft_magnitude[mask], label=col)
    
                    if self.do_peaks and not skip_flags['peaks']:
                        peaks, properties = find_peaks(fft_magnitude, height=self.peak_height_threshold)
                        if self.do_individual_plots:
                            peaks_txt_path = os.path.join(analysis_dirs['individual_pc_peaks'], f"{trial_prefix}_peaks - {core_dataset_name}.txt")
                            if not (self.smart_skip and os.path.exists(peaks_txt_path)):
                                self.save_peaks(freqs, fft_magnitude, peaks, properties, analysis_dirs['individual_pc_peaks'], trial_prefix, core_dataset_name)
                        combined_ax_peaks.plot(freqs[peaks], fft_magnitude[peaks], label=col)
                    
                    if self.do_periodogram and not skip_flags['periodogram']:
                        if self.do_individual_plots:
                            periodogram_plot_path = os.path.join(analysis_dirs['individual_pc_periodogram'], f"{trial_prefix}_periodogram.png")
                            if not (self.smart_skip and os.path.exists(periodogram_plot_path)):
                                self.plot_individual_periodogram(signal, time * 1000, analysis_dirs['individual_pc_periodogram'], trial_prefix, core_dataset_name)
                        fs = 1000.0 / np.mean(np.diff(time) * 1000)
                        freqs_comb, pxx_comb = periodogram(signal, fs=fs)
                        mask_comb = (freqs_comb >= self.min_freq) & (freqs_comb <= self.max_freq)   # << ONLY freq filtering
                        combined_ax_periodogram.semilogy(freqs_comb[mask_comb], pxx_comb[mask_comb], label=col)

                    # if self.do_periodogram and not skip_flags['periodogram']:
                    #     if self.do_individual_plots:
                    #         periodogram_plot_path = os.path.join(analysis_dirs['individual_pc_periodogram'], f"{trial_prefix}_periodogram.png")
                    #         if not (self.smart_skip and os.path.exists(periodogram_plot_path)):
                    #             self.plot_individual_periodogram(signal, time * 1000, analysis_dirs['individual_pc_periodogram'], trial_prefix, core_dataset_name)
                    #     fs = 1000.0 / np.mean(np.diff(time) * 1000)
                    #     freqs_comb, pxx_comb = periodogram(signal, fs=fs)
                    #     mask_comb = (freqs_comb >= self.min_freq) & (freqs_comb <= self.max_freq) & (pxx_comb >= self.min_amplitude_threshold)
                    #     combined_ax_periodogram.semilogy(freqs_comb[mask_comb], pxx_comb[mask_comb], label=col)
    
                    if self.do_lombscargle and not skip_flags['lombscargle_periodogram']:
                        if self.do_individual_plots:
                            lombscargle_plot_path = os.path.join(analysis_dirs['individual_pc_lombscargle'], f"{trial_prefix}_lombscargle.png")
                            if not (self.smart_skip and os.path.exists(lombscargle_plot_path)):
                                self.plot_lombscargle_periodogram(signal, time, analysis_dirs['individual_pc_lombscargle'], trial_prefix, core_dataset_name)
                        freqs_lomb = np.linspace(self.min_freq, self.max_freq, 1000)
                        angular_freqs = 2 * np.pi * freqs_lomb
                        power = lombscargle(time, signal, angular_freqs)
                        mask_lomb = (power >= self.min_amplitude_threshold)
                        combined_ax_lombscargle.plot(freqs_lomb[mask_lomb], power[mask_lomb], label=col)
    
                    if self.do_spectrogram and not skip_flags['spectrogram']:
                        peaks, _ = find_peaks(signal)
                        if len(peaks) > 0 and self.do_individual_plots:
                            spectrogram_path = os.path.join(analysis_dirs['spectrogram'], f"{trial_prefix}_spectrogram.png")
                            if not (self.smart_skip and os.path.exists(spectrogram_path)):
                                self.plot_spectrogram(signal, time * 1000, analysis_dirs['spectrogram'], trial_prefix, core_dataset_name)
    
                if self.do_fft and not skip_flags['fft']:
                    combined_ax_fft.set_title(f"Combined FFT - {core_dataset_name} - Taste {taste} - Trial {trial}")
                    combined_ax_fft.set_xlabel("Frequency (Hz)")
                    combined_ax_fft.set_ylabel("Magnitude")
                    combined_ax_fft.legend(fontsize='small')
                    combined_fig_fft.tight_layout()
                    combined_fig_fft.savefig(os.path.join(analysis_dirs['fft'], f"trial_{trial}_combined_fft.png"))
                    plt.close(combined_fig_fft)
    
                if self.do_periodogram and not skip_flags['periodogram']:
                    combined_ax_periodogram.set_title(f"Combined Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}")
                    combined_ax_periodogram.set_xlabel("Frequency (Hz)")
                    combined_ax_periodogram.set_ylabel("Power Spectral Density")
                    combined_ax_periodogram.legend(fontsize='small')
                    combined_fig_periodogram.tight_layout()
                    combined_fig_periodogram.savefig(os.path.join(analysis_dirs['periodogram'], f"trial_{trial}_combined_periodogram.png"))
                    plt.close(combined_fig_periodogram)
    
                if self.do_peaks and not skip_flags['peaks']:
                    combined_ax_peaks.set_title(f"Combined Peaks - {core_dataset_name} - Taste {taste} - Trial {trial}")
                    combined_ax_peaks.set_xlabel("Frequency (Hz)")
                    combined_ax_peaks.set_ylabel("Magnitude")
                    combined_ax_peaks.legend(fontsize='small')
                    combined_fig_peaks.tight_layout()
                    combined_fig_peaks.savefig(os.path.join(analysis_dirs['peaks'], f"trial_{trial}_combined_peaks.png"))
                    plt.close(combined_fig_peaks)
    
                if self.do_lombscargle and not skip_flags['lombscargle_periodogram']:
                    combined_ax_lombscargle.set_title(f"Combined Lomb-Scargle Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}")
                    combined_ax_lombscargle.set_xlabel("Frequency (Hz)")
                    combined_ax_lombscargle.set_ylabel("Power")
                    combined_ax_lombscargle.legend(fontsize='small')
                    combined_fig_lombscargle.tight_layout()
                    combined_fig_lombscargle.savefig(os.path.join(analysis_dirs['lombscargle_periodogram'], f"trial_{trial}_combined_lombscargle.png"))
                    plt.close(combined_fig_lombscargle)
    
            if self.do_full_trials:
                self._plot_full_trials_for_taste(core_dataset_name, taste_idx, taste_df, analysis_dirs['full_trial_plots'])



    def _plot_full_trials_for_taste(self, core_dataset_name, taste_idx, taste_df, output_dir):
        unique_trials = taste_df['trial'].unique().to_list()
        pc_columns = self.get_data_columns(taste_df)

        if core_dataset_name not in self.standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping full trial plots.")
            return

        changepoints = self.standardized_changepoints_dict[core_dataset_name]

        for trial_chunk in tqdm(range(0, len(unique_trials), 4), desc=f"Taste {taste_idx} full trial plots"):
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))
            fig.suptitle(f"Full Trial RNN Plots for {core_dataset_name} - Taste {taste_idx}", fontsize=16)

            for i, trial in enumerate(unique_trials[trial_chunk:trial_chunk + 4]):
                trial_df = taste_df.filter(pl.col('trial') == trial)
                time_values = trial_df['time'].to_numpy()
                pc_values = [trial_df[col].to_numpy() for col in pc_columns]

                mask = (time_values >= self.start_time) & (time_values <= self.end_time)
                time_values = time_values[mask]
                pc_values = [pc[mask] for pc in pc_values]

                ax = axs[i] if len(unique_trials) > 1 else axs

                colors = plt.cm.tab10.colors  # Get default matplotlib color cycle
                legend_entries = []

                for j, pc_series in enumerate(pc_values):
                    color = colors[j % len(colors)]
                    ax.plot(time_values, pc_series, label=f'{pc_columns[j]}', color=color)
                    peaks, _ = find_peaks(pc_series)
                    ax.plot(time_values[peaks], pc_series[peaks], 'x', color=color, label=f'{pc_columns[j]}_peak')

                trial_changepoints = changepoints[taste_idx][trial]
                for changepoint_time in trial_changepoints:
                    if self.start_time <= changepoint_time <= self.end_time:
                        ax.axvline(changepoint_time, color='black', linestyle='--', linewidth=2, label='Changepoint')

                if self.start_time <= 2000 <= self.end_time:
                    ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

                ax.set_title(f"Trial {trial}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Principal Component Value")

            handles, labels = axs[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize='small', frameon=False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            output_file = os.path.join(output_dir, f"rnn_plots_taste_{taste_idx}_chunk_{trial_chunk // 4 + 1}.png")
            plt.savefig(output_file)
            plt.close(fig)


    def compute_fft(self, signal, time):
        n = len(signal)
        dt = np.mean(np.diff(time)) / 1000.0
        fft_vals = fft(signal)
        fft_magnitude = np.abs(fft_vals)[:n // 2]
        freqs = fftfreq(n, dt)[:n // 2]
        return freqs, fft_magnitude

    def save_fft(self, freqs, fft_magnitude, base_dir, name, dataset_name):
        out_path = os.path.join(base_dir, f"{name}_fft - {dataset_name}.txt")
        np.savetxt(out_path, np.column_stack((freqs, fft_magnitude)), header='Frequency(Hz) FFT_Magnitude', fmt='%0.6f')

    def save_peaks(self, freqs, fft_magnitude, peaks, properties, base_dir, name, dataset_name):
        out_path = os.path.join(base_dir, f"{name}_peaks - {dataset_name}.txt")
        with open(out_path, 'w') as f:
            for peak_idx in peaks:
                if self.min_freq <= freqs[peak_idx] <= self.max_freq:
                    f.write(f"Freq={freqs[peak_idx]:.3f} Hz, Height={properties['peak_heights'][np.where(peaks == peak_idx)[0][0]]:.3f}\n")

    def plot_individual_fft(self, freqs, fft_magnitude, peaks, base_dir, name, dataset_name):
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[mask], fft_magnitude[mask], label='FFT Magnitude')
        plt.plot(freqs[peaks], fft_magnitude[peaks], label='Peaks')
        plt.title(f"Individual FFT {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_fft_peaks.png"))
        plt.close()

    def plot_individual_periodogram(self, signal, time, base_dir, name, dataset_name):
        fs = 1000.0 / np.mean(np.diff(time))
        freqs, pxx = periodogram(signal, fs=fs)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        plt.figure(figsize=(8, 4))
        plt.semilogy(freqs[mask], pxx[mask])
        plt.title(f"Individual Periodogram {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_periodogram.png"))
        plt.close()

    def plot_lombscargle_periodogram(self, signal, time, base_dir, name, dataset_name):
        freqs = np.linspace(self.min_freq, self.max_freq, 1000)
        angular_freqs = 2 * np.pi * freqs
        power = lombscargle(time, signal, angular_freqs)
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, power)
        plt.title(f"Individual Lomb-Scargle Periodogram {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_lombscargle.png"))
        plt.close()

    def plot_spectrogram(self, signal, time, base_dir, name, dataset_name):
        fs = 1000.0 / np.mean(np.diff(time))
        nperseg = min(self.spectrogram_nperseg, len(signal))
        noverlap = nperseg // 2

        f_spect, t_spect, Sxx = spectrogram(
            signal,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )

        t_spect_ms = t_spect * 1000
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(t_spect_ms, f_spect, 10 * np.log10(Sxx), shading='auto')
        plt.colorbar(label="Power (dB)")
        plt.title("Spectrogram - {dataset_name}")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_spectrogram.png"))
        plt.close()
    # this should be a familiar plot lol    
    
    
    
    
# older stuff here: 
# because we have space spect in this household 
def safe_spectrogram(data, fs=1000.0):
    """
    Computes a spectrogram on 'data', automatically handling the case where
    len(data) < default nperseg (256). Also ensures noverlap < nperseg.
    """
    # Pick your base nperseg (e.g., 256). If data is shorter, reduce nperseg accordingly:
    nperseg = min(16, len(data)) # this is very changable; calibrate to enviornent 
    
    # Choose noverlap so that it’s strictly less than nperseg
    # e.g. 50% overlap -> noverlap = nperseg // 2
    noverlap = nperseg // 3  # or int(0.75 * nperseg), etc. # this is also very changable and should be calibrated
    
    # If the data is extremely short or empty, handle gracefully:
    if len(data) < 2:
        return None, None, None  # or handle however you wish

    f_spect, t_spect, Sxx = spectrogram(
        data,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    return f_spect, t_spect, Sxx

spect_output = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/SPECTROGRAM'
def plot_spectrograms_for_latents(epoch_dataframes_dict, standardized_changepoints_dict, output_dir):
    """
    For each dataset's DataFrame -> each taste -> each latent dimension,
    chunk trials in groups of 5. Each figure has up to 5 subplots 
    (one trial per subplot). Overlays changepoints in ms. 
    The x-axis is in ms (no longer seconds), and the frequency axis is 0–25 Hz.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"No changepoints found for '{core_dataset_name}'. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        dataset_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        unique_tastes = df['taste'].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for this taste
            taste_df = df.filter(pl.col('taste') == taste)
            unique_trials = taste_df['trial'].unique().to_list()

            n_dims = 8  # Adjust if you have more or fewer dims
            for dim_idx in range(n_dims):
                dim_col = f'latent_dim_{dim_idx}'
                if dim_col not in taste_df.columns:
                    continue  # skip if this dimension doesn't exist

                # Break trials into chunks of 5 (so each figure has up to 5 subplots)
                trial_chunks = [unique_trials[i : i + 5] for i in range(0, len(unique_trials), 5)]

                for chunk_i, chunk_trials in enumerate(trial_chunks, start=1):
                    fig = plt.figure(figsize=(12, 16))
                    fig.suptitle(
                        f"Spectrograms (ms)\n"
                        f"Dataset={df_name}, Taste={taste}, Dim={dim_idx}, "
                        f"Trials {chunk_trials}",
                        fontsize=14
                    )

                    for subplot_idx, trial in enumerate(chunk_trials):
                        ax = fig.add_subplot(5, 1, subplot_idx + 1)
                        trial_df = taste_df.filter(pl.col('trial') == trial)

                        # Grab the times (ms)
                        trial_times = trial_df['time'].to_numpy()
                        if len(trial_times) < 2:
                            continue  # too short or empty trial

                        min_time_ms = trial_times[0]
                        max_time_ms = trial_times[-1]
                        data = trial_df[dim_col].to_numpy()

                        # Check how many samples we have
                        # Debug print example:
                        # print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}, "
                        #       f"time_min={min_time_ms}, time_max={max_time_ms}")

                        # Compute spectrogram with fs=1000
                        f_spect, t_spect, Sxx = safe_spectrogram(data, fs=1000.0)
                        print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}")
                        if f_spect is None:
                            # Data too short
                            continue

                        # Convert the spectrogram's time axis from seconds to ms
                        t_spect_ms = t_spect * 1000.0

                        # Shift so that time=0 in the spectrogram lines up with the trial's min_time_ms
                        t_spect_total = t_spect_ms + min_time_ms

                        # Plot the spectrogram in dB
                        cmesh = ax.pcolormesh(
                            t_spect_total,
                            f_spect,
                            10 * np.log10(Sxx),
                            shading='auto'
                        )

                        # Plot changepoints (they're presumably in ms already)
                        trial_changepoints = []
                        if taste_idx < len(changepoints) and trial in changepoints[taste_idx]:
                            trial_changepoints = changepoints[taste_idx][trial]

                        # Overlay each changepoint in ms
                        for cp_ms in trial_changepoints:
                            ax.axvline(cp_ms, linestyle='--', color='green', linewidth=2)

                        # Frequency limit 0–25 Hz
                        ax.set_ylim(0, 15)

                        # X-axis in ms
                        ax.set_xlim(min_time_ms, max_time_ms)

                        ax.set_title(f"Trial {trial}", fontsize=12)
                        if subplot_idx == len(chunk_trials) - 1:
                            ax.set_xlabel("Time (ms)")
                        ax.set_ylabel("Freq (Hz)")

                    # Adjust layout
                    fig.tight_layout(rect=[0, 0, 1, 0.94])

                    # Single colorbar for the figure
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(cmesh, cax=cbar_ax, label="Power (dB)")

                    # Save figure
                    output_filename = (
                        f"spect_{core_dataset_name}_taste_{taste}_dim_{dim_idx}_chunk_{chunk_i}.png"
                    )
                    output_path = os.path.join(dataset_output_dir, output_filename)
                    plt.savefig(output_path, dpi=150)
                    plt.close(fig)

                    print(f"Saved spectrogram figure: {output_path}")

    print("Finished generating all spectrogram plots.")

# trying a new thing out.... idk if this will work 

# def uniform_spectrogram(data, start_ms, end_ms):
#     """
#     1) Uniformly space 'data' between start_ms and end_ms.
#     2) Compute an effective fs so that we treat the entire data 
#        as one uniform chunk from 0 -> (end_ms - start_ms) in ms.
#     3) Return (f_spect, t_spect, Sxx), with t_spect in seconds 
#        from 0 -> total_duration_s. We'll still shift it afterward 
#        to [start_ms, end_ms].
#     """
#     # If data is extremely short, abort
#     if len(data) < 2:
#         return None, None, None
    
#     range_ms = end_ms - start_ms
#     if range_ms < 1:
#         # No real range -> can't do a meaningful spectrogram
#         return None, None, None

#     # Create a uniform time array for these samples: 0..range_ms
#     # (We won't actually pass this time array to spectrogram, but 
#     #  we need to figure out an effective fs that matches this layout.)
#     n_samples = len(data)
    
#     # Effective sample frequency (in Hz) so that n_samples spans 'range_ms' in milliseconds
#     # total duration in seconds = range_ms / 1000
#     # n_samples - 1 intervals in that duration:
#     fs_eff = (n_samples - 1) / (range_ms / 1000.0)  

#     # Now run spectrogram with that effective fs
#     # We'll do smaller nperseg if data is short
#     nperseg = min(48, n_samples)  
#     noverlap = nperseg // 2
#     f_spect, t_spect, Sxx = spectrogram(
#         data,
#         fs=fs_eff,
#         window='hann',
#         nperseg=nperseg,
#         noverlap=noverlap,
#         scaling='density',
#         mode='psd'
#     )
#     return f_spect, t_spect, Sxx, fs_eff


# def plot_spectrograms_for_latents(epoch_dataframes_dict, standardized_changepoints_dict, output_dir):
#     """
#     For each dataset's DataFrame -> each taste -> each latent dimension,
#     chunk trials in groups of 5. Each figure has up to 5 subplots 
#     (one trial per subplot). Overlays changepoints in ms, converted to seconds.

#     Frequency axis (y-axis) is truncated at 25 Hz; time axis goes from the actual 
#     earliest to latest ms of each trial (converted to s).
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for df_name, df in epoch_dataframes_dict.items():
#         # Extract base dataset name
#         core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]

#         # Skip if no changepoints
#         if core_dataset_name not in standardized_changepoints_dict:
#             print(f"No changepoints found for '{core_dataset_name}'. Skipping...")
#             continue

#         changepoints = standardized_changepoints_dict[core_dataset_name]

#         # Create output folder for this dataset
#         dataset_output_dir = os.path.join(output_dir, df_name)
#         os.makedirs(dataset_output_dir, exist_ok=True)

#         # Unique tastes
#         unique_tastes = df['taste'].unique().to_list()

#         for taste_idx, taste in enumerate(unique_tastes):
#             # Filter for this taste
#             taste_df = df.filter(pl.col('taste') == taste)
#             unique_trials = taste_df['trial'].unique().to_list()

#             # We'll create one figure per dimension for every chunk of 5 trials
#             n_dims = 8  # or however many latent dims you have

#             for dim_idx in range(n_dims):
#                 dim_col = f'latent_dim_{dim_idx}'
#                 if dim_col not in taste_df.columns:
#                     # Skip if this dimension doesn't exist in the data
#                     continue

#                 # Break the trials into chunks of 5
#                 trial_chunks = [unique_trials[i : i + 5] for i in range(0, len(unique_trials), 5)]

#                 for chunk_i, chunk_trials in enumerate(trial_chunks, start=1):
#                     # Create a figure that will have up to 5 subplots (one for each trial)
#                     fig = plt.figure(figsize=(10, 16))
#                     fig.suptitle(
#                         f"Spectrograms\n"
#                         f"Dataset={df_name}, Taste={taste}, Dim={dim_idx}, "
#                         f"Trials {chunk_trials}",
#                         fontsize=14
#                     )

#                     # For each trial in this chunk
#                     for subplot_idx, trial in enumerate(chunk_trials):
#                         # Subplot for this trial
#                         ax = fig.add_subplot(5, 1, subplot_idx + 1)

#                         # Filter data for this trial
#                         trial_df = taste_df.filter(pl.col('trial') == trial)

#                         # Grab this trial's array of times (ms)
#                         trial_times = trial_df['time'].to_numpy()
#                         if trial_times.size == 0:
#                             continue

#                         min_time_ms = trial_times[0]
#                         max_time_ms = trial_times[-1]

#                         # Grab the latent dimension data
#                         data = trial_df[dim_col].to_numpy()

#                         # Compute spectrogram
#                         f_spect, t_spect, Sxx = safe_spectrogram(data, fs=1000.0)
#                         print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}")
#                         if f_spect is None:
#                             # too short or empty
#                             print("Skipping spectrogram: data is too short.")
#                             continue

#                         # Shift spectrogram's time axis so that 0 corresponds to min_time_ms
#                         # i.e. each spectrogram time bin = t_spect + (start_time_in_s)
#                         t_spect_shifted = t_spect + (min_time_ms / 1000.0)

#                         # Plot the spectrogram in dB
#                         cmesh = ax.pcolormesh(
#                             t_spect_shifted, 
#                             f_spect, 
#                             10 * np.log10(Sxx), 
#                             shading='auto'
#                         )

#                         # Overlay changepoints (ms -> s)
#                         trial_changepoints = []
#                         if taste_idx < len(changepoints) and trial in changepoints[taste_idx]:
#                             trial_changepoints = changepoints[taste_idx][trial]

#                         for cp_time_ms in trial_changepoints:
#                             ax.axvline(cp_time_ms / 1000.0, linestyle='--', color='white', linewidth=2)

#                         # Limit freq axis from 0 to 25 Hz -- chagne this as needed
#                         #ax.set_ylim(0, 25)

#                         # Set time axis from min_time_ms -> max_time_ms (in s)
#                         ax.set_xlim(min_time_ms / 1000.0, max_time_ms / 1000.0)

#                         # Add subplot title
#                         ax.set_title(f"Trial {trial}", fontsize=12)

#                         # Add labels only for the bottom subplot or so
#                         if subplot_idx == len(chunk_trials) - 1:
#                             ax.set_xlabel("Time (s)")
#                         ax.set_ylabel("Freq (Hz)")

#                     # Adjust layout, add colorbar
#                     fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
#                     # Add a single colorbar for the figure, right side:
#                     # (to do that gracefully, can do fig.colorbar(..., ax=axes, location='right')
#                     # but since we have subplots, a quick approach is to just add it to the last subplot)
#                     #
#                     # Alternatively, you could individually create colorbars in each subplot if you prefer.
#                     # For a single colorbar across all subplots:
#                     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
#                     fig.colorbar(cmesh, cax=cbar_ax, label="Power (dB)")

#                     # Save figure
#                     output_filename = (
#                         f"spect_{core_dataset_name}_taste_{taste}_dim_{dim_idx}_chunk_{chunk_i}.png"
#                     )
#                     output_path = os.path.join(dataset_output_dir, output_filename)
#                     plt.savefig(output_path, dpi=150)
#                     plt.close(fig)

#                     print(f"Saved spectrogram figure: {output_path}")

#     print("Finished generating all spectrogram plots.")

