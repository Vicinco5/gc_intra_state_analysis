# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Apr 19 11:04:07 2025

# @author: vincentcalia-bogan
# """

#     def _analyze_dataset(self, dataset_name, df):
#         print(f"Processing dataset: {dataset_name}")
#         data_cols = self.get_data_columns(df)
#         tastes = df['taste'].unique().to_list()
#         core_dataset_name = dataset_name.split('_repacked_raw_latent_vectors')[0]

#         dataset_dir = os.path.join(self.tld, dataset_name)
#         os.makedirs(dataset_dir, exist_ok=True)

#         for taste in tastes:
#             taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
#             analysis_dirs = {
#                 'fft': os.path.join(taste_dir, 'fft'),
#                 'peaks': os.path.join(taste_dir, 'peaks'),
#                 'periodogram': os.path.join(taste_dir, 'periodogram'),
#                 'spectrogram': os.path.join(taste_dir, 'spectrogram'),
#                 'lombscargle_periodogram': os.path.join(taste_dir, 'lombscargle_periodogram'),
#                 'individual_pc_lombscargle': os.path.join(taste_dir, 'lombscargle_periodogram', 'individual-pc'),
#                 'individual_pc_periodogram': os.path.join(taste_dir, 'periodogram', 'individual-pc'),
#                 'individual_pc_fft': os.path.join(taste_dir, 'fft', 'individual-pc'),
#                 'individual_pc_peaks': os.path.join(taste_dir, 'peaks', 'individual'),
#                 'full_trial_plots': os.path.join(taste_dir, 'full_trial_plots')
#             }
#             for path in analysis_dirs.values():
#                 os.makedirs(path, exist_ok=True)

#             taste_df = df.filter(pl.col('taste') == taste)
#             trials = taste_df['trial'].unique().to_list()

#             for trial in trials:
#                 trial_df = taste_df.filter(pl.col('trial') == trial)
#                 time = trial_df['time'].to_numpy() / 1000.0

#                 if self.do_fft:
#                     combined_fig_fft, combined_ax_fft = plt.subplots(figsize=(8, 5))
#                 if self.do_periodogram:
#                     combined_fig_periodogram, combined_ax_periodogram = plt.subplots(figsize=(8, 5))
#                 if self.do_peaks:
#                     combined_fig_peaks, combined_ax_peaks = plt.subplots(figsize=(8, 5))
#                 if self.do_lombscargle:
#                     combined_fig_lombscargle, combined_ax_lombscargle = plt.subplots(figsize=(8, 5))

#                 for col in data_cols:
#                     signal = trial_df[col].to_numpy()
#                     trial_prefix = f"trial_{trial}_{col}"

#                     if self.do_fft:
#                         freqs, fft_magnitude = self.compute_fft(signal, time * 1000)
#                         mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
#                         self.save_fft(freqs[mask], fft_magnitude[mask], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
#                         self.plot_individual_fft(freqs, fft_magnitude, [], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
#                         combined_ax_fft.plot(freqs[mask], fft_magnitude[mask], label=col)

#                     if self.do_peaks:
#                         freqs, fft_magnitude = self.compute_fft(signal, time * 1000)
#                         peaks, properties = find_peaks(fft_magnitude, height=self.peak_height_threshold)
#                         self.save_peaks(freqs, fft_magnitude, peaks, properties, analysis_dirs['individual_pc_peaks'], trial_prefix, core_dataset_name)
#                         combined_ax_peaks.plot(freqs[peaks], fft_magnitude[peaks], label=col)

#                     if self.do_periodogram:
#                         self.plot_individual_periodogram(signal, time * 1000, analysis_dirs['individual_pc_periodogram'], trial_prefix, core_dataset_name)
#                         fs = 1000.0 / np.mean(np.diff(time) * 1000)
#                         freqs_comb, pxx_comb = periodogram(signal, fs=fs)
#                         mask_comb = (freqs_comb >= self.min_freq) & (freqs_comb <= self.max_freq)
#                         combined_ax_periodogram.semilogy(freqs_comb[mask_comb], pxx_comb[mask_comb], label=col)

#                     if self.do_lombscargle:
#                         self.plot_lombscargle_periodogram(signal, time, analysis_dirs['individual_pc_lombscargle'], trial_prefix, core_dataset_name)
#                         freqs_lomb = np.linspace(self.min_freq, self.max_freq, 1000)
#                         angular_freqs = 2 * np.pi * freqs_lomb
#                         power = lombscargle(time, signal, angular_freqs)
#                         combined_ax_lombscargle.plot(freqs_lomb, power, label=col)

#                     if self.do_spectrogram:
#                         peaks, _ = find_peaks(signal)
#                         if len(peaks) > 0:
#                             self.plot_spectrogram(signal, time * 1000, analysis_dirs['spectrogram'], trial_prefix, core_dataset_name)

#                 if self.do_fft:
#                     combined_ax_fft.set_title(f"Combined FFT - {core_dataset_name} - Trial {trial}")
#                     combined_ax_fft.set_xlabel("Frequency (Hz)")
#                     combined_ax_fft.set_ylabel("Magnitude")
#                     combined_ax_fft.grid(True)
#                     combined_ax_fft.legend(fontsize='small')
#                     combined_fig_fft.tight_layout()
#                     combined_fig_fft.savefig(os.path.join(analysis_dirs['fft'], f"trial_{trial}_combined_fft.png"))
#                     plt.close(combined_fig_fft)

#                 if self.do_periodogram:
#                     combined_ax_periodogram.set_title(f"Combined Periodogram - {core_dataset_name} - Trial {trial}")
#                     combined_ax_periodogram.set_xlabel("Frequency (Hz)")
#                     combined_ax_periodogram.set_ylabel("Power Spectral Density")
#                     combined_ax_periodogram.grid(True)
#                     combined_ax_periodogram.legend(fontsize='small')
#                     combined_fig_periodogram.tight_layout()
#                     combined_fig_periodogram.savefig(os.path.join(analysis_dirs['periodogram'], f"trial_{trial}_combined_periodogram.png"))
#                     plt.close(combined_fig_periodogram)

#                 if self.do_peaks:
#                     combined_ax_peaks.set_title(f"Combined Peaks - {core_dataset_name} - Trial {trial}")
#                     combined_ax_peaks.set_xlabel("Frequency (Hz)")
#                     combined_ax_peaks.set_ylabel("Magnitude")
#                     combined_ax_peaks.grid(True)
#                     combined_ax_peaks.legend(fontsize='small')
#                     combined_fig_peaks.tight_layout()
#                     combined_fig_peaks.savefig(os.path.join(analysis_dirs['peaks'], f"trial_{trial}_combined_peaks.png"))
#                     plt.close(combined_fig_peaks)

#                 if self.do_lombscargle:
#                     combined_ax_lombscargle.set_title(f"Combined Lomb-Scargle Periodogram - {core_dataset_name} - Trial {trial}")
#                     combined_ax_lombscargle.set_xlabel("Frequency (Hz)")
#                     combined_ax_lombscargle.set_ylabel("Power")
#                     combined_ax_lombscargle.grid(True)
#                     combined_ax_lombscargle.legend(fontsize='small')
#                     combined_fig_lombscargle.tight_layout()
#                     combined_fig_lombscargle.savefig(os.path.join(analysis_dirs['lombscargle_periodogram'], f"trial_{trial}_combined_lombscargle.png"))


# class FrequencyAnalysisPipeline:
#     def __init__(self, tld, standardized_changepoints_dict, modified_tastes, peak_height_threshold=0.05, spectrogram_nperseg=32, min_freq=1, max_freq=np.inf, start_time=1500, end_time=4500,
#                  do_fft=True, do_periodogram=True, do_peaks=True, do_lombscargle=True, do_full_trials=True, do_spectrogram=True, do_individual_plots=True, smart_skip=False, min_amplitude_threshold=0.0):
#         self.tld = tld
#         self.standardized_changepoints_dict = standardized_changepoints_dict
#         self.modified_tastes = modified_tastes
#         self.peak_height_threshold = peak_height_threshold
#         self.spectrogram_nperseg = spectrogram_nperseg
#         self.min_freq = min_freq
#         self.max_freq = max_freq
#         self.start_time = start_time
#         self.end_time = end_time

#         self.do_fft = do_fft
#         self.do_periodogram = do_periodogram
#         self.do_peaks = do_peaks
#         self.do_lombscargle = do_lombscargle
#         self.do_full_trials = do_full_trials
#         self.do_spectrogram = do_spectrogram
#         self.do_individual_plots = do_individual_plots
#         self.smart_skip = smart_skip
#         self.min_amplitude_threshold = min_amplitude_threshold

#         os.makedirs(tld, exist_ok=True)

#         print("Frequency Analysis Pipeline Settings:")
#         print(f"FFT: {self.do_fft}, Periodogram: {self.do_periodogram}, Peaks: {self.do_peaks}, Lomb-Scargle: {self.do_lombscargle}")
#         print(f"Full Trials: {self.do_full_trials}, Spectrogram: {self.do_spectrogram}, Individual Plots: {self.do_individual_plots}, Smart Skip: {self.smart_skip}")
#         print(f"Min Frequency: {self.min_freq} Hz, Max Frequency: {self.max_freq} Hz, Min Amplitude Threshold: {self.min_amplitude_threshold}")

#     @staticmethod
#     def get_data_columns(df: pl.DataFrame):
#         return [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]

#     def run(self, dataset_dict):
#         for dataset_name, df in tqdm(dataset_dict.items(), desc="Processing datasets"):
#             self._analyze_dataset(dataset_name, df)

#     def _analyze_dataset(self, dataset_name, df):
#         print(f"Processing dataset: {dataset_name}")
#         data_cols = self.get_data_columns(df)
#         tastes = df['taste'].unique().to_list()
#         core_dataset_name = dataset_name.split('_repacked_raw_latent_vectors')[0]

#         dataset_dir = os.path.join(self.tld, dataset_name)
#         os.makedirs(dataset_dir, exist_ok=True)

#         for taste_idx, taste in enumerate(tastes):
#             taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
#             analysis_dirs = {
#                 'fft': os.path.join(taste_dir, 'fft'),
#                 'peaks': os.path.join(taste_dir, 'peaks'),
#                 'periodogram': os.path.join(taste_dir, 'periodogram'),
#                 'spectrogram': os.path.join(taste_dir, 'spectrogram'),
#                 'lombscargle_periodogram': os.path.join(taste_dir, 'lombscargle_periodogram'),
#                 'individual_pc_lombscargle': os.path.join(taste_dir, 'lombscargle_periodogram', 'individual-pc'),
#                 'individual_pc_periodogram': os.path.join(taste_dir, 'periodogram', 'individual-pc'),
#                 'individual_pc_fft': os.path.join(taste_dir, 'fft', 'individual-pc'),
#                 'individual_pc_peaks': os.path.join(taste_dir, 'peaks', 'individual'),
#                 'full_trial_plots': os.path.join(taste_dir, 'full_trial_plots')
#             }
#             for path in analysis_dirs.values():
#                 os.makedirs(path, exist_ok=True)

#             # SMART SKIP based on whole directory existence
#             skip_fft = self.smart_skip and os.listdir(analysis_dirs['fft'])
#             skip_periodogram = self.smart_skip and os.listdir(analysis_dirs['periodogram'])
#             skip_peaks = self.smart_skip and os.listdir(analysis_dirs['peaks'])
#             skip_lombscargle = self.smart_skip and os.listdir(analysis_dirs['lombscargle_periodogram'])
#             skip_spectrogram = self.smart_skip and os.listdir(analysis_dirs['spectrogram'])

#             taste_df = df.filter(pl.col('taste') == taste)
#             trials = taste_df['trial'].unique().to_list()

#             for trial in trials:
#                 trial_df = taste_df.filter(pl.col('trial') == trial)
#                 time = trial_df['time'].to_numpy() / 1000.0

#                 if self.do_fft and not skip_fft:
#                     combined_fig_fft, combined_ax_fft = plt.subplots(figsize=(8, 5))
#                 if self.do_periodogram and not skip_periodogram:
#                     combined_fig_periodogram, combined_ax_periodogram = plt.subplots(figsize=(8, 5))
#                 if self.do_peaks and not skip_peaks:
#                     combined_fig_peaks, combined_ax_peaks = plt.subplots(figsize=(8, 5))
#                 if self.do_lombscargle and not skip_lombscargle:
#                     combined_fig_lombscargle, combined_ax_lombscargle = plt.subplots(figsize=(8, 5))

#                 for col in data_cols:
#                     signal = trial_df[col].to_numpy()
#                     trial_prefix = f"trial_{trial}_{col}"

#                     if (self.do_fft or self.do_peaks) and not skip_fft:
#                         freqs, fft_magnitude = self.compute_fft(signal, time * 1000)
#                         mask = (freqs >= self.min_freq) & (freqs <= self.max_freq) & (fft_magnitude >= self.min_amplitude_threshold)

#                     if self.do_fft and not skip_fft:
#                         if self.do_individual_plots:
#                             fft_txt_path = os.path.join(analysis_dirs['individual_pc_fft'], f"{trial_prefix}_fft - {core_dataset_name}.txt")
#                             fft_plot_path = os.path.join(analysis_dirs['individual_pc_fft'], f"{trial_prefix}_fft_peaks.png")
#                             if not (self.smart_skip and os.path.exists(fft_txt_path)):
#                                 self.save_fft(freqs[mask], fft_magnitude[mask], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
#                             if not (self.smart_skip and os.path.exists(fft_plot_path)):
#                                 self.plot_individual_fft(freqs[mask], fft_magnitude[mask], [], analysis_dirs['individual_pc_fft'], trial_prefix, core_dataset_name)
#                         combined_ax_fft.plot(freqs[mask], fft_magnitude[mask], label=col)

#                     if self.do_peaks and not skip_peaks:
#                         peaks, properties = find_peaks(fft_magnitude, height=self.peak_height_threshold)
#                         if self.do_individual_plots:
#                             peaks_txt_path = os.path.join(analysis_dirs['individual_pc_peaks'], f"{trial_prefix}_peaks - {core_dataset_name}.txt")
#                             if not (self.smart_skip and os.path.exists(peaks_txt_path)):
#                                 self.save_peaks(freqs, fft_magnitude, peaks, properties, analysis_dirs['individual_pc_peaks'], trial_prefix, core_dataset_name)
#                         combined_ax_peaks.plot(freqs[peaks], fft_magnitude[peaks], label=col)

#                     if self.do_periodogram and not skip_periodogram:
#                         if self.do_individual_plots:
#                             periodogram_plot_path = os.path.join(analysis_dirs['individual_pc_periodogram'], f"{trial_prefix}_periodogram.png")
#                             if not (self.smart_skip and os.path.exists(periodogram_plot_path)):
#                                 self.plot_individual_periodogram(signal, time * 1000, analysis_dirs['individual_pc_periodogram'], trial_prefix, core_dataset_name)
#                         fs = 1000.0 / np.mean(np.diff(time) * 1000)
#                         freqs_comb, pxx_comb = periodogram(signal, fs=fs)
#                         mask_comb = (freqs_comb >= self.min_freq) & (freqs_comb <= self.max_freq) & (pxx_comb >= self.min_amplitude_threshold)
#                         combined_ax_periodogram.semilogy(freqs_comb[mask_comb], pxx_comb[mask_comb], label=col)

#                     if self.do_lombscargle and not skip_lombscargle:
#                         if self.do_individual_plots:
#                             lombscargle_plot_path = os.path.join(analysis_dirs['individual_pc_lombscargle'], f"{trial_prefix}_lombscargle.png")
#                             if not (self.smart_skip and os.path.exists(lombscargle_plot_path)):
#                                 self.plot_lombscargle_periodogram(signal, time, analysis_dirs['individual_pc_lombscargle'], trial_prefix, core_dataset_name)
#                         freqs_lomb = np.linspace(self.min_freq, self.max_freq, 1000)
#                         angular_freqs = 2 * np.pi * freqs_lomb
#                         power = lombscargle(time, signal, angular_freqs)
#                         mask_lomb = (power >= self.min_amplitude_threshold)
#                         combined_ax_lombscargle.plot(freqs_lomb[mask_lomb], power[mask_lomb], label=col)

#                     if self.do_spectrogram and not skip_spectrogram:
#                         peaks, _ = find_peaks(signal)
#                         if len(peaks) > 0 and self.do_individual_plots:
#                             spectrogram_path = os.path.join(analysis_dirs['spectrogram'], f"{trial_prefix}_spectrogram.png")
#                             if not (self.smart_skip and os.path.exists(spectrogram_path)):
#                                 self.plot_spectrogram(signal, time * 1000, analysis_dirs['spectrogram'], trial_prefix, core_dataset_name)

#                 if self.do_fft and not skip_fft:
#                     combined_ax_fft.set_title(f"Combined FFT - {core_dataset_name} - Taste {taste} - Trial {trial}")
#                     combined_ax_fft.set_xlabel("Frequency (Hz)")
#                     combined_ax_fft.set_ylabel("Magnitude")
#                     combined_ax_fft.legend(fontsize='small')
#                     combined_fig_fft.tight_layout()
#                     combined_fig_fft.savefig(os.path.join(analysis_dirs['fft'], f"trial_{trial}_combined_fft.png"))
#                     plt.close(combined_fig_fft)

#                 if self.do_periodogram and not skip_periodogram:
#                     combined_ax_periodogram.set_title(f"Combined Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}")
#                     combined_ax_periodogram.set_xlabel("Frequency (Hz)")
#                     combined_ax_periodogram.set_ylabel("Power Spectral Density")
#                     combined_ax_periodogram.legend(fontsize='small')
#                     combined_fig_periodogram.tight_layout()
#                     combined_fig_periodogram.savefig(os.path.join(analysis_dirs['periodogram'], f"trial_{trial}_combined_periodogram.png"))
#                     plt.close(combined_fig_periodogram)

#                 if self.do_peaks and not skip_peaks:
#                     combined_ax_peaks.set_title(f"Combined Peaks - {core_dataset_name} - Taste {taste} - Trial {trial}")
#                     combined_ax_peaks.set_xlabel("Frequency (Hz)")
#                     combined_ax_peaks.set_ylabel("Magnitude")
#                     combined_ax_peaks.legend(fontsize='small')
#                     combined_fig_peaks.tight_layout()
#                     combined_fig_peaks.savefig(os.path.join(analysis_dirs['peaks'], f"trial_{trial}_combined_peaks.png"))
#                     plt.close(combined_fig_peaks)

#                 if self.do_lombscargle and not skip_lombscargle:
#                     combined_ax_lombscargle.set_title(f"Combined Lomb-Scargle Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}")
#                     combined_ax_lombscargle.set_xlabel("Frequency (Hz)")
#                     combined_ax_lombscargle.set_ylabel("Power")
#                     combined_ax_lombscargle.legend(fontsize='small')
#                     combined_fig_lombscargle.tight_layout()
#                     combined_fig_lombscargle.savefig(os.path.join(analysis_dirs['lombscargle_periodogram'], f"trial_{trial}_combined_lombscargle.png"))
#                     plt.close(combined_fig_lombscargle)

#             if self.do_full_trials:
#                 self._plot_full_trials_for_taste(core_dataset_name, taste_idx, taste_df, analysis_dirs['full_trial_plots'])


#     def _plot_full_trials_for_taste(self, core_dataset_name, taste_idx, taste_df, output_dir):
#         unique_trials = taste_df['trial'].unique().to_list()
#         pc_columns = self.get_data_columns(taste_df)

#         if core_dataset_name not in self.standardized_changepoints_dict:
#             print(f"Changepoints not found for {core_dataset_name}. Skipping full trial plots.")
#             return

#         changepoints = self.standardized_changepoints_dict[core_dataset_name]

#         for trial_chunk in tqdm(range(0, len(unique_trials), 4), desc=f"Taste {taste_idx} full trial plots"):
#             fig, axs = plt.subplots(4, 1, figsize=(15, 15))
#             fig.suptitle(f"Full Trial RNN Plots for {core_dataset_name} - Taste {taste_idx}", fontsize=16)

#             for i, trial in enumerate(unique_trials[trial_chunk:trial_chunk + 4]):
#                 trial_df = taste_df.filter(pl.col('trial') == trial)
#                 time_values = trial_df['time'].to_numpy()
#                 pc_values = [trial_df[col].to_numpy() for col in pc_columns]

#                 mask = (time_values >= self.start_time) & (time_values <= self.end_time)
#                 time_values = time_values[mask]
#                 pc_values = [pc[mask] for pc in pc_values]

#                 ax = axs[i] if len(unique_trials) > 1 else axs

#                 colors = plt.cm.tab10.colors  # Get default matplotlib color cycle
#                 legend_entries = []

#                 for j, pc_series in enumerate(pc_values):
#                     color = colors[j % len(colors)]
#                     ax.plot(time_values, pc_series, label=f'{pc_columns[j]}', color=color)
#                     peaks, _ = find_peaks(pc_series)
#                     ax.plot(time_values[peaks], pc_series[peaks], 'x', color=color, label=f'{pc_columns[j]}_peak')

#                 trial_changepoints = changepoints[taste_idx][trial]
#                 for changepoint_time in trial_changepoints:
#                     if self.start_time <= changepoint_time <= self.end_time:
#                         ax.axvline(changepoint_time, color='black', linestyle='--', linewidth=2, label='Changepoint')

#                 if self.start_time <= 2000 <= self.end_time:
#                     ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

#                 ax.set_title(f"Trial {trial}")
#                 ax.set_xlabel("Time (ms)")
#                 ax.set_ylabel("Principal Component Value")

#             handles, labels = axs[0].get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))
#             fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize='small', frameon=False)

#             plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#             output_file = os.path.join(output_dir, f"rnn_plots_taste_{taste_idx}_chunk_{trial_chunk // 4 + 1}.png")
#             plt.savefig(output_file)
#             plt.close(fig)


#     def compute_fft(self, signal, time):
#         n = len(signal)
#         dt = np.mean(np.diff(time)) / 1000.0
#         fft_vals = fft(signal)
#         fft_magnitude = np.abs(fft_vals)[:n // 2]
#         freqs = fftfreq(n, dt)[:n // 2]
#         return freqs, fft_magnitude

#     def save_fft(self, freqs, fft_magnitude, base_dir, name, dataset_name):
#         out_path = os.path.join(base_dir, f"{name}_fft - {dataset_name}.txt")
#         np.savetxt(out_path, np.column_stack((freqs, fft_magnitude)), header='Frequency(Hz) FFT_Magnitude', fmt='%0.6f')

#     def save_peaks(self, freqs, fft_magnitude, peaks, properties, base_dir, name, dataset_name):
#         out_path = os.path.join(base_dir, f"{name}_peaks - {dataset_name}.txt")
#         with open(out_path, 'w') as f:
#             for peak_idx in peaks:
#                 if self.min_freq <= freqs[peak_idx] <= self.max_freq:
#                     f.write(f"Freq={freqs[peak_idx]:.3f} Hz, Height={properties['peak_heights'][np.where(peaks == peak_idx)[0][0]]:.3f}\n")

#     def plot_individual_fft(self, freqs, fft_magnitude, peaks, base_dir, name, dataset_name):
#         mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
#         plt.figure(figsize=(8, 4))
#         plt.plot(freqs[mask], fft_magnitude[mask], label='FFT Magnitude')
#         plt.plot(freqs[peaks], fft_magnitude[peaks], label='Peaks')
#         plt.title(f"Individual FFT {name} - {dataset_name}")
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Magnitude")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(base_dir, f"{name}_fft_peaks.png"))
#         plt.close()

#     def plot_individual_periodogram(self, signal, time, base_dir, name, dataset_name):
#         fs = 1000.0 / np.mean(np.diff(time))
#         freqs, pxx = periodogram(signal, fs=fs)
#         mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
#         plt.figure(figsize=(8, 4))
#         plt.semilogy(freqs[mask], pxx[mask])
#         plt.title(f"Individual Periodogram {name} - {dataset_name}")
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Power Spectral Density")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(base_dir, f"{name}_periodogram.png"))
#         plt.close()

#     def plot_lombscargle_periodogram(self, signal, time, base_dir, name, dataset_name):
#         freqs = np.linspace(self.min_freq, self.max_freq, 1000)
#         angular_freqs = 2 * np.pi * freqs
#         power = lombscargle(time, signal, angular_freqs)
#         plt.figure(figsize=(8, 4))
#         plt.plot(freqs, power)
#         plt.title(f"Individual Lomb-Scargle Periodogram {name} - {dataset_name}")
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Power")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(base_dir, f"{name}_lombscargle.png"))
#         plt.close()

#     def plot_spectrogram(self, signal, time, base_dir, name, dataset_name):
#         fs = 1000.0 / np.mean(np.diff(time))
#         nperseg = min(self.spectrogram_nperseg, len(signal))
#         noverlap = nperseg // 2

#         f_spect, t_spect, Sxx = spectrogram(
#             signal,
#             fs=fs,
#             window='hann',
#             nperseg=nperseg,
#             noverlap=noverlap,
#             scaling='density',
#             mode='psd'
#         )

#         t_spect_ms = t_spect * 1000
#         plt.figure(figsize=(8, 5))
#         plt.pcolormesh(t_spect_ms, f_spect, 10 * np.log10(Sxx), shading='auto')
#         plt.colorbar(label="Power (dB)")
#         plt.title("Spectrogram - {dataset_name}")
#         plt.ylabel("Frequency (Hz)")
#         plt.xlabel("Time (ms)")
#         plt.tight_layout()
#         plt.savefig(os.path.join(base_dir, f"{name}_spectrogram.png"))
#         plt.close()
#     # this should be a familiar plot lol


# tld = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/FFT_FREQUENCY_SUITE'
# pipeline = FrequencyAnalysisPipeline(
#     tld=tld,
#     standardized_changepoints_dict= standardized_changepoints_dict,
#     modified_tastes=modified_tastes,
#     peak_height_threshold=0.25,
#     spectrogram_nperseg=16,
#     min_freq=2,        # You can set this!
#     max_freq=15,
#     do_fft=True,
#     do_periodogram=True,
#     do_peaks=True,
#     do_lombscargle=True,
#     do_full_trials=True,
#     do_spectrogram=True, # And this too!
#     do_individual_plots = True,
#     smart_skip = True,
#     min_amplitude_threshold = 0.05
# )

# # dataset_dict = {dataset_name: polars_dataframe}
# pipeline.run(first_derivs)

#                     plt.close(combined_fig_lombscargle)
