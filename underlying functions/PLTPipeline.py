import os
import re
from typing import Dict, List, Tuple, Any

import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class PlottingPipeline:
    """
    Pipeline for generating plots from Polars DataFrame outputs, using external changepoints.

    Parameters
    ----------
    standardized_changepoints_dict : dict
        Mapping core dataset names to nested changepoint arrays.
    output_dir : str, optional
        Directory to save generated plots.
    start_time_ms : int, default=1500
        Lower bound of the time window (ms) for plotting.
    end_time_ms : int, default=4500
        Upper bound of the time window (ms) for plotting.
    stim_time_ms : int, default=2000
        Time of stimulus delivery (ms) to mark on plots.
    plot_configs : dict
        Additional configuration for plot styles or parameters.
    """
    # Columns always present in DataFrame
    META_COLS = ["taste", "trial", "changepoint", "time"]

    def __init__(
        self,
        standardized_changepoints_dict: Dict[str, Any],
        output_dir: str = None,
        start_time_ms: int = 1500,
        end_time_ms: int = 4500,
        stim_time_ms: int = 2000,
        **plot_configs
    ):
        self.standardized_changepoints_dict = standardized_changepoints_dict
        self.output_dir = output_dir
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.stim_time_ms = stim_time_ms
        self.plot_configs = plot_configs
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize default style parameters
        self._init_plot_style()
    
    def run(self, dataset_dict: Dict[str, pl.DataFrame]):
        """
        Iterate over each dataset key and DataFrame, using external changepoints.
        Detects whether dataset is 'warped' or 'unwarped' from its analysis_type.
        """
        for dataset_name, df in tqdm(dataset_dict.items(), desc='Processing datasets'):
            core_name, analysis_type = self.parse_dataset_name(dataset_name)
            cps = self.standardized_changepoints_dict.get(core_name)
            if cps is None:
                print(f'No changepoints for {core_name}, skipping plots.')
                continue
            data_cols = self.get_data_columns(df)
            meta_cols = self.get_meta_columns(df)
            # Determine warp flags
            kind = analysis_type.lower()
            is_unwarped = kind.endswith('_unwarped')
            is_warped   = kind.endswith('_warped')
            # Call plotting for this dataset; flags reset per-call
            self._analyze_dataset(
                core_name,
                analysis_type,
                df,
                data_cols,
                meta_cols,
                cps,
                ignore_start=False,
                ignore_end=False,
                is_warped=is_warped,
                is_unwarped=is_unwarped
            )

    def _analyze_dataset(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        changepoints: Any,
        ignore_start: bool = False,
        ignore_end: bool = False,
        is_warped: bool = False,
        is_unwarped: bool = False,
    ):
        """
        Per-dataset plotting logic.
        """

        # If this is warped data, call non-stationarity vs similarity plot and return
        if is_warped:
            self._plot_nonstationarity_vs_similarity(core_name, analysis_type, df,
                                                     data_cols, meta_cols,
                                                     changepoints,
                                                     ignore_start, ignore_end,
                                                     True, False)
            return
        if is_unwarped:
            self._plot_full_by_trial(
                core_name,
                analysis_type,
                df,
                data_cols,
                meta_cols,
                ignore_start,
                ignore_end,
                is_warped,
                is_unwarped,
            )
            return
    # --- Metrics calculation methods --- discrete logic steps that shouldn't run unless called
    def compute_nonstationarity(self, df: pl.DataFrame, unit: str, seed: int = 42) -> float:
        """
        Normalize each trial vector, shuffle, and measure mean abs deviation.
        """
        rng = np.random.RandomState(seed)
        values = []
        trials = df.select('trial').unique().to_series().to_list()
        for tr in trials:
            sub = df.filter(pl.col('trial') == tr)
            vec = sub.select(unit).to_series().to_numpy().astype(float)
            if vec.size == 0:
                continue
            if vec.max() > vec.min():
                vec = (vec - vec.min()) / (vec.max() - vec.min())
            shuf = rng.permutation(vec)
            values.append(np.mean(np.abs(vec - shuf)))
        return float(np.mean(values)) if values else 0.0

    def compute_intertrial_similarity(self, df: pl.DataFrame, unit: str) -> float:
        """
        Compute mean pairwise cosine similarity across trial vectors.
        """
        trials = df.select('trial').unique().to_series().to_list()
        vecs = []
        for tr in trials:
            sub = df.filter(pl.col('trial') == tr)
            vec = sub.select(unit).to_series().to_numpy().astype(float)
            if vec.size:
                vecs.append(vec)
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                v1, v2 = vecs[i], vecs[j]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 and n2:
                    sims.append(np.dot(v1, v2) / (n1 * n2))
        return float(np.mean(sims)) if sims else 0.0
    
    # ---- Plotting methods ----- # 
    def _plot_nonstationarity_vs_similarity(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        changepoints: Any,
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        if not is_warped:
            print(f"Skipping metric plot on unwarped data {core_name}{analysis_type}")
            return
        tastes = sorted(df.select('taste').unique().to_series().to_list())
        fig, axes = plt.subplots(4, len(tastes), figsize=(12, 12), dpi=self.dpi, tight_layout=True)
        fig_name = 'Non-Stationarity vs Inter-Trial Similarity'
        fig.suptitle(fig_name, y = self.suptitle_y, **self.title_fontdict)
        title = f"{core_name}{analysis_type}"
        fig.text(0.5, self.subtitle_y, title, ha='center', **self.subtitle_fontdict)
        # fig.subplots_adjust(top = self.subplot_top)
        for i in range(4):
            for j, taste in enumerate(tastes):
                ax = axes[i, j]
                sub = df.filter((pl.col('changepoint') == i) & (pl.col('taste') == taste))
                x_vals = [self.compute_nonstationarity(sub, unit) for unit in data_cols]
                y_vals = [self.compute_intertrial_similarity(sub, unit) for unit in data_cols]
                labels = [f"{u[0]}_{u.split('_')[-1]}" for u in data_cols]
                ax.scatter(x_vals, y_vals)
                for xi, yi, lbl in zip(x_vals, y_vals, labels): ax.annotate(lbl, (xi, yi))
                ax.set_xlabel('Non-stationarity', **self.label_fontdict)
                ax.set_ylabel('Similarity', **self.label_fontdict)
                ax.set_title(f"State {i}, Taste {taste}", **self.subtitle_fontdict)
        if self.output_dir:
            self._save_figure(fig, fig_name, fig_name, core_name, analysis_type, tastes)
        else:
            plt.show()
    
    
    def _plot_full_by_trial(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        # Only handle un-warped data
        if not is_unwarped:
            print(f"Skipping full-by-trial plot on warped data {core_name}{analysis_type}")
            return

        # Initialize plot style
        self._init_plot_style()

        # Constant figure name
        fig_name = 'full_trial_plots'

        # Unique sorted tastes
        tastes = sorted(df.select('taste').unique().to_series().to_list())
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col('taste') == taste)
            trials = sorted(taste_df.select('trial').unique().to_series().to_list()) # ensure sequential?

            # 4 trials per figure
            for chunk_start in range(0, len(trials), 4):
                fig, axes = plt.subplots(
                    4,
                    1,
                    figsize=(15, 15),
                    dpi=self.dpi,
                    tight_layout=True
                )
                # Main titles
                fig.suptitle(
                    fig_name,
                    y=self.suptitle_y,
                    **self.title_fontdict
                )
                fig.text(
                    0.5,
                    self.subtitle_y,
                    f"{core_name}{analysis_type}",
                    ha='center',
                    **self.subtitle_fontdict
                )

                # Plot each trial in the chunk
                for i, trial in enumerate(trials[chunk_start:chunk_start + 4]):
                    ax = axes[i]
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time = trial_df['time'].to_numpy()

                    # Build time mask
                    mask = np.ones_like(time, dtype=bool)
                    if not ignore_start:
                        mask &= time >= self.start_time_ms
                    if not ignore_end:
                        mask &= time <= self.end_time_ms

                    # Plot data columns
                    for col in data_cols:
                        series = trial_df[col].to_numpy()[mask]
                        ax.plot(time[mask], series, label=col)

                    # Plot changepoints
                    lower = self.start_time_ms if not ignore_start else time.min()
                    upper = self.end_time_ms if not ignore_end else time.max()
                    for cp in self.get_trial_changepoints(
                        self.standardized_changepoints_dict[core_name],
                        taste_idx,
                        trial
                    ):
                        if lower <= cp <= upper:
                            ax.axvline(cp, **self.line_kwargs['changepoint'])

                    # Stimulus delivery line
                    if lower <= self.stim_time_ms <= upper:
                        ax.axvline(self.stim_time_ms, **self.line_kwargs['stim'])

                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Value (see analysis type)")

                # Shared legend below subplots
                handles, labels = axes[-1].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc='lower center',
                    ncol=5,
                    fontsize='small',
                    frameon=False
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                # Save figure with chunk identifier
                chunk_id = chunk_start // 4 + 1
                title = f"{fig_name}_chunk{chunk_id}"
                self._save_figure(
                    fig,
                    fig_name,
                    title,
                    core_name,
                    analysis_type,
                    [taste]
                )
                
    # -- Auxilliary and helper methods --- # 
    
    def _init_plot_style(self):
        """
        Set up default matplotlib styling parameters for axes labels, ticks, lines, titles, colormaps, and more.
        Can be overridden per-plot by specifying custom parameters in the plotting methods.
        """
        # Uncomment and set fig size per-plot as needed:
        # self.figsize = (8, 6)  # Set on case-by-case basis
        self.dpi = 300

        # Axis label fonts
        self.label_fontdict = {
            'fontsize': 12,
            'family': 'sans-serif',
            'weight': 'bold'
        }
        # Axis tick fonts
        self.tick_fontdict = {
            'fontsize': 10,
            'family': 'sans-serif',
            'weight': 'normal'
        }
        # Default line width and marker size
        self.default_linewidth = 1
        self.default_markersize = 3

        # Grid default off, but style defined
        plt.rcParams['axes.grid'] = False
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.3

        # Legend styling
        self.legend_kwargs = {
            'fontsize': 10,
            'frameon': False,
            'loc': 'best'
        }

        # Spine visibility defaults
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
       #  plt.rcParams['tight_layout'] = True # enforce tight layout by default -- apparently not a valid default??

        # Colormaps
        self.colormaps = {
            'heatmap': 'viridis',
            'line': 'tab10'
        }
        # Color cycle for lines
        cmap = plt.get_cmap(self.colormaps['line'])
        colors = [cmap(i) for i in range(cmap.N)]
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors)

        # Line styles for changepoints and stimulus
        self.line_kwargs = {
            'changepoint': {
                'color': 'black',
                'linestyle': '--',
                'linewidth': 2,
                'label': 'Changepoint'
            },
            'stim': {
                'color': 'black',
                'linestyle': ':',
                'linewidth': 2,
                'label': 'Stimulus Delivery'
            }
        }

        # Title and subtitle fonts
        self.title_fontdict = {
            'fontsize': 14,
            'weight': 'bold'
        }
        self.subtitle_fontdict = {
            'fontsize': 12,
            'weight': 'normal'
        }
        #––– Title / subtitle spacing tweaks –––#
        # How high up the figure the suptitle sits (default ≈0.98)
        self.suptitle_y = 0.998 
        # How high up the subtitle sits (must be < suptitle_y)
        self.subtitle_y = 0.972
        # How far down the axes themselves are pushed (leaves more room at top)
        self.subplot_top = 0.88 # not used anymore -- clashes 
        # You can also set a global default for all figures if you like:
        # plt.rcParams['figure.subplot.top'] = self.subplot_top
    
    def _save_figure(
        self,
        fig: plt.Figure,
        dir_title: str,
        title: str, 
        core_name: str,
        analysis_type: str,
        tastes: List[Any]
    ):
        """
        General method to save a figure under output_dir/title_folder/core_name/taste/analysis_type/{PNG,SVG}.
        """
        # Normalize title for folder name
        folder_name = re.sub(r'[^0-9A-Za-z]+', '_', dir_title).lower().strip('_')
        title_dir = os.path.join(self.output_dir, folder_name)
        dataset_dir = os.path.join(title_dir, core_name)
        plot_type = analysis_type.lstrip('_')
        for taste in tastes:
            taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
            plot_dir = os.path.join(taste_dir, plot_type)
            png_dir = os.path.join(plot_dir, 'PNG')
            svg_dir = os.path.join(plot_dir, 'SVG')
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)
            png_path = os.path.join(png_dir, f"{title}_taste{taste}.png")
            svg_path = os.path.join(svg_dir, f"{title}_taste{taste}.svg")
            fig.savefig(png_path, dpi=self.dpi)
            fig.savefig(svg_path, dpi=self.dpi)
        plt.close(fig)
    
    def get_trial_changepoints(self, changepoints, taste_idx, trial):
        """
        Retrieve the list of changepoint times for a given taste index and trial.
        """
        try:
            return changepoints[taste_idx][trial]
        except (KeyError, IndexError):
            return []

    # ---- static utils --- # 
    @staticmethod
    def get_meta_columns(df: pl.DataFrame) -> List[str]:
        """
        Identify and return meta columns in the DataFrame, mapping 'state' to 'changepoint'.
        """
        cols = df.columns
        if 'state' in cols and 'changepoint' not in cols:
            df.rename({'state': 'changepoint'})
            cols = df.columns
        return [c for c in PlottingPipeline.META_COLS if c in cols]

    @staticmethod
    def get_data_columns(df: pl.DataFrame) -> List[str]:
        """
        Return a list of column names corresponding to data vectors.
        """
        return [
            col
            for col in df.columns
            if col.startswith('PC_') or col.startswith('latent_dim_') or col.startswith('neuron_')
        ]

    @staticmethod
    def parse_dataset_name(name: str) -> Tuple[str, str]:
        """
        Split a compound key into (core_dataset_name, analysis_type).
        Assumes the core name ends with a timestamp YYMMDD_HHMMSS.
        """
        pattern = re.compile(r'^(.+?\d{6}_\d{6})(_.+)$')
        m = pattern.match(name)
        if m:
            return m.group(1), m.group(2)
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], f'_{parts[1]}'
        return name, ''

