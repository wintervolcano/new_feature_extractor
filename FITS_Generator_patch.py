import json
import argparse
import numpy as np
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
from astropy.io import fits

from ar_processor import ARProcessor


class FITSGen:
    """
    Class FITSGen takes the following inputs:

    config_path: a json file that specifies the desired features along with
                 downsampling, centering and normalization requirements.
                 For 2D features (time_phase, freq_phase, ffdot), `nbins`
                 can be:
                   - int (applied to both axes)
                   - [rows, cols] (per-axis)
                   - null (no resampling)
                 For 1D features (intensity, dm_curve), `nbins` can be:
                   - int
                   - null (no resampling)
    ar_file_path: path to an archive (.ar) file
    output_file_path: path to an output fits (.fits) file
    """

    def __init__(self, config_path, ar_file_path, output_file_path):
        self.load_config(config_path)
        self.load_archive(ar_file_path)
        self.output_file = output_file_path

    # ----------------------- config & archive -----------------------

    def load_config(self, config_path):
        with open(config_path) as file:
            self.config = json.load(file)
        print("LOAD CONFIG SUCCESS")

    def load_archive(self, ar_file_path):
        process_here = ARProcessor(ar_file_path)
        process_here.get_time_phase()
        process_here.get_freq_phase()
        process_here.get_dm_curve()
        process_here.get_ffdot()
        process_here.get_intensity_prof()
        process_here.get_DM()
        process_here.get_SNR()
        process_here.get_pepoch()
        process_here.get_period()

        self.TP = process_here.TP
        self.FP = process_here.FP
        self.IP = process_here.intensity_prof
        self.TP_half = process_here.TP_half
        self.FP_half = process_here.FP_half
        self.IP_half = process_here.intensity_prof_half
        self.ffdot = process_here.ffdot
        self.dm_curve = process_here.dm_curve
        self.DM = process_here.DM
        self.period = process_here.period
        self.pepoch = process_here.pepoch
        self.SNR = process_here.SNR

        print("LOAD AR SUCCESS")

    # ----------------------- helpers -----------------------

    @staticmethod
    def _parse_bins_2d(nbins):
        """
        Accepts:
          - int            -> (int, int)
          - [rows, cols]   -> (rows|None, cols|None)
          - None           -> (None, None)  (no resample)
        """
        if nbins is None:
            return (None, None)
        if isinstance(nbins, int):
            return (nbins, nbins)
        if isinstance(nbins, (list, tuple)) and len(nbins) == 2:
            r = None if nbins[0] is None else int(nbins[0])
            c = None if nbins[1] is None else int(nbins[1])
            return (r, c)
        raise ValueError(f"nbins must be None, int, or length-2 list/tuple, got: {nbins!r}")

    @staticmethod
    def _resample_2d_if_needed(data, nbins):
        rows, cols = FITSGen._parse_bins_2d(nbins)
        out = data
        # Only resample axes that were requested and actually change size
        if rows is not None and rows != out.shape[0]:
            out = resample(out, rows, axis=0)
        if cols is not None and cols != out.shape[1]:
            out = resample(out, cols, axis=1)
        return out

    @staticmethod
    def _resample_1d_if_needed(data, nbins):
        if nbins is None:
            return data
        n = int(nbins)
        if n == data.shape[0]:
            return data
        return resample(data, n)

    @staticmethod
    def minmax_2D(data):
        flattened_data = data.flatten().reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(flattened_data)
        return scaled.reshape(data.shape)

    @staticmethod
    def minmax_1D(data):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data.reshape(-1, 1))
        return scaled.flatten()

    # ----------------------- centering -----------------------

    def center_IP(self):
        """Recenters a 1D Intensity Profile around maximum peak."""
        max_idx = int(np.argmax(self.IP))
        center_idx = len(self.IP) // 2
        shift = center_idx - max_idx
        self.IP = np.roll(self.IP, shift)

    def center_2d(self, data):
        """
        Recenters along the phase axis (axis=1) using the phase bin with
        the highest summed power across the other axis.
        """
        phase_profile = np.sum(data, axis=0)   # profile vs phase
        max_x = int(np.argmax(phase_profile))
        shift = data.shape[1] // 2 - max_x
        return np.roll(data, shift, axis=1)

    # ----------------------- feature pipelines -----------------------

    def play_with_time_phase(self):
        cfg = self.config['time_phase']
        if cfg['store']:
            if self.config.get('half_or_full', {}).get('half', False):
                self.TP = self.TP_half

            self.TP = self._resample_2d_if_needed(self.TP, cfg['nbins'])

            if cfg.get('centering', False):
                self.TP = self.center_2d(self.TP)

            if cfg.get('normalize', False):
                self.TP = self.minmax_2D(self.TP)

    def play_with_freq_phase(self):
        cfg = self.config['freq_phase']
        if cfg['store']:
            if self.config.get('half_or_full', {}).get('half', False):
                self.FP = self.FP_half

            self.FP = self._resample_2d_if_needed(self.FP, cfg['nbins'])

            if cfg.get('centering', False):
                self.FP = self.center_2d(self.FP)

            if cfg.get('normalize', False):
                self.FP = self.minmax_2D(self.FP)

            # Keep your original flip (if desired for frequency ordering)
            self.FP = np.flipud(self.FP)

    def play_with_intensity(self):
        cfg = self.config['intensity']
        if cfg['store']:
            if self.config.get('half_or_full', {}).get('half', False):
                self.IP = self.IP_half

            self.IP = self._resample_1d_if_needed(self.IP, cfg['nbins'])

            if cfg.get('centering', False):
                self.center_IP()

            if cfg.get('normalize', False):
                self.IP = self.minmax_1D(self.IP)

    def play_with_ffdot(self):
        cfg = self.config['ffdot']
        if cfg['store']:
            self.ffdot = self._resample_2d_if_needed(self.ffdot, cfg['nbins'])
            if cfg.get('normalize', False):
                self.ffdot = self.minmax_2D(self.ffdot)

    def play_with_dm_curve(self):
        cfg = self.config['dm_curve']
        if cfg['store']:
            self.dm_curve = self._resample_1d_if_needed(self.dm_curve, cfg['nbins'])
            if cfg.get('normalize', False):
                self.dm_curve = self.minmax_1D(self.dm_curve)

    # ----------------------- FITS writing -----------------------

    @staticmethod
    def _is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating, np.bool_)) and not isinstance(x, bool)

    @staticmethod
    def _col_from_value(name, value):
        """Create a FITS Column from value (number -> 'E', else string)."""
        if FITSGen._is_number(value):
            arr = np.array([float(value)], dtype=np.float32)
            fmt = 'E'
        else:
            s = str(value)
            # choose a safe width
            width = max(8, min(64, len(s)))
            arr = np.array([s], dtype=f'S{width}')
            fmt = f'{width}A'
        return fits.Column(name=name, format=fmt, array=arr)

    def write_fits(self):
        hdu_0 = fits.PrimaryHDU()

        # Process 1D first (as before)
        if self.config['intensity']['store']:
            self.play_with_intensity()
            hdu_1 = fits.ImageHDU(self.IP, name='Intensity Profile')
        else:
            hdu_1 = fits.ImageHDU(name='Empty Intensity')

        if self.config['dm_curve']['store']:
            self.play_with_dm_curve()
            hdu_2 = fits.ImageHDU(self.dm_curve, name='DM Curve')
        else:
            hdu_2 = fits.ImageHDU(name='Empty DM Curve')

        print("MID SUCCESS")

        # Process 2D
        if self.config['time_phase']['store']:
            self.play_with_time_phase()
            hdu_3 = fits.ImageHDU(self.TP, name='Time-Phase Plot')
        else:
            hdu_3 = fits.ImageHDU(name='Empty Time-Phase')

        if self.config['freq_phase']['store']:
            self.play_with_freq_phase()
            hdu_4 = fits.ImageHDU(self.FP, name='Frequency-Phase Plot')
        else:
            hdu_4 = fits.ImageHDU(name='Empty Frequency-Phase')

        if self.config['ffdot']['store']:
            self.play_with_ffdot()
            hdu_5 = fits.ImageHDU(self.ffdot, name='F-Fdot Plot')
        else:
            hdu_5 = fits.ImageHDU(name='Empty F-Fdot')

        # Metadata
        meta_data = self.config.get("meta_data", {})

        dm_value = self.DM if meta_data.get("dm_value") else "Empty DM"
        snr_value = self.SNR if meta_data.get("snr") else "Empty SNR"
        period_value = self.period if meta_data.get("period") else "Empty Period"
        pepoch_value = self.pepoch if meta_data.get("pepoch") else "Empty Pepoch"

        col_dm = self._col_from_value('DM Value', dm_value)
        col_snr = self._col_from_value('SNR', snr_value)
        col_period = self._col_from_value('Period', period_value)
        col_pepoch = self._col_from_value('Pepoch', pepoch_value)

        metadata_table = fits.BinTableHDU.from_columns(
            [col_dm, col_snr, col_period, col_pepoch], name='Metadata'
        )

        hdul = fits.HDUList([hdu_0, hdu_1, hdu_2, hdu_3, hdu_4, hdu_5, metadata_table])
        hdul.writeto(f'{self.output_file}.fits', overwrite=True)
        print('FITS file created:', f'{self.output_file}.fits')
        print("PROCESSING SUCCESS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file path", required=True)
    parser.add_argument("--ar", help="ar file path", required=True)
    parser.add_argument("--output", help="output file path (without .fits)", default="test")
    args = parser.parse_args()

    generator = FITSGen(args.config, args.ar, args.output)
    generator.write_fits()
    print(f"Successful! Processed file: {args.ar}")


if __name__ == "__main__":
    main()
