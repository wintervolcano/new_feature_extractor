import numpy as np
import sys
import argparse
import logging

def calculate_spin(f=None, f_dot=None, p=None, p_dot=None):
    if f is not None and f_dot is not None:
        p = 1 / f
        p_dot = -f_dot / (f**2)
    elif p is not None and p_dot is not None:
        f = 1 / p
        f_dot = -p_dot / (p**2)
    else:
        raise ValueError("Either (f, f_dot) or (p, p_dot) must be provided")
    return f, f_dot, p, p_dot

def save_chi2_array(period_diff_ms, pdot_for_plot, red_chi2, filename):
    unique_period_diffs = np.unique(period_diff_ms)
    unique_pdot_diffs = np.unique(pdot_for_plot)

    expected_size = len(unique_period_diffs) * len(unique_pdot_diffs)
    if red_chi2.size != expected_size:
        raise ValueError(f"Shape mismatch: red_chi2 has {red_chi2.size} elements, "
                         f"but expected {expected_size} from unique grid values.")

    red_chi2_grid = red_chi2.reshape(len(unique_pdot_diffs), len(unique_period_diffs))

    output_filename = filename + ".npy"
    np.save(output_filename, red_chi2_grid)
    logging.info("Reduced χ² grid saved as '%s'.", output_filename)

def main(input_filename=None):
    parser = argparse.ArgumentParser(description="Save chi² grid from a .ppdot2d file as .npy")
    parser.add_argument("filename", type=str, help="Input .ppdot2d file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    if input_filename is None:
        args = parser.parse_args()
        filename = args.filename
        verbose = args.verbose
    else:
        filename = input_filename
        verbose = False

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info("Processing file: %s", filename)

    pipeline_period = 0.0
    pipeline_pdot = 0.0
    data_lines = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                if 'pfold' in line:
                    try:
                        pipeline_period = float(line.split('=')[1].strip())
                        logging.debug("Parsed pipeline_period: %s s", pipeline_period)
                    except Exception as e:
                        logging.error("Error parsing pfold: %s", e)
                elif 'pdfold' in line:
                    try:
                        pipeline_pdot = float(line.split('=')[1].strip())
                        logging.debug("Parsed pipeline_pdot: %s s/s", pipeline_pdot)
                    except Exception as e:
                        logging.error("Error parsing pdfold: %s", e)
            else:
                data_lines.append(line)

    if pipeline_period == 0 and pipeline_pdot == 0:
        logging.error("Pipeline period and period derivative values not found in the file header.")
        sys.exit(1)

    data = np.array([list(map(float, line.split())) for line in data_lines])
    measured_period = data[:, 0]   # seconds
    measured_pdot = data[:, 1]     # s/s
    red_chi2 = data[:, 2]

    pipeline_freq, pipeline_freq_deriv, _, _ = calculate_spin(p=pipeline_period, p_dot=pipeline_pdot)

    delta_pdot = measured_pdot - pipeline_pdot
    period_diff_ms = (measured_period - pipeline_period) * 1e3
    pdot_for_plot = delta_pdot

    save_chi2_array(period_diff_ms, pdot_for_plot, red_chi2, filename)

if __name__ == "__main__":
    main()