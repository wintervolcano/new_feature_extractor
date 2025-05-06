import argparse
import time, sys
import numpy as np
import configparser
import presto.prepfold as pp
import matplotlib.pyplot as plt
import os
from scipy import signal
import subprocess
import presto.prepfold as pp


class PFDProcessor:
    def __init__(self, pfd_file):
        self.pfd_file = pfd_file
        self.pfd_contents = pp.pfd(self.pfd_file)  # presto

    def get_time_phase(self):
        self.pfd_contents.dedisperse()
        self.pfd_contents.adjust_period()
        self.TP = self.pfd_contents.profs.sum(1)  # Time Phase Plot

    def get_freq_phase(self):
        self.pfd_contents.dedisperse()
        self.pfd_contents.adjust_period()
        self.FP = self.pfd_contents.profs.sum(0)  # Freq Phase Plot

    def get_dm_curve(self):
        lodm = self.pfd_contents.dms[0]
        hidm = self.pfd_contents.dms[-1]
        self.chis, self.DMs = self.pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=64, device='/null')  # DM Curve 64 bins hard coded
        self.dm_curve = self.chis #(self.chis, self.DMs)

    def get_intensity_prof(self):
        if 'subdelays' not in self.pfd_contents.__dict__:
            print("Dedispersing first...")
            self.pfd_contents.dedisperse()
        self.pfd_contents.adjust_period()
        self.intensity_prof = self.pfd_contents.sumprof  #Intensity Profile

    #def get_ffdot(self):
        # subprocess.run(["show_pfd", "-noxwin", self.pfd_file])
        # ppdot2d_file = self.pfd_file.replace('.pfd', '.pfd.ppdot2d')  # New file named this way
    def get_ffdot(self):
        try:
            # Run PRESTO's show_pfd to generate the .ppdot2d file
            subprocess.run(["show_pfd", "-noxwin", self.pfd_file], check=True)

            # Extract base name
            base_name = os.path.basename(self.pfd_file)
            ppdot2d_name = base_name.replace('.pfd', '.pfd.ppdot2d')
            ppdot2d_file = os.path.join(os.getcwd(), ppdot2d_name)

            # Run your script to convert .ppdot2d to .npy
            subprocess.run(["python3", "ppdot_saver.py", ppdot2d_file], check=True)

            # Generate the full path to the .npy file
            npy_file = os.path.join(os.getcwd(), base_name.replace('.pfd', '.pfd.ppdot2d.npy'))
            
            # Check if the .npy file exists
            if os.path.exists(npy_file):
                self.ffdot = np.load(npy_file)
                print(f"Successfully loaded {npy_file}")
            else:
                raise FileNotFoundError(f"File {npy_file} not found!")

            # Cleanup all related temporary files
            extensions_to_remove = ['.pfd.bestprof', '.pfd.ppdot2d', '.pfd.ppdot2d.npy', '.pfd.png', '.pfd.ps']
            for ext in extensions_to_remove:
                file_to_remove = os.path.join(os.getcwd(), base_name.replace('.pfd', ext))
                try:
                    os.remove(file_to_remove)
                    print(f"Removed {file_to_remove}")
                except FileNotFoundError:
                    pass  # It's fine if the file doesn't exist
            
        except subprocess.CalledProcessError as e:
            print(f"Error while running subprocess: {e}")
        except FileNotFoundError as e:
            print(f"File error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="pfd file", required=True)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    parser.add_argument("--tag", help="Tag for naming the saved plots (default: 'test')", default="test")
    args = parser.parse_args()

    # Initialize the PFDProcessor with the provided file
    processor = PFDProcessor(args.file)
    print(f"Processing file: {processor.pfd_file}")
    
    # If debug mode is enabled, print the 5 plots and save them as pngs and npys
    if args.debug:
        processor.get_time_phase()
        plt.plot(processor.TP)
        plt.title('Time Phase')
        plt.savefig(f"{args.tag}_time_phase.png")
        np.save(f"{args.tag}_time_phase.npy", processor.TP)
        plt.close()
        print(f"Time Phase data: {processor.TP}")

        processor.get_freq_phase()
        plt.plot(processor.FP)
        plt.title('Frequency Phase')
        plt.savefig(f"{args.tag}_freq_phase.png")
        np.save(f"{args.tag}_freq_phase.npy", processor.FP)
        plt.close()
        print(f"Frequency Phase data: {processor.FP}")

        processor.get_dm_curve()
        plt.plot(processor.dm_curve)
        plt.title('DM Curve')
        plt.savefig(f"{args.tag}_dm_curve.png")
        np.save(f"{args.tag}_dm_curve.npy", processor.dm_curve)
        plt.close()
        print(f"DM curve: {processor.dm_curve}")

        processor.get_intensity_prof()
        plt.plot(processor.intensity_prof)
        plt.title('Intensity Profile')
        plt.savefig(f"{args.tag}_intensity_prof.png")
        np.save(f"{args.tag}_intensity_prof.npy", processor.intensity_prof)
        plt.close()
        print(f"Intensity profile: {processor.intensity_prof}")

        processor.get_ffdot()
        plt.plot(processor.ffdot)
        plt.title('FFDot')
        plt.savefig(f"{args.tag}_ffdot.png")
        np.save(f"{args.tag}_ffdot.npy", processor.ffdot)
        plt.close()
        print(f"FFDot: {processor.ffdot}")

if __name__ == "__main__":
    main()
  

#example usage:
#python pfd_processor.py --file your_pfd_file.pfd --debug --tag custom_tag
