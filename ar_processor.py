from astropy.io import fits
import sys,os
import re
import numpy as np
#import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy import signal
import argparse
import subprocess

"""
Class to process an archive file, run pulsarX DMffdot to produce .px file and extract Time-Phase, Freq-Phase, DM Curve, Intensity Profile and F-Fdot plots
"""
class ARProcessor:
    def __init__(self,file,tag='test',debug=False):
        self.ar_file = file
        self.save_tag = tag
        self.debug = debug
        self.run_dmffdot()
        self.fits_file = self.ar_file.replace('.ar', '.px')
        self.hdul = fits.open(self.fits_file)
    
        if self.debug:
            self.np_saver()
            print("Running in debug mode")

    def get_time_phase(self):
        time_phase = self.hdul[11].data[0]
        x = np.array(time_phase[0])
        y = np.array(time_phase[1])
        z = np.array(time_phase[2])
        self.TP= z.reshape((y.shape[0],x.shape[0])) # Time Phase Plot
        self.TP_half =self.TP[:, :self.TP.shape[1]//2] #only first half to include only one peak
        #self.TP=np.array(time_phase,dtype=object)

    def get_freq_phase(self):
        freq_phase = self.hdul[8].data[0]
        x = np.array(freq_phase[0])
        y = np.array(freq_phase[1])
        z = np.array(freq_phase[2])
        self.FP = z.reshape((y.shape[0],x.shape[0])) 
        self.FP_half = self.FP[:, :self.FP.shape[1]//2]  # Freq Phase Plot #only first half to include only one peak
        #self.FP=np.array(freq_phase,dtype=object)

    def get_dm_curve(self):
        self.dm_curve = self.hdul[21].data[0][1] #only chis of the DM Curve
    
    def get_intensity_prof(self):
        self.intensity_prof = self.hdul[7].data[0][1] 
        self.intensity_prof_half=self.intensity_prof[:len(self.intensity_prof)//2] #Intensity Profile #only first half to include only one peak

    def get_ffdot(self):
        ffdot = self.hdul[16].data[0]
        x = np.array(ffdot[0])
        y = np.array(ffdot[1])
        z = np.array(ffdot[2])
        self.ffdot = z.reshape((x.shape[0],y.shape[0])) # FFDOT


    @staticmethod
    def extract_number(text):
        match = re.search(r"[-+]?\d*\.\d+|\d+", text.split('=')[1])  # Get the number after '='
        return match.group(0) if match else None

    def extract_and_convert(self, keyword):
        """
        Extracts and converts the value associated with the given keyword
        from the 15th HDU data of the FITS file (Meta).
        
        Args:
            keyword: The keyword to search for (e.g., 'P0', 'Pepoch', 'DM', 'S/N').

        Returns:
            The converted float value if found, otherwise None.
        """
        # Extract the value for the given keyword
        value = next((self.extract_number(row[4]) for row in self.hdul[15].data if keyword in row[4]), None)
        
        # Convert the value to float if found, otherwise return None
        return float(value) if value is not None else None

    def get_period(self):
        self.period = self.extract_and_convert("P0")

    def get_pepoch(self):
        self.pepoch = self.extract_and_convert("Pepoch")

    def get_DM(self):
        self.DM = self.extract_and_convert("DM")

    def get_SNR(self):
        self.SNR = self.extract_and_convert("S/N")
    

    def run_dmffdot(self):
        command = ["dmffdot", "-f", self.ar_file, "--saveimage"]
        print("Running command:", " ".join(command))
        # Run the command
        subprocess.run(command) 

        # Print the final command message
        print("Run complete")

    def np_saver(self):
        self.get_time_phase()
        self.get_freq_phase()
        self.get_dm_curve()
        self.get_intensity_prof()
        self.get_ffdot()
        np.save(f'{self.save_tag}_time_phase.npy',self.TP)
        np.save(f'{self.save_tag}_freq_phase.npy',self.FP)
        np.save(f'{self.save_tag}_dm_curve.npy',self.dm_curve)
        np.save(f'{self.save_tag}_intensity_prof.npy',self.intensity_prof)
        np.save(f'{self.save_tag}_ffdot.npy',self.ffdot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="ar file", required=True)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    parser.add_argument("--tag", help="Tag for naming the saved plots (default: 'test')", default="test")
    args=parser.parse_args()

    # Initialize the ARProcessor with the provided file, tag and debug option
    processor = ARProcessor(args.file,args.tag,args.debug)
    print(f"Succesful! Processed file: {processor.ar_file}")
    
if __name__ == "__main__":
    main()
  


 


