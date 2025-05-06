import json
import sys, os, math
import argparse
import subprocess
import numpy as np
from scipy import signal
from astropy.io import fits
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler

from pfd_processor import PFDProcessor


class FITSGen:

    """
    Class FITSGen takes the following inputs:

    config_path: a json file that specifies the desired features along with downsampling, centering and normalization requirements
    ar_file_path: path to an archive (.ar) file
    output_file_path: path to an output fits (.fits) file

    """

    def __init__(self, config_path, pfd_file_path, output_file_path): #, verbose=False):
        #self.verbose = verbose
        self.load_config(config_path)
        self.load_pfd(pfd_file_path)

        self.bins_TP=self.config['time_phase']['nbins']
        self.bins_FP=self.config['freq_phase']['nbins']
        self.bins_IP=self.config['intensity']['nbins']
        self.bins_ffdot=self.config['ffdot']['nbins']
        self.bins_dm_curve=self.config['dm_curve']['nbins']

        self.output_file = output_file_path
        pass 

    def load_config(self, config_path):
        with open(config_path) as file:
            self.config = json.load(file)
        print("LOAD CONFIG SUCCESS")
        

    def load_pfd(self, pfd_file_path):
        process_here = PFDProcessor(pfd_file_path)
        process_here.get_time_phase()
        process_here.get_freq_phase()
        process_here.get_dm_curve()
        process_here.get_ffdot()
        process_here.get_intensity_prof()
        # process_here.get_DM()
        # process_here.get_SNR()
        # process_here.get_pepoch()
        # process_here.get_period()

        self.TP=process_here.TP
        np.save('TP_test.npy',self.TP)
        self.FP=process_here.FP
        self.IP=process_here.intensity_prof
        # self.TP_half=process_here.TP_half
        # self.FP_half=process_here.FP_half
        # self.IP_half=process_here.intensity_prof_half
        self.ffdot=process_here.ffdot
        self.dm_curve=process_here.dm_curve
        # self.DM=process_here.DM
        # self.period=process_here.period
        # self.pepoch=process_here.pepoch
        # self.SNR=process_here.SNR
        
        print("LOAD PFD SUCCESS")

    
    def center_IP(self):  #recenters a 1D Intensity Profile around maximum peak #assumes 1 profile
        max_idx = np.argmax(self.IP)
        center_idx = len(self.IP) // 2 
        self.shift_amount = center_idx - max_idx
        self.IP = np.roll(self.IP, self.shift_amount)

    def center_2d(self, data):  #recenters 2D Frequency Phase plot or Time Phase plot around maximum peak (rolls along x axis that is phase information) #assumes 1 profile
        #max_index = np.unravel_index(np.argmax(data), data.shape)
        #max_x = max_index[1]
        #shift = data.shape[1] // 2 - max_x
        centered_arr = np.roll(data, self.shift_amount, axis=1)
        return centered_arr
    
    @staticmethod
    def resample_2D(data,bins): 
        resampled_array = np.array([resample(row, bins) for row in data])   # Resample along rows
        resampled_array = np.array([resample(col, bins) for col in resampled_array.T]).T  # Resample along columns
        return resampled_array

    @staticmethod
    def minmax_2D(data):
        flattened_data = data.flatten()
        flattened_data_reshaped = flattened_data.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(flattened_data_reshaped)
        # Reshape the scaled data back to its original shape 
        scaled_data_reshaped = scaled_data.reshape(data.shape)
        return scaled_data_reshaped

    @staticmethod
    def mean_sub_2D(data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        # mean=np.mean(data.flatten())
        # std=np.std(data.flatten())
        normalized_data = (data - mean) / std
        return normalized_data
    
    @staticmethod
    def minmax_1D(data):
        scaler = MinMaxScaler()
        data_reshaped = data.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data_reshaped)
        # Flatten the scaled data back to 1D
        scaled_data_1d = scaled_data.flatten()
        return scaled_data_1d

    @staticmethod
    def fullprof_maker_2D(data): #takes single TP or FP profile, concatenates with itself
        return np.hstack([data, data])
    
    @staticmethod
    def fullprof_maker_1D(data): #for IP
        return np.tile(data, 2)
        
    #play_with functions control the downsampling, centering, normalizing and storing of the features as chosen by the user
 
    def play_with_time_phase(self):
        

        if self.config['time_phase']['store']:

            # center it if selected by user
            if self.config['time_phase']['centering']:
                self.TP=self.center_2d(self.TP) 

            #min-max scaling if selected by user
            if self.config['time_phase']['normalize']:
                self.TP = self.mean_sub_2D(self.TP)
                self.TP = self.minmax_2D(self.TP)

            #one or two profiles
            if self.config['half_or_full']['half']:
                self.TP = self.TP_half
            elif self.config['half_or_full']['full']:
                self.TP = self.fullprof_maker_2D(self.TP)

            #resample into nbins x nbins size
            self.TP=self.resample_2D(self.TP,self.bins_TP) 

            
    def play_with_freq_phase(self):

      

        # center it if selected by user
        if self.config['freq_phase']['centering']:
            self.FP=self.center_2d(self.FP) 

        #min-max scaling if selected by user
        if self.config['freq_phase']['normalize']:
            self.FP = self.mean_sub_2D(self.FP)
            self.FP = self.minmax_2D(self.FP)

        #one or two profiles
        if self.config['half_or_full']['half']:
            self.FP = self.FP_half
        elif self.config['half_or_full']['full']:
            self.FP = self.fullprof_maker_2D(self.FP)

        #resample into nbins x nbins size
        self.FP=self.resample_2D(self.FP,self.bins_FP) 

    def play_with_intensity(self):

        if self.config['intensity']['store']:

            #one or two profiles
            if self.config['half_or_full']['half']: 
                self.IP = self.IP_half
            elif self.config['half_or_full']['full']:
                self.IP = self.fullprof_maker_1D(self.IP)
            

            # center it if selected by user
            if self.config['intensity']['centering']:
                self.center_IP()

            #min-max scaling if selected by user
            if self.config['intensity']['normalize']:
                self.IP = self.minmax_1D(self.IP)

             #resample into nbins 
            self.IP=resample(self.IP,self.bins_IP)

            

    def play_with_ffdot(self):

        if self.config['ffdot']['store']:

            #resample into nbins x nbins size
            self.ffdot=self.resample_2D(self.ffdot,self.bins_ffdot) 

            #min-max scaling if selected by user
            if self.config['ffdot']['normalize']:
                #self.ffdot = self.mean_sub_2D(self.ffdot)
                self.ffdot = self.minmax_2D(self.ffdot)

    def play_with_dm_curve(self):

        if self.config['dm_curve']['store']:

            #resample into nbins 
            self.dm_curve=resample(self.dm_curve,self.bins_dm_curve)

            #min-max scaling if selected by user
            if self.config['dm_curve']['normalize']:
                self.dm_curve = self.minmax_1D(self.dm_curve)

     #The final write_fits function makes a fits file of all the features chosen by the user

    def write_fits(self):
        hdu_0 = fits.PrimaryHDU()  

        if self.config['intensity']['store']:
            self.play_with_intensity()
            hdu_1 = fits.ImageHDU(self.IP, name='Intensity Profile')
        else:
            hdu_1 = fits.ImageHDU(name='Empty Intensity')  # Placeholder HDU

        if self.config['dm_curve']['store']:
            self.play_with_dm_curve()
            hdu_2 = fits.ImageHDU(self.dm_curve, name='DM Curve')
        else:
            hdu_2 = fits.ImageHDU(name='Empty DM Curve') # Placeholder HDU

        print("MID SUCCESS")

        if self.config['time_phase']['store']:
            self.play_with_time_phase()
            hdu_3 = fits.ImageHDU(self.TP, name='Time-Phase Plot')
        else:
            hdu_3 = fits.ImageHDU(name='Empty Time-Phase') # Placeholder HDU

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

       # Extract metadata from self.config
        meta_data = self.config.get("meta_data", {})

        # # Prepare metadata values (store actual values if enabled, otherwise store 'Empty')
        # dm_value = self.DM if meta_data.get("dm_value") else "Empty DM"
        # snr_value = self.SNR if meta_data.get("snr") else "Empty SNR"
        # period_value = self.period if meta_data.get("period") else "Empty Period"
        # pepoch_value = self.pepoch if meta_data.get("pepoch") else "Empty Pepoch"

        # Convert non-empty values to float arrays, empty values to string arrays
        # col_dm = fits.Column(name='DM Value', format='E' if isinstance(dm_value, float) else '20A',
        #                  array=np.array([dm_value], dtype=np.float32 if isinstance(dm_value, float) else 'S20'))
    
        # col_snr = fits.Column(name='SNR', format='E' if isinstance(snr_value, float) else '20A',
        #                   array=np.array([snr_value], dtype=np.float32 if isinstance(snr_value, float) else 'S20'))
    
        # col_period = fits.Column(name='Period', format='E' if isinstance(period_value, float) else '20A',
        #                      array=np.array([period_value], dtype=np.float32 if isinstance(period_value, float) else 'S20'))
    
        # col_pepoch = fits.Column(name='Pepoch', format='E' if isinstance(pepoch_value, float) else '20A',
        #                      array=np.array([pepoch_value], dtype=np.float32 if isinstance(pepoch_value, float) else 'S20'))

        # # Create metadata table
        # metadata_table = fits.BinTableHDU.from_columns([col_dm, col_snr, col_period, col_pepoch], name='Metadata')

        # Writing to FITS
        hdul = fits.HDUList([hdu_0, hdu_1, hdu_2, hdu_3, hdu_4, hdu_5])#, metadata_table])
        hdul.writeto(f'{self.output_file}.fits', overwrite=True)
        print('FITS file created:', f'{self.output_file}.fits') #if self.verbose else None

        print("PROCESSING SUCCESS")

        #hdul = fits.HDUList([hdu_0, hdu_1, hdu_2, hdu_3, hdu_4, hdu_5])
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file path", required=True)
    parser.add_argument("--pfd", help="pfd file path",  required=True)
    parser.add_argument("--output", help="output file path", default="test")
    #parser.add_argument("--v", help="verbose", default=False)
    args=parser.parse_args()

    # Initialize the FITSGenerator
    generator = FITSGen(args.config,args.pfd,args.output) #,args.v)
    generator.write_fits()
    print(f"Succesful! Processed file: {args.pfd}")
    
if __name__ == "__main__":
    main()











    

        


    

