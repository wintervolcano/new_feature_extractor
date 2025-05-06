How to use:

Update json file based on required features.

Current features:

Intensity Profile
DM Curve
Time-Phase
Frequency-Phase
F-Fdot

Each can be downsampled to desired bin size.

Each can be normalized.

User can choose between half profile (1 peak) and full profile (2 peaks).

Intensity profile, time-phase, frequency-phase can be centered to the peak in case of half profile.



Use parallel_Launch_features.sh to extract the features from all the files in a directory or use FITS_generator directly.py for .ar files and FITS_generator_for_pfd.py for pfd files.

This work runs PulsarX (https://arxiv.org/abs/2309.02544) for .ar files and PRESTO (https://github.com/scottransom/presto) for .pfd files under the hood.
