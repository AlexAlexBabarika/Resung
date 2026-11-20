"""Resung's 44.1 kHz front end.

audio     resampling, loudness normalisation, segmentation
mel       the vocoder's log-mel, and A-weighted loudness on the same frames
f0        RMVPE and F0 grid conversions
content   ContentVec features (frozen, run on the fly)
timbre    ECAPA speaker embedding
perturb   the WORLD shift applied to the content encoder's input in training
process   writes the training cache
"""
