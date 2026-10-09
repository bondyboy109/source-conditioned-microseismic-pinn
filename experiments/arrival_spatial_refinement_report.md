# Spatial resolution of the FMM arrival-time baseline

The previous arrival-time source search used a `6 x 6 x 4` coarse grid.
Its nominal local offset was `(2400, 2400, 466.7) m`, exactly the coarse
grid spacing. All 27 local candidates therefore fell on the original grid:
the search did not refine spatial resolution. This explains why a roughly
one-kilometer location error persisted after correcting the origin-time
picker bias.

On the 16 historical development events, we kept the same six-sigma MAD
picker and leave-one-event-out timing correction, and changed only the FMM
candidate search. Three coarse seeds were refined in three rounds with
offsets of one-half, one-quarter and one-eighth of the coarse grid spacing.
Mean FD-observation location error changed from **1069.8 to 290.8 m**;
mean origin-time error changed from **0.0810 to 0.0248 s**. With ideal picks
constructed from FMM travel times at the known sources, the same coarse
search still erred by **1057.0 m**, versus **215.5 m** after refinement.
This ideal-pick arm is an explicit diagnostic that uses truth; it is never
used to invert the observed FD records. The diagnostic does not separate
all residual error sources or establish field-data accuracy.

Before producing further data, we registered eight unseen source positions,
the development-only offset of **-0.262792 s**, the fixed refinement rule,
input hashes and acceptance thresholds in
`arrival_spatial_refinement_protocol.json`. The FD waveforms and solver
reports were hashed in `arrival_spatial_refinement_manifest.json`. Both
inversions used exactly the same 25 selected receivers, corrected picks,
160 m reciprocal FMM fields and origin-time bounds. Source truth was read
after both searches completed, solely to calculate errors.

| Event | Coarse location (m) | Refined location (m) | Coarse origin error (s) | Refined origin error (s) |
|---|---:|---:|---:|---:|
| spatial_01 | 1454 | 225 | 0.0228 | 0.0004 |
| spatial_02 | 905 | 247 | 0.0964 | 0.0256 |
| spatial_03 | 1228 | 113 | 0.0304 | 0.0295 |
| spatial_04 | 1294 | 185 | 0.0626 | 0.0190 |
| spatial_05 | 812 | 160 | 0.0740 | 0.0094 |
| spatial_06 | 763 | 118 | 0.0701 | 0.0026 |
| spatial_07 | 1763 | 138 | 0.0282 | 0.0153 |
| spatial_08 | 606 | 226 | 0.0114 | 0.0119 |
| **Mean** | **1103** | **177** | **0.0495** | **0.0142** |

The **84.0%** reduction in mean spatial error passes the predeclared
25% criterion. Mean origin-time error also fell by 0.0353 s, and both arms
had zero failures. All eight events improved spatially; one origin-time
error increased slightly. On this machine, shared FMM receiver-field setup
took **3.84 s**, while mean per-event coarse search took **0.013 s** and
refined search **0.032 s**. These figures exclude the FD waveform
generation (about 39 s per event) and network training. The refined
search evaluated roughly 282-360 candidate entries per event, with
possible duplicates at clipped boundaries.

This is an improved **FMM baseline**, not evidence of an improvement from
the trained mixture-of-experts model. The newly established 177 m mean
error replaces the misleading coarse-grid result as the relevant spatial
benchmark under these conditions. The experiment tests new source
positions within one synthetic Overthrust velocity model with the same
surface array, 5 Hz source, FD simulator and fixed picker calibration.
It does not establish robustness to new geology, different wavelets,
noisy arrivals or field measurements. In particular, the fixed timing
offset was calibrated for this one synthetic source wavelet.

Machine-readable per-event results, estimates, timings and checksums are
in `arrival_spatial_refinement_results.json`. The FD `.npz` files are
ignored by Git; transfer them alongside the manifest to reproduce this
test on another machine. The write-once evaluation script refuses to
overwrite the existing result.
