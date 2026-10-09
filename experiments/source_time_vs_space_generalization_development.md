# Source-time versus spatial generalization (development diagnostic)

We froze `outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth`.
The 16 receiver-training events fit well (mean original-amplitude relative
L2 = 0.0936; range 0.0704--0.1245), whereas the three existing development
events have mean L2 = 0.8868. The development mean is 9.48 times the
training mean; this describes a generalization gap, not a formal uncertainty
estimate. Mean training 1--2 s relative L2 is 0.1144, compared with
0.9250, 1.2129, and 1.0608 for the 250, 800, and 2000 m development
events, respectively.

To isolate origin-time conditioning from spatial generalization, we kept
training event 01 at (3500, 4000, 500) m and generated three **new FD
records** at origin times 0.16, 0.20, and 0.28 s. The training record has
origin time 0.12 s. All four use the same 5 Hz source, 40 m FD grid, 0.002 s
sampling, 0--2 s window, 25 receivers, and six zero-Dirichlet boundaries.
The new FD records are development diagnostics, not training or independent
test data.

| Same spatial source, origin time (s) | Change in time (s) | Raw relative L2 | Peak amplitude ratio |
| ---: | ---: | ---: | ---: |
| 0.12 (training) | 0.00 | 0.0978 | 0.998 |
| 0.16 (new FD) | 0.04 | 0.1072 | 1.013 |
| 0.20 (new FD) | 0.08 | 0.1060 | 1.012 |
| 0.28 (new FD) | 0.16 | 0.1040 | 1.012 |

By contrast, the existing development event displaced 250 m from training
source 01 with the same 0.12 s origin time has raw relative L2 = 0.7405.
The 800 m development event has L2 = 0.9832. The third development source
is 2000 m from training source 01 but only about 1136 m from training source
09, with a different origin time; its L2 is 0.9369. These three events are
not a controlled sweep of increasing nearest-training distance.

This controlled comparison shows that the network can interpolate tested
origin times at one memorized location while generalizing poorly even to a
250 m spatial displacement. It does not establish that every unseen time
or location behaves likewise. Simply tuning the receiver/PDE weight is not
enough to resolve the spatial waveform discrepancy. Source geometry and
heterogeneous propagation remain the next development target before any
localization or uncertainty claim.

An auxiliary zero-padded shift of the original training trace gave L2 around
0.09, but the shifted record differs from each independently recomputed FD
record by 0.059--0.062 relative L2. The 5 Hz Ricker tail before t = 0 is
not zero, so the shifted trace is an approximate diagnostic, **not** the
reference used for the table above.

Reproduction (PowerShell, with `torch_env` active):

```powershell
foreach ($ts in @(0.16, 0.20, 0.28)) {
  $tag = ('{0:000}' -f [int]($ts * 100))
  python examples/generate_fd_observation.py --config data/observations/experiment_draft.json --source 3500 4000 500 $ts --duration 2.0 --output "data/observations/development_time_shift/train01_t${tag}_0to2s.npz"
}
$events = Get-ChildItem data/observations/development_time_shift/train01_*_0to2s.npz | Sort-Object Name | ForEach-Object FullName
python examples/audit_forward_error_decomposition.py outputs/receiver_weight_0.03_hybrid_1000/conditioned_5Hz.pth $events --output experiments/receiver_weight_0.03_fd_time_shift_development.json
```

Event-level artifacts are in
`experiments/receiver_weight_0.03_hybrid_training_diagnostic.json`,
`experiments/receiver_weight_0.03_hybrid_development.json`, and
`experiments/receiver_weight_0.03_fd_time_shift_development.json`.
