# Constant-velocity stress diagnostic

The formal constant-velocity branch uses the same 15 perturbation conditions
as the FMM-aligned stress test, including three realizations at 20, 10, and
5 dB and 20% and 40% receiver removal. It uses a 968-point spatial coarse
grid followed by three local spatial refinements and a clipped median origin
time based on `vavg=5000 m/s`.

The detailed rows are in
`experiments/constant_velocity_formal_stress_results.json`. The branch shows
kilometre-scale errors across clean and noisy conditions, with additional
failures when valid arrivals are lost. The result is consistent with a strong
systematic error from replacing heterogeneous traveltimes by a constant-speed
shift.

This is a control diagnostic, not an independent confirmation: the velocity
was selected from development experiments, the source locations and receiver
geometry are shared with the registered stress test, and the implementation
does not use differentiable local optimization. It should support the
physical motivation for FMM alignment, while final accuracy claims should use
the locked hybrid/FMM protocol.
