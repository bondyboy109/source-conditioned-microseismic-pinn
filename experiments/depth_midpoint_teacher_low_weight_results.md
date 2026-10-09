# Low-weight FD-midpoint teacher follow-up

The first teacher-weight sweep tested 0.003, 0.01 and 0.02. Every weight improved the ±80 m midpoint records, but none met the predeclared requirement that the ±160 m endpoint error remain within 10% of the receiver-only control. The smallest tested weight, 0.003, missed that endpoint gate narrowly: it increased endpoint full L2 by 12.9%.

This adaptive development follow-up therefore tested 0.001, 0.002 and 0.0025 using the same base checkpoint, seed, receiver batches, learning rate, 500 updates and four FD-derived midpoint pseudo-labels. The selection rule remained unchanged. All records were previously inspected development data; this is a hyperparameter study, not independent validation.

| Teacher weight | ±80 m midpoint full L2 | ±160 m endpoint full L2 | ±280 m extrapolation full L2 | Eight non-endpoint full L2 | Eight non-endpoint active L2 | Passes gate |
|--:|--:|--:|--:|--:|--:|:--:|
| 0 (control) | 0.2018 | 0.0770 | 0.6462 | 0.4240 | 0.3686 | reference |
| 0.001 | 0.1666 | 0.0821 | 0.6317 | 0.3992 | 0.3428 | yes |
| **0.002** | **0.1600** | **0.0845** | **0.6325** | **0.3962** | **0.3403** | **yes, selected** |
| 0.0025 | 0.1591 | 0.0857 | 0.6330 | 0.3961 | 0.3404 | no |
| 0.003 | 0.1589 | 0.0869 | 0.6334 | 0.3962 | 0.3408 | no |
| 0.01 | 0.1601 | 0.0978 | 0.6345 | 0.3973 | 0.3440 | no |
| 0.02 | 0.1739 | 0.1177 | 0.6349 | 0.4044 | 0.3513 | no |

Weight 0.002 is selected by the registered rule. Relative to the paired control, it reduces midpoint full L2 by 20.7%, reduces the eight-record non-endpoint full L2 by 6.6%, and reduces non-endpoint active L2 by 7.7%. The ±280 m extrapolation group improves by only 2.1%, so the method mainly helps interpolation. Endpoint full L2 increases by 9.7%, remaining just inside the 10% tolerance.

This provides a controlled positive development result: low-weight FD interpolation supervision transfers some local source continuity into the neural forward model. It does not establish broad source generalization. The selected model still has a non-endpoint full L2 of 0.396 and its out-of-bracket error remains 0.632. A new localization experiment is therefore premature.

The selected standard, loadable development checkpoint is `outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth`. It should next be tested on a newly preregistered forward-only set, with no weight adjustment after viewing those records. Only if every new event passes a stated forward-error gate should the same frozen checkpoint proceed to source inversion.

Raw results and provenance are in `depth_midpoint_teacher_weight_sweep_results.json`, `depth_midpoint_teacher_low_weight_protocol.json`, and `depth_midpoint_teacher_low_weight_results.json`.
