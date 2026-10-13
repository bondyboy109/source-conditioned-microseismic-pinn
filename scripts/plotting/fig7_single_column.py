"""Plot-only layout revision of Fig. 7 from unchanged archived records.

The two panels are stacked at actual single-column publication size;
the legend sits centered below both panels with a measured 4 pt gap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
EXP = Path(__file__).resolve().parents[2] / "experiments"
REFERENCE = Path(__file__).resolve().parents[2] / "data/figure_arrays/neural_plot_data.json"
BRANCHES = [("waveform_only", "Waveform-only", "#7b3294", "s"),
            ("hybrid", "Hybrid", "#c62828", "o")]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.fontsize": 7.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-9), (actual, expected)


def read_records():
    paths = [EXP / "traveltime_aligned_moe_confirmation_results.json",
             EXP / "traveltime_aligned_moe_stress_results.json", REFERENCE]
    hashes = {str(p): sha256(p) for p in paths}
    confirmation, stress, reference = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert confirmation["independent_confirmation"] and len(confirmation["events"]) == 8
    conditions = [("Clean\n(n = 8)", confirmation["summary"], 8, 0)]
    rows = [{"condition": "clean", "trial_count": 8, "failures": 0,
             "branches": {b: confirmation["summary"][b] for b, _, _, _ in BRANCHES}}]
    levels = [("20 dB", "noise", 20., "noise:20.0"),
              ("10 dB", "noise", 10., "noise:10.0"),
              ("5 dB", "noise", 5., "noise:5.0"),
              ("20%\nremoved", "missing_receivers", 20., "missing_receivers:20.0"),
              ("40%\nremoved", "missing_receivers", 40., "missing_receivers:40.0")]
    for label, kind, level, key in levels:
        summary = stress["summary"]["by_level"][key]
        records = [r for r in stress["records"] if r["condition_kind"] == kind and r["level"] == level]
        assert len(records) == summary["trial_count"] == 24
        failures = sum(r["status"] != "ok" for r in records)
        for branch, _, _, _ in BRANCHES:
            successful = [r[branch] for r in records if r[branch] is not None]
            assert len(successful) + failures == 24
            assert summary[branch]["failure_count"] == failures
            close(np.median([r["location_error_m"] for r in successful]),
                  summary[branch]["median_location_error_m"])
            close(np.median([r["origin_time_error_s"] for r in successful]),
                  summary[branch]["median_origin_time_error_s"])
        conditions.append((label + "\n(n = 24)", summary, 24, failures))
        rows.append({"condition": key, "trial_count": 24, "failures": failures,
                     "branches": {b: summary[b] for b, _, _, _ in BRANCHES}})
    assert [r[3] for r in conditions] == [0, 0, 0, 11, 3, 7]
    assert rows == reference["stress"], "Scientific stress fields differ from the prior plotted-data archive"
    return conditions, rows, hashes


def make_panels(conditions, size, single_column=False):
    x = np.arange(len(conditions))
    if single_column:
        fig, axes = plt.subplots(2, 1, figsize=size)
        axes[0].set_position([.17, .625, .815, 1.47 / size[1]])
        axes[1].set_position([.17, .72 / size[1], .815, 1.47 / size[1]])
    else:
        fig, axes = plt.subplots(1, 2, figsize=size)
    for branch, label, color, marker in BRANCHES:
        axes[0].plot(x, [r[branch]["median_location_error_m"] for _, r, _, _ in conditions],
                     color=color, marker=marker, ms=3.3 if single_column else 4.3,
                     lw=.95 if single_column else 1.15, label=label)
    axes[0].set_yscale("log")
    axes[0].set_yticks([100, 300, 1000, 3000], ["100", "300", "1000", "3000"])
    axes[0].set_ylim(100, 4000)
    axes[0].axhline(500, ls="--", lw=.7, color=".4")
    axes[0].set_ylabel("Median location error\n(m; log scale)" if single_column
                       else "Median location error (m; log scale)")
    axes[0].set_title("(a) Successful trials only", loc="left")
    for i, (_, _, n, failures) in enumerate(conditions):
        rate = failures / n * 100
        axes[1].bar(i, rate, width=.56, color="#d98484", edgecolor="#b85252", linewidth=.4)
        axes[1].text(i, rate + 1.5, f"{failures}/{n}", ha="center", fontsize=7 if single_column else 8)
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel("Pipeline failure\nrate (%)" if single_column else "Pipeline failure rate (%)")
    axes[1].set_title("(b) All registered trials", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".89", linewidth=.5)
        ax.set_axisbelow(True)
        ax.set_xticks(x, [r[0] for r in conditions], fontsize=7 if single_column else 7.3)
        ax.tick_params(axis="x", length=0, pad=3)
        if single_column:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=7.5,
                         fontweight="normal", color="#333333", pad=3)
    return fig, axes


def axis_dimensions(fig, axes):
    return [{"width_in": ax.get_position().width * fig.get_figwidth(),
             "height_in": ax.get_position().height * fig.get_figheight(),
             "width_over_height": (ax.get_position().width * fig.get_figwidth()) /
                                  (ax.get_position().height * fig.get_figheight()),
             "x_limits": list(ax.get_xlim()), "y_limits": list(ax.get_ylim())}
            for ax in axes]


def main():
    conditions, rows, input_hashes = read_records()
    old_fig, old_axes = make_panels(conditions, (7.15, 3.65))
    old_fig.legend(*old_axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(.53, 1.02), handlelength=1.5)
    old_fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.7)
    old_fig.canvas.draw()
    old_dimensions = axis_dimensions(old_fig, old_axes)
    plt.close(old_fig)

    fig, axes = make_panels(conditions, (3.43, 4.6), single_column=True)
    # Fit every label into a physical 3.297 in column rather than shrinking
    # a double-column design in LaTeX. Save padding is 0.01 in at each edge.
    target_width = 3.297
    for _ in range(5):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in axes]
        content_width = (max(b.x1 for b in bboxes) - min(b.x0 for b in bboxes)) / fig.dpi
        fig.set_figwidth(fig.get_figwidth() + (target_width - .02 - content_width))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    upper_box, lower_box = [ax.get_tightbbox(renderer) for ax in axes]
    # Place the upper panel so its bottom text clears the lower title by 10 pt.
    vertical_shift_px = lower_box.y1 + 10 * fig.dpi / 72 - upper_box.y0
    position = axes[0].get_position()
    axes[0].set_position([position.x0, position.y0 + vertical_shift_px / fig.bbox.height,
                          position.width, position.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = [ax.get_tightbbox(renderer) for ax in axes]
    bottom_px = min(box.y0 for box in text_boxes)
    center_px = (min(box.x0 for box in text_boxes) + max(box.x1 for box in text_boxes)) / 2
    center_x, anchor_y = fig.transFigure.inverted().transform(
        [center_px, bottom_px - 4 * fig.dpi / 72])
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                        frameon=False, bbox_to_anchor=(center_x, anchor_y),
                        handlelength=1.5, handletextpad=.45, columnspacing=1.4,
                        borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    legend_box = legend.get_window_extent(renderer)
    gap_pt = (bottom_px - legend_box.y1) / fig.dpi * 72
    assert abs(gap_pt - 4) < .02
    assert abs((legend_box.x0 + legend_box.x1) / 2 - center_px) < .02
    assert all(a["x_limits"] == b["x_limits"] and a["y_limits"] == b["y_limits"]
               for a, b in zip(old_dimensions, axis_dimensions(fig, axes)))
    # Multiline tick labels must stay separate in both panels.
    for ax in axes:
        tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:]))
    pdf = OUT / "fig_final_stress_neural_ablation.pdf"
    png = OUT / "fig_final_stress_neural_ablation.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.01, dpi=300)
    fig.savefig(png, bbox_inches="tight", pad_inches=.01, dpi=300)
    new_dimensions = axis_dimensions(fig, axes)
    axes_boxes = [ax.get_tightbbox(renderer) for ax in axes]
    panel_gap_pt = (axes_boxes[0].y0 - axes_boxes[1].y1) / fig.dpi * 72
    crop_bbox = fig.get_tightbbox(renderer)
    exported_size = [crop_bbox.width + .02, crop_bbox.height + .02]
    print('exported_size_before_assert', exported_size, 'panel_gap_pt', panel_gap_pt)
    assert abs(exported_size[0] - target_width) < .01
    assert 4.1 <= exported_size[1] <= 4.6
    assert abs(panel_gap_pt - 10) < .02
    new_figsize = list(fig.get_size_inches())
    plt.close(fig)
    assert all(sha256(Path(p)) == h for p, h in input_hashes.items())
    data_path = OUT / "plotted_fig7_data.json"
    data_path.write_text(json.dumps({"stress": rows}, indent=2), encoding="utf-8")
    report = {
        "date": "2026-10-12", "scope": "Fig. 7 true single-column layout only",
        "scientific_fields_exactly_equal_to_prior_plot_archive": True,
        "result_files_changed": False, "new_training_or_localization": False,
        "old_axes": old_dimensions, "new_axes": new_dimensions,
        "old_figsize_in": [7.15, 3.65], "new_figsize_in": new_figsize,
        "exported_size_in": exported_size, "expected_latex_columnwidth_in": target_width,
        "measured_interpanel_text_gap_pt": panel_gap_pt,
        "panel_layout": "Two rows; both have independent full condition labels and sample counts",
        "legend_position": "Centered below combined axes and labels",
        "measured_axis_text_to_legend_gap_pt": gap_pt,
        "tick_labels_do_not_overlap": True,
        "failure_counts": [r[3] for r in conditions], "trial_counts": [r[2] for r in conditions],
        "input_hashes_sha256": input_hashes,
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "outputs_sha256": {str(p): sha256(p) for p in [pdf, png, data_path]},
    }
    (OUT / "FIG7_LAYOUT_REVISION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
