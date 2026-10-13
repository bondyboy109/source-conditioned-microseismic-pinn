"""Presentation-only Fig. 8 layout revision; archived scientific values unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION = ROOT / "experiments/traveltime_aligned_moe_confirmation_results.json"
BIAS = ROOT / "experiments/traveltime_aligned_moe_velocity_bias_results.json"
REFERENCE = Path(__file__).resolve().parents[2] / "data/figure_arrays/neural_plot_data.json"
BRANCHES = [("waveform_only", "Waveform-only", "#7b3294", "s"),
            ("hybrid", "Hybrid", "#c62828", "o")]
EXPECTED_ORDER = ["confirm_01_west_shallow.npz", "confirm_02_east_deep.npz",
                  "confirm_03_north_mid.npz", "confirm_04_south_deep.npz",
                  "confirm_05_center_shallow.npz", "confirm_06_southwest_deep.npz",
                  "confirm_07_northeast_mid.npz", "confirm_08_north_deep.npz"]

ORIGINAL_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.7, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
}
SINGLE_COLUMN_RC = {
    **ORIGINAL_RC,
    "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.fontsize": 7.3,
}
plt.rcParams.update(ORIGINAL_RC)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=".89", linewidth=.5)
    ax.set_axisbelow(True)


def draw(rows, figsize):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for branch, label, color, marker in BRANCHES:
        for ax, metric in zip(axes, ["median_location_error_m", "median_origin_time_error_s"]):
            ax.plot([90, 95, 100, 105, 110], [r[branch][metric] for r in rows],
                    color=color, marker=marker, lw=1.15, ms=4.3, label=label)
    axes[0].set(yscale="log", ylabel="Median location error (m; log scale)")
    axes[1].set(ylabel="Median origin-time error (s)")
    for i, ax in enumerate(axes):
        ax.set(xlabel="Inversion velocity / true velocity (%)", xticks=[90, 95, 100, 105, 110])
        ax.set_title(["(a) Spatial sensitivity", "(b) Origin-time sensitivity"][i], loc="left")
        ax.axvline(100, color=".6", ls="--", lw=.7)
        style(ax)
    return fig, axes


def dimensions(fig, axes):
    w, h = fig.get_size_inches()
    return [{"width_in": ax.get_position().width * w,
             "height_in": ax.get_position().height * h,
             "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
             "xscale": ax.get_xscale(), "yscale": ax.get_yscale()}
            for ax in axes]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {p: sha256(p) for p in [CONFIRMATION, BIAS, REFERENCE]}
    confirmation = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    bias = json.loads(BIAS.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["velocity_bias"]
    rows = []
    for key in ["velocity_90pct", "velocity_95pct", "velocity_105pct", "velocity_110pct"]:
        summary = bias["summary"][key]
        records = [r for r in bias["records"] if r["condition_id"] == key]
        assert len(records) == summary["trial_count"] == 8
        assert [r["event"] for r in records] == EXPECTED_ORDER
        assert all(r["valid_pick_count"] >= 6 for r in records)
        for branch, *_ in BRANCHES:
            for metric in ["location_error_m", "origin_time_error_s"]:
                assert np.isclose(np.median([r[branch][metric] for r in records]),
                                  summary[branch]["median_" + metric], rtol=1e-10, atol=1e-9)
        rows.append(summary)
    rows.insert(2, confirmation["summary"])
    data = {"factors_percent": [90, 95, 100, 105, 110], "points": [
        {"velocity_percent": factor, "branches": {b: row[b] for b, *_ in BRANCHES}}
        for factor, row in zip([90, 95, 100, 105, 110], rows)]}
    assert data == reference, "Velocity-bias scientific data differ from archived plot data"

    # Recreate the previous layout only to verify original axis limits and dimensions.
    original, original_axes = draw(rows, (7.15, 4.0))
    original.legend(*original_axes[0].get_legend_handles_labels(), loc="upper center",
                    ncol=2, frameon=False)
    original.tight_layout(rect=(0, 0, 1, .94), w_pad=1.7)
    original.canvas.draw()
    before = dimensions(original, original_axes)
    old_major_ticks = [ax.get_yticks().tolist() for ax in original_axes]
    plt.close(original)

    plt.rcParams.update(SINGLE_COLUMN_RC)
    fig = plt.figure(figsize=(3.4, 4.4))
    fw, fh = fig.get_size_inches()
    panel_height_in = 1.40
    panel_left_in = .56
    panel_right_in = 3.32
    axes = [fig.add_axes([panel_left_in / fw, bottom / fh,
                         (panel_right_in - panel_left_in) / fw,
                         panel_height_in / fh]) for bottom in [2.64, .62]]
    for branch, label, color, marker in BRANCHES:
        for ax, metric in zip(axes, ["median_location_error_m", "median_origin_time_error_s"]):
            ax.plot([90, 95, 100, 105, 110], [r[branch][metric] for r in rows],
                    color=color, marker=marker, lw=1.05, ms=3.5, label=label)
    axes[0].set(yscale="log", ylabel="Median location error\n(m; log scale)")
    axes[1].set(ylabel="Median origin-time error\n(s)")
    for i, (ax, prior, old_ticks) in enumerate(zip(axes, before, old_major_ticks)):
        ax.set_xlabel("Inversion velocity / true velocity (%)", labelpad=3)
        ax.set_xticks([90, 95, 100, 105, 110])
        ax.set_xlim(prior["xlim"])
        ax.set_ylim(prior["ylim"])
        ax.yaxis.set_major_locator(FixedLocator(old_ticks))
        ax.set_title(["(a) Spatial sensitivity", "(b) Origin-time sensitivity"][i],
                     loc="left", fontsize=7.5, fontweight="normal", color="#333333", pad=3)
        ax.axvline(100, color=".6", ls="--", lw=.7)
        style(ax)

    # Fit the complete text-inclusive drawing to a 3.30-in manuscript column.
    target_pdf_width_in = 3.30
    export_pad_in = .035
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in axes]
        content_width_in = (max(b.x1 for b in boxes) - min(b.x0 for b in boxes)) / fig.dpi
        delta_in = target_pdf_width_in - 2 * export_pad_in - content_width_in
        if abs(delta_in) < 1e-6:
            break
        for ax in axes:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width + delta_in / fw, box.height])

    # Space the complete panel contents, including labels and titles, by 10 pt.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    upper_box, lower_box = [ax.get_tightbbox(renderer) for ax in axes]
    panel_gap_pt = 10.0
    current_gap_in = (upper_box.y0 - lower_box.y1) / fig.dpi
    shift_in = current_gap_in - panel_gap_pt / 72
    lower_position = axes[1].get_position()
    axes[1].set_position([lower_position.x0, lower_position.y0 + shift_in / fh,
                          lower_position.width, lower_position.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_boxes = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in axes]
    lowest = min(box.y0 for box in tight_boxes)
    left = min(box.x0 for box in tight_boxes)
    right = max(box.x1 for box in tight_boxes)
    center = (left + right) / 2
    gap_pt = 4.0
    anchor_y = lowest - gap_pt / (72 * fig.get_size_inches()[1])
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                        frameon=False, bbox_to_anchor=(center, anchor_y), borderaxespad=0,
                        handletextpad=.4, handlelength=1.6, columnspacing=1.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lowest_px = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    legend_box = legend.get_window_extent(renderer)
    actual_gap = (lowest_px - legend_box.y1) / fig.dpi * 72
    assert abs(actual_gap - gap_pt) < .05
    panel_boxes_px = [ax.get_tightbbox(renderer) for ax in axes]
    actual_panel_gap = (panel_boxes_px[0].y0 - panel_boxes_px[1].y1) / fig.dpi * 72
    assert abs(actual_panel_gap - panel_gap_pt) < .05
    for ax, prior in zip(axes, before):
        assert list(ax.get_xlim()) == prior["xlim"]
        assert list(ax.get_ylim()) == prior["ylim"]
    after = dimensions(fig, axes)
    assert abs(after[0]["width_in"] - after[1]["width_in"]) < 1e-9
    assert abs(after[0]["height_in"] - after[1]["height_in"]) < 1e-9
    assert panel_boxes_px[0].y0 > panel_boxes_px[1].y1, "Panel labels overlap"
    content_width = (max(b.x1 for b in panel_boxes_px) - min(b.x0 for b in panel_boxes_px)) / fig.dpi
    export_width = content_width + 2 * export_pad_in
    export_height = (max(b.y1 for b in panel_boxes_px) - legend_box.y0) / fig.dpi + 2 * export_pad_in
    assert abs(export_width - target_pdf_width_in) < 1e-5
    assert 4.1 <= export_height <= 4.4
    stem = OUT / "fig_final_velocity_bias_neural_ablation"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.035, dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=.035, dpi=240)
    plt.close(fig)
    for path, initial in inputs.items():
        assert sha256(path) == initial, f"Archived input modified: {path}"
    (OUT / "plotted_velocity_bias_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    report = {
        "date": "2026-10-12", "scope": "Fig. 8 presentation-only true single-column vertical layout revision",
        "scientific_data_exactly_equal_to_archived_velocity_bias": data == reference,
        "inputs_unchanged": [{"path": str(p), "sha256": h} for p, h in inputs.items()],
        "original_axis_dimensions": before, "revised_axis_dimensions": after,
        "original_y_major_ticks": old_major_ticks,
        "export_width_in": export_width, "export_height_in": export_height,
        "single_column_stacked_panels": True, "complete_panel_gap_pt": actual_panel_gap,
        "final_font_sizes_pt": {"axis_labels": 7.5, "titles": 7.5, "tick_labels": 7.0, "legend": 7.3},
        "legend_net_gap_pt": actual_gap, "legend_center_figure_x": center,
        "legend_below_panels": True, "panel_label_overlap": False,
        "new_training_or_localization": False,
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "outputs": [{"path": str(stem.with_suffix(ext)), "sha256": sha256(stem.with_suffix(ext))}
                    for ext in [".pdf", ".png"]],
    }
    (OUT / "FIG8_LAYOUT_REVISION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
