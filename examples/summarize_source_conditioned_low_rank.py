"""Summarize the low-rank experiments without rerunning or tuning models."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values)}


def paired_summary(rows: list[dict]) -> dict:
    paired = [row for row in rows if "waveform_plus_fmm" in row["branches"]]
    wave = [row["branches"]["waveform_only"]["location_error_m"] for row in paired]
    hybrid = [row["branches"]["waveform_plus_fmm"]["location_error_m"] for row in paired]
    return {"attempted": len(rows), "available": len(paired),
            "unavailable": len(rows) - len(paired),
            "unavailable_fraction": (len(rows) - len(paired)) / len(rows),
            "waveform_eloc_m": stats(wave), "hybrid_eloc_m": stats(hybrid),
            "hybrid_minus_waveform_eloc_m": stats([h - w for h, w in zip(hybrid, wave)]),
            "hybrid_improved": sum(h < w for h, w in zip(hybrid, wave))}


def main() -> None:
    names = {
        "development": "source_conditioned_low_rank_development_results.json",
        "localization": "source_conditioned_low_rank_localization_results.json",
        "transfer": "source_transfer_low_rank_results.json",
        "noise": "source_conditioned_low_rank_noise_results.json",
        "cross_geology": "source_conditioned_low_rank_cross_geology_results.json",
        "cost": "source_conditioned_low_rank_cross_geology_cost_results.json",
    }
    reports = {}
    provenance = {}
    for key, name in names.items():
        path = ROOT / "experiments" / name
        data = path.read_bytes()
        reports[key] = json.loads(data)
        provenance[key] = {"file": name, "sha256": hashlib.sha256(data).hexdigest()}
    # The cost rerun may change timing, but must preserve checkpoint and errors.
    old, rerun = reports["cross_geology"], reports["cost"]
    assert old["checkpoint_sha256"] == rerun["checkpoint_sha256"]
    assert old["protocol_sha256"] == rerun["protocol_sha256"]
    assert [r["file"] for r in old["events"]] == [r["file"] for r in rerun["events"]]
    max_error_change = max(
        abs(a["branches"][branch][metric] - b["branches"][branch][metric])
        for a, b in zip(old["events"], rerun["events"])
        for branch in ("waveform_only", "waveform_plus_fmm", "fmm")
        for metric in ("location_error_m", "origin_time_error_s"))
    assert max_error_change < 1e-5, "Timing rerun changed scientific results"
    clean = paired_summary([{"branches": {
        name: row[name] for name in ("waveform_only", "waveform_plus_fmm") if name in row}}
        for row in reports["localization"]["events"]])
    noise = {str(snr): paired_summary([r for r in reports["noise"]["events"]
                                      if r["snr_db"] == snr])
             for snr in sorted({r["snr_db"] for r in reports["noise"]["events"]}, reverse=True)}
    hybrid_seconds = rerun["summary"]["waveform_plus_fmm"]["mean_search_seconds"]
    fmm_seconds = rerun["summary"]["fmm"]["mean_search_seconds"]
    result = {
        "source_files": provenance, "clean_paired": clean, "noise_paired": noise,
        "timing_rerun_max_error_change": max_error_change,
        "cost": {"fresh_cross_geology_fmm_setup_seconds": rerun["fmm_setup_seconds"],
                 "mean_hybrid_search_seconds": hybrid_seconds,
                 "mean_fmm_search_seconds": fmm_seconds,
                 "fmm_minus_hybrid_search_seconds": fmm_seconds - hybrid_seconds,
                 "hybrid_to_fmm_search_ratio": hybrid_seconds / fmm_seconds,
                 "favorable_break_even_vs_fmm": False,
                 "note": "Both methods reuse the same FMM fields. Hybrid adds neural training and costs more per search, with worse location errors. No finite favorable break-even event count exists for this measured comparison. No accuracy-matched FD break-even count was measured."},
        "transfer_budget_note": "source_to_target and target_to_target each use 1500 first-stage plus 1500 target-stage updates. target_only_1500 uses only 1500 total updates. Only the two staged arms match the total update budget; the shorter target-only arm is a target-stage-budget control.",
        "scope_note": "These are synthetic, post-hoc evaluations using previously inspected event records. Current low-rank results are separate from earlier MoE/PINN results and must not inherit their prospective-test or UQ claims.",
    }
    dev = reports["development"]["development_summary"]
    lines = ["# 新低秩源条件算子：实验与成本汇总", "",
             "本汇总只对应 source_conditioned_low_rank.py；不混用之前 MoE/PINN 的数字。未修改论文正文。", "",
             "## 前向拟合与清洁定位", "",
             f"16 个开发事件的平均相对 L2 从 {dev['base_mean_full_l2']:.4f} 降至 {dev['operator_mean_full_l2']:.4f}，改善 {dev['relative_mean_improvement']:.2%}。这是开发集结果，不是独立测试性能。", "",
             "清洁定位尝试 8 个事件，混合目标能运行 7 个，1 个因有效到时不足而无法运行。下表只比较同一批 7 个事件。", "",
             "| 样本 | 波形平均 eloc (m) | 混合平均 eloc (m) | 波形中位 eloc (m) | 混合中位 eloc (m) | 改善事件 |",
             "|---|---:|---:|---:|---:|---:|",
             f"| 同一批清洁事件 | {clean['waveform_eloc_m']['mean']:.1f} | {clean['hybrid_eloc_m']['mean']:.1f} | {clean['waveform_eloc_m']['median']:.1f} | {clean['hybrid_eloc_m']['median']:.1f} | {clean['hybrid_improved']}/{clean['available']} |", "",
             "## 噪声：配对比较与不可用率", "",
             "不可用率指有效到时不足而未能执行混合搜索，不等同于定位误差超过某阈值。误差只在双方均可运行的相同事件上计算；因此它仍是条件误差，不包含未运行事件。", "",
             "| SNR (dB) | 配对数/尝试数 | 混合不可用率 | 波形平均 eloc (m) | 混合平均 eloc (m) | 混合改善数 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for snr, row in noise.items():
        lines.append(f"| {float(snr):g} | {row['available']}/{row['attempted']} | {row['unavailable_fraction']:.1%} | {row['waveform_eloc_m']['mean']:.1f} | {row['hybrid_eloc_m']['mean']:.1f} | {row['hybrid_improved']}/{row['available']} |")
    lines.extend(["", "## 跨地质模型：误差与搜索成本", "",
                  "4 个已有合成事件的事后零样本审计。各方法使用相同接收器和 FMM 场；搜索耗时不含建场、模型加载和共享记录准备。神经两分支共享的粗搜耗时分别计入假设独立运行的搜索成本，不能把两列相加当作脚本总耗时。", "",
                  "| 方法 | 中位 eloc (m) | 平均 eloc (m) | 平均搜索耗时 (s/事件) |",
                  "|---|---:|---:|---:|"])
    for name, label in (("waveform_only", "波形"), ("waveform_plus_fmm", "波形 + FMM"), ("fmm", "纯 FMM")):
        row = rerun["summary"][name]
        lines.append(f"| {label} | {row['median_location_error_m']:.1f} | {row['mean_location_error_m']:.1f} | {row['mean_search_seconds']:.4f} |")
    lines.extend(["", f"25 个接收器的 FMM 场重新建场：{rerun['fmm_setup_seconds']:.2f} s（CPU）。与原缓存最大差值 {rerun['rebuilt_vs_cached_max_absolute_difference_s']:.1f} s，复算定位误差最大变化 {max_error_change:g}。",
                  f"模型参数量 {rerun['trainable_architecture_parameters']:,}；联合评估峰值 CUDA 已分配显存 {rerun['cuda_peak_allocated_mib_during_evaluation']:.1f} MiB（不是整机显存占用，也不是逐分支峰值）。",
                  f"已有 Overthrust 开发运行记录：FMM 建场 {reports['development']['fmm_setup_seconds']:.2f} s，低秩模型训练 {reports['development']['training_seconds']:.2f} s。跨地质审计仅重建 FMM 场，不重新训练网络。", "",
                  f"本次混合搜索约为纯 FMM 的 {hybrid_seconds / fmm_seconds:.1f} 倍耗时，且定位误差更高。两者均需 FMM 场，混合方法还需模型训练；因此本比较没有有利的有限盈亏平衡事件数。与连续精度匹配 FD 的盈亏平衡仍未测量。单次、固定顺序的计时只能描述本次运行，不能给出稳定速度提升或统计显著性。", "",
                  f"环境：{rerun['environment']}", "", "## 迁移学习", "",
                  result["transfer_budget_note"], "",
                  "源域→目标域平均定位误差 565.1 m；同预算的目标域→目标域为 465.2 m。当前结果不支持源域预训练提高定位精度。较短 target-only 的 503.0 m 不能作为匹配总成本的对照。", "",
                  "## 当前可写结论与下一步", "",
                  "可写：显式源条件低秩接口改善开发集波形拟合；到时约束在清洁数据及一个跨地质审计中改善波形定位。不能写：优于 FMM、具有计算加速优势、源域迁移已提高定位精度或噪声下普遍改善。", "",
                  "下一项有价值的实验是在开发集上改进到时拾取及到时损失的可靠性，先分析误拾取与深度误差；冻结方案后用新事件确认。当前重复使用的事件只可作开发/事后分析。论文文本压缩留到实验结束。", "",
                  "## 输入记录", ""])
    for item in provenance.values():
        lines.append(f"- `{item['file']}`；SHA256：`{item['sha256']}`")
    (ROOT / "experiments/source_conditioned_low_rank_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "experiments/source_conditioned_low_rank_summary_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"clean_paired": clean, "noise_paired": noise, "cost": result["cost"]},
                     indent=2))


if __name__ == "__main__":
    main()
