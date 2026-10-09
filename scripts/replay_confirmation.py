"""Replay archived events with current functions, without replacing historical results."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import run_traveltime_aligned_moe_localization_current as runner
from examples.confirm_traveltime_aligned_moe import fmm_baseline

def preflight():
    spec = json.loads((ROOT / 'experiments/traveltime_aligned_moe_confirmation_protocol.json').read_text(encoding='utf-8'))
    loc = json.loads((ROOT / spec['localization_protocol']).read_text(encoding='utf-8'))
    for path_key, hash_key in [('model_checkpoint', 'model_checkpoint_sha256'),
                               ('receiver_fmm_fields', 'receiver_fmm_fields_sha256'),
                               ('localization_protocol', 'localization_protocol_sha256')]:
        if runner.sha256(ROOT / spec[path_key]) != spec[hash_key]:
            raise ValueError('Input changed: ' + spec[path_key])
    mapping = json.loads((ROOT / 'experiments/protocol_hash_resolution.json').read_text(encoding='utf-8'))
    if runner.sha256(ROOT / mapping['current_runner']) != mapping['current_runner_sha256']:
        raise ValueError('Current runner checksum changed')
    archived = json.loads((ROOT / spec['output_result']).read_text(encoding='utf-8'))
    hashes = {r['event']: r['sha256'] for r in archived['events']}
    paths = []
    for row in spec['events']:
        p = ROOT / spec['confirmation_directory'] / row['file']
        if runner.sha256(p) != hashes[row['file']]:
            raise ValueError('Observation changed: ' + row['file'])
        with np.load(p) as d:
            if d['data'].shape != (25, 1001) or not np.isclose(d['times'][1] - d['times'][0], .002):
                raise ValueError('Final observation dimensions changed')
        paths.append(p)
    return spec, loc, paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--events', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec, loc, paths = preflight()
    print('Current model/cache/protocol/runner and eight observations verified. Historical add81f source is unavailable.', flush=True)
    if args.check_inputs:
        return
    if not 1 <= args.events <= len(paths):
        parser.error('--events must be between 1 and 8')
    path = (args.output or ROOT / f'outputs/reproduction/current_replay_{args.events}_events.json').resolve()
    path.relative_to((ROOT / 'outputs/reproduction').resolve())
    if path.exists():
        raise FileExistsError('Refusing to overwrite replay output: ' + str(path))
    first = runner.load_event(paths[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, domain = runner.load_model(loc, first['receivers'], device)
    fields = np.load(ROOT / spec['receiver_fmm_fields'], mmap_mode='r')
    bounds = np.asarray(loc['source_bounds_m_s'], dtype=np.float32)
    search, arrival_spec = loc['search'], loc['arrival_constraint']
    candidates = runner.coarse_grid(bounds, search['coarse_grid_xyzt'])
    rows = []
    for p in paths[:args.events]:
        event = runner.load_event(p, first['receivers'])
        _, picks = runner.pick_arrivals(event, arrival_spec)
        if np.isfinite(picks).sum() < arrival_spec['minimum_valid_picks']:
            raise ValueError('Too few valid arrivals: ' + p.name)
        coarse = runner.sampled_waveforms(event, domain, search['coarse_receiver_stride'], search['coarse_time_stride'], device)
        fine = runner.sampled_waveforms(event, domain, search['refine_receiver_stride'], search['refine_time_stride'], device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        wave = runner.waveform_scores(model, candidates, domain, coarse, search['candidate_batch'])
        arrival = runner.arrival_scores(model, candidates, picks, arrival_spec['arrival_scale_s'], search['candidate_batch'])
        branches = {}
        for name, alpha in [('waveform_only', 0.), ('hybrid', arrival_spec['hybrid_weight'])]:
            total = wave + alpha * arrival
            ids = runner.choose_diverse_starts(candidates, total, search['restarts'], search['minimum_start_separation_m'])
            starts = [{'source_m_s': candidates[i].tolist(), 'coarse_total_loss': float(total[i])} for i in ids]
            winner, _ = runner.refine(model, starts, domain, bounds, fine, picks, search, arrival_spec, alpha)
            branches[name] = winner
        branches['fmm'] = fmm_baseline(fields, picks, bounds, spec)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        with np.load(p) as d:
            truth = np.asarray(d['source_truth'], dtype=np.float64)
        for b in branches.values():
            error = np.asarray(b['estimated_source_m_s']) - truth
            b['component_error_m_s'] = error.tolist()
            b['location_error_m'] = float(np.linalg.norm(error[:3]))
            b['origin_time_error_s'] = float(abs(error[3]))
        row = {'event': p.name, 'sha256': runner.sha256(p), 'valid_raw_picks': int(np.isfinite(picks).sum()),
               'source_truth_m_s_post_search': truth.tolist(), 'search_seconds': elapsed, **branches}
        rows.append(row)
        print(p.name + ': ' + ', '.join(f'{k}={b["location_error_m"]:.1f} m' for k, b in branches.items()), flush=True)
    result = {'purpose': 'current-version replay of archived, already-inspected confirmation events',
              'independent_confirmation': False, 'historical_byte_identical_reproduction': False,
              'device': str(device), 'current_runner_sha256': runner.sha256(Path(runner.__file__)),
              'event_count': len(rows), 'summary': {k: runner.summarize(rows, k) for k in ('waveform_only', 'hybrid', 'fmm')},
              'events': rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('Saved new replay output: ' + str(path))
if __name__ == '__main__':
    main()
