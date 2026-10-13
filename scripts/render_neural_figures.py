"""Render the current single-column Fig. 7/8 from preserved source records."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=int, nargs='+', choices=(7, 8), default=[7, 8])
    parser.add_argument('--output', type=Path, default=Path('outputs/figure_reproduction/current'))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.relative_to((ROOT / 'outputs/figure_reproduction').resolve())
    output.mkdir(parents=True, exist_ok=False)
    view = json.loads((ROOT / 'results/final_neural_ablation.json').read_text(encoding='utf-8'))
    # Read only named input records; never extract archive paths into the checkout.
    with tempfile.TemporaryDirectory(prefix='moe-figure-inputs-') as tmp:
        inputs = Path(tmp)
        with zipfile.ZipFile(ROOT / 'archive/supporting-research-records.zip') as archive:
            for key in ('confirmation', 'stress', 'velocity_bias'):
                row = view[key]
                data = archive.read(row['source_result'])
                if hashlib.sha256(data).hexdigest() != row['source_result_sha256']:
                    raise ValueError('Archived input checksum failed: ' + key)
                target = inputs / row['source_result']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for number in dict.fromkeys(args.figures):
            path = ROOT / 'scripts/plotting' / f'fig{number}_single_column.py'
            spec = importlib.util.spec_from_file_location(f'figure_{number}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.OUT = output / f'fig{number}'
            module.OUT.mkdir()
            module.REFERENCE = ROOT / 'data/figure_arrays/neural_plot_data.json'
            if number == 7:
                module.EXP = inputs / 'experiments'
            else:
                module.CONFIRMATION = inputs / view['confirmation']['source_result']
                module.BIAS = inputs / view['velocity_bias']['source_result']
            module.main()
    print('Current single-column neural figures saved to', output)

if __name__ == '__main__':
    main()
