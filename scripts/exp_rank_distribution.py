#!/usr/bin/env python
"""Figure 5 -- per-layer rank distribution for CP vs. TT.

Reads one or more ranks.json files and writes a comparison table + plot.

    python scripts/exp_rank_distribution.py \
        --ranks runs/search_r18_cp/ranks.json runs/search_r18_tt/ranks.json \
        --labels "CP Ranks" "TT Ranks" --out runs/fig5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rene.utils import load_json, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranks', nargs='+', required=True)
    p.add_argument('--labels', nargs='*', default=None)
    p.add_argument('--out', default='./runs/fig5')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    labels = args.labels or [os.path.basename(os.path.dirname(r)) for r in args.ranks]
    series = {}
    for path, label in zip(args.ranks, labels):
        spec = load_json(path)
        series[label] = {'layers': list(spec['ranks'].keys()),
                         'ranks': list(spec['ranks'].values()),
                         'summary': spec.get('summary', {})}
        print(f'{label}: {series[label]["ranks"]}')

    save_json(series, os.path.join(args.out, 'fig5.json'))

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not installed; wrote the JSON only')
            return
        fig, ax = plt.subplots(figsize=(8, 3.2))
        width = 0.8 / max(len(series), 1)
        for k, (label, s) in enumerate(series.items()):
            xs = [i + k * width for i in range(len(s['ranks']))]
            ax.bar(xs, s['ranks'], width=width, label=label)
        ax.set_xlabel('Layer'); ax.set_ylabel('Rank'); ax.legend()
        ax.set_xticks(range(len(next(iter(series.values()))['ranks'])))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'rank_distribution.png'), dpi=160)
        print(f"-> {os.path.join(args.out, 'rank_distribution.png')}")


if __name__ == '__main__':
    main()
