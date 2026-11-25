"""Bounded asset preparation subprocess, entirely outside inference measurements."""
import argparse
from pathlib import Path
from common import Events
from vit import prepare_assets

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--images',nargs=4)
    a=p.parse_args()
    ev=Events(a.run/'logs/assets.jsonl')
    ev.emit('assets_start')
    assets=prepare_assets(a.run,a.images)
    ev.emit('assets_end',assets=assets)
