"""Plot saved numeric data in a separate environment; torch is not needed."""
from pathlib import Path
import sys
from common import read_json, write_json
from analyze import figures

if __name__=='__main__':
    run=Path(sys.argv[1]).resolve()
    data=read_json(run/'figures/plot_data.json')
    status=figures(run,data['summary'],data['shape'],data['config'])
    write_json(run/'figures/plot_status.json',status)
    print(status)
