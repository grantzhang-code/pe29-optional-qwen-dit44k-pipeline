import json
import sys
from pathlib import Path
import prepare_inputs as P
from streaming import LocalDataset
from mariana.data.processors.base import Keys

P.initialize()
records = []
for r in json.loads((P.ROOT / 'input-audit.json').read_text())['rejections']:
    events = []
    def trace(frame, event, value):
        if frame.f_code.co_name == '_prepare_text' and event == 'return' and value is None:
            local = frame.f_locals
            events.append({'file': frame.f_code.co_filename, 'line': frame.f_lineno,
                           'scalars': {k:v for k,v in local.items() if isinstance(v, (int,float,bool,type(None)))},
                           'metadata_keys': sorted(local.get('metadata', {}))})
        return trace
    row = dict(LocalDataset(local=str(P.ROOT / 'mds' / r['source']))[r['source_index']])
    row[Keys.SAMPLE_ID] = r['ordinal']
    sys.settrace(trace)
    value = P.PROCESSOR._to_light_sample(row, epoch=0)
    sys.settrace(None)
    assert value is None
    records.append({'request_id': r['request_id'], 'events':events})
P.atomic(P.ROOT / 'rejection-diagnostics.json', records)
print(json.dumps(records, indent=2))
