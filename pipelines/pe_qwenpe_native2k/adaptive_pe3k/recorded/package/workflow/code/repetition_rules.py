"""Versioned repetition patterns; hit rate is not calibrated accuracy.

Comparison copies ignore only a region's own label. This module never rewrites
captions or changes the field/region ordering used by the text encoder.
"""
import hashlib
import json
import re

VERSION = 'exact_region3x6_string8x128_v1'
POLICY = {
    'version': VERSION,
    'region': {'min_repeats': 3, 'min_span': 6, 'max_period': 32, 'ignore_fields': ['label']},
    'string': {'min_repeats': 8, 'min_span': 128, 'max_period': 256},
    'lexical_tokenizer': r"\w+(?:[-']\w+)*|[^\w\s]",
    'calibration': 'Thresholds require human-labeled calibration; finite output cannot prove an infinite process.',
}


def periodic(seq, max_period, min_repeats, min_span):
    best = None
    for period in range(1, min(max_period, len(seq)//min_repeats)+1):
        run = 0
        for i in range(period, len(seq)):
            run = run+1 if seq[i] == seq[i-period] else 0
            span = run+period
            if span >= min_span and span//period >= min_repeats and (best is None or span > best['span']):
                best = {'start': i-span+1, 'end': i+1, 'period': period,
                        'span': span, 'complete_repeats': span//period}
    return best


def recover_layout(text):
    match = re.search(r'"layout"\s*:\s*\[', text)
    if not match:
        return []
    pos = match.end()
    decoder = json.JSONDecoder()
    result = []
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \n\r\t,':
            pos += 1
        if pos == len(text) or text[pos] == ']':
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except (ValueError, RecursionError):
            break
        if not isinstance(obj, dict):
            break
        result.append(obj)
        pos = end
    return result


def inspect(text):
    if not isinstance(text, str):
        return {'version': VERSION, 'hit': False, 'reasons': [], 'unavailable': True}
    try:
        layout = json.loads(text)['elements']['layout']
        if not isinstance(layout, list) or any(not isinstance(x, dict) for x in layout):
            raise TypeError('invalid region list')
        extraction = 'full_json'
    except (ValueError, KeyError, TypeError, RecursionError):
        layout = recover_layout(text)
        extraction = 'contiguous_partial_layout'
    fingerprints = [json.dumps({k: v for k, v in x.items() if k != 'label'},
                               sort_keys=True, ensure_ascii=False, separators=(',', ':')) for x in layout]
    cycle = periodic(fingerprints, 32, 3, 6)
    if cycle:
        unit = fingerprints[cycle['start']:cycle['start']+cycle['period']]
        cycle['unit_sha256'] = hashlib.sha256(json.dumps(unit).encode()).hexdigest()
    strings = []
    # Includes unfinished final strings in length-truncated JSON. Offsets refer
    # to the raw response, so the evidence remains traceable without repair.
    for match in re.finditer(r'"((?:[^"\\]|\\.)*)(?:"|$)', text):
        words = re.findall(POLICY['lexical_tokenizer'], match.group(1))
        if len(words) < 128:
            continue
        found = periodic(words, 256, 8, 128)
        if found:
            unit = words[found['start']:found['start']+found['period']]
            found.update(string_char_start=match.start(), string_char_length=len(match.group(1)),
                         unit_excerpt=' '.join(unit[:80]),
                         unit_sha256=hashlib.sha256(json.dumps(unit).encode()).hexdigest())
            strings.append(found)
    reasons = []
    if cycle:
        reasons.append('exact_region_sequence_loop')
    if strings:
        reasons.append('long_string_loop')
    return {'version': VERSION, 'hit': bool(reasons), 'reasons': reasons,
            'region_extraction': extraction, 'decoded_regions': len(layout),
            'region_cycle': cycle, 'string_cycles': strings}


def validate_selected(row, plan, recompute=False):
    attempts = row['generation_attempts']
    assert row['generation_policy'] == plan['strategy']
    assert len(attempts) in (1, 2) and row['selected_attempt'] == len(attempts)-1
    assert attempts[0]['mode'] == 'greedy'
    assert bool(attempts[0]['repetition_audit']['hit']) == (len(attempts) == 2)
    for i, attempt in enumerate(attempts):
        mode = 'greedy' if i == 0 else 'retry'
        assert attempt['mode'] == mode
        assert all(attempt['sampling'][k] == v for k, v in plan[mode].items())
        assert attempt['sampling']['seed'] == 42 and attempt['sampling']['sampling_mode'] == mode
        assert attempt['repetition_audit']['version'] == VERSION
    last = attempts[-1]
    assert row['sampling'] == last['sampling']
    assert hashlib.sha256(row['response'].encode()).hexdigest() == last['response_sha256']
    assert row['repetition_audit'] == last['repetition_audit']
    assert row['repetition_rejected'] == bool(last['repetition_audit']['hit'])
    if recompute:
        assert row['repetition_audit'] == inspect(row['response'])
