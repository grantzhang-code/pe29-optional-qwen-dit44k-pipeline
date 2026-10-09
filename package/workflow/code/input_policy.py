"""Instruction routing and provenance shared by preparation and PE inference."""
import copy
import hashlib
import json

ORIGINAL = 'original_user_instruction_verbatim'
QWEN = 'qwen_image_2_1_pe_i2i_rewritten_prompt_verbatim'
QWEN_MODEL = 'Qwen/Qwen-Image-2.1-PE-I2I'
QWEN_REVISION = '72927bc08afc99b7888ceb7d7d51a12db3700bbd'
QWEN_MANIFEST = '656224fd3e0d991bcb73c59ae28a5265007578ac0f728ed80c077406c2524b83'
QWEN_SYSTEM = 'e378fea686a1431581ba4c654d332ae96adad633f144ae738ec8ce9c4fd66439'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def unique(rows):
    result = {}
    sample_ids = set()
    for row in rows:
        key = row['input_index']
        if type(key) is not int or key < 0 or key in result or row['sample_id'] in sample_ids:
            raise ValueError('Duplicate or invalid input identity')
        result[key] = row
        sample_ids.add(row['sample_id'])
    if not result:
        raise ValueError('Empty input')
    return result


def build_inputs(originals, *, use_qwen_pe=False, predictions=None, predictions_sha=None, predictions_uri=None):
    """Replace instruction verbatim; never concatenate or fall back on Qwen failure."""
    unique(originals)
    if type(use_qwen_pe) is not bool:
        raise ValueError('use_qwen_pe must be a boolean')
    if use_qwen_pe:
        from checkpoint_sources import require_s3
        require_s3(predictions_uri)
        if not isinstance(predictions_sha, str) or len(predictions_sha) != 64:
            raise ValueError('Missing frozen Qwen predictions SHA256')
        by_index = unique(predictions or [])
        if set(by_index) != {r['input_index'] for r in originals}:
            raise ValueError('Qwen predictions must exactly cover original inputs')
    elif predictions is not None or predictions_sha is not None or predictions_uri is not None:
        raise ValueError('Qwen artifacts supplied while Qwen-PE is disabled')
    output = []
    for source in originals:
        original = source['instruction']
        if not isinstance(original, str) or not original.strip():
            raise ValueError('Empty original instruction')
        if source.get('original_instruction', original) != original or source['selection'].get('instruction_policy') == QWEN:
            raise ValueError('Expected original user inputs, received rewritten inputs')
        row = copy.deepcopy(source)
        selection = row['selection']
        for key in list(selection):
            if key.startswith('qwen_pe_') or key == 'rewritten_instruction_sha256':
                del selection[key]
        row['original_instruction'] = original
        selection.update(instruction_policy=QWEN if use_qwen_pe else ORIGINAL,
                         original_instruction_en=original, original_instruction_sha256=sha(original.encode()))
        if use_qwen_pe:
            pred = by_index[row['input_index']]
            binding = dict(sample_id=row['sample_id'], instruction_en=original,
                           source_s3_uri=row['source_s3_uri'], source_sha256=selection['source_sha256'],
                           gpt_output_sha256=selection['gpt_output_sha256'],
                           pe_target_canvas_size=row['target_canvas_size'], status='complete',
                           model=QWEN_MODEL, model_revision=QWEN_REVISION,
                           model_manifest_sha256=QWEN_MANIFEST, system_prompt_sha256=QWEN_SYSTEM)
            if any(pred.get(k) != v for k, v in binding.items()):
                raise ValueError(f"Qwen binding mismatch at {row['input_index']}")
            instruction = pred.get('rewritten_prompt')
            attempts = pred.get('attempts') or []
            if (not isinstance(instruction, str) or not instruction.strip() or not attempts
                    or attempts[-1].get('parse_ok') is not True
                    or attempts[-1].get('finish_reason') != 'stop'):
                raise ValueError(f"Invalid Qwen rewrite at {row['input_index']}")
            row['instruction'] = instruction
            selection.update(qwen_pe_model=QWEN_MODEL, qwen_pe_revision=QWEN_REVISION,
                             qwen_pe_predictions_sha256=predictions_sha, qwen_pe_predictions_uri=predictions_uri,
                             qwen_pe_record_sha256=sha(encode(pred)),
                             rewritten_instruction_sha256=sha(instruction.encode()))
        output.append(row)
    return output


def validate_instruction(row, config):
    enabled = config.get('use_qwen_pe')
    if type(enabled) is not bool:
        raise ValueError('Run configure_pipeline.py before inference')
    policy = QWEN if enabled else ORIGINAL
    selection = row['selection']
    original, instruction = row.get('original_instruction'), row['instruction']
    if (not isinstance(original, str) or not original.strip()
            or config.get('instruction_policy') != policy or selection.get('instruction_policy') != policy
            or selection.get('original_instruction_en') != original
            or selection.get('original_instruction_sha256') != sha(original.encode())):
        raise ValueError('Original instruction provenance or selected mode changed')
    if enabled:
        if (selection.get('rewritten_instruction_sha256') != sha(instruction.encode())
                or not config.get('qwen_pe_predictions_sha256')
                or selection.get('qwen_pe_predictions_sha256') != config['qwen_pe_predictions_sha256']
                or selection.get('qwen_pe_predictions_uri') != config['qwen_pe_predictions_uri']
                or selection.get('qwen_pe_model') != QWEN_MODEL
                or selection.get('qwen_pe_revision') != QWEN_REVISION):
            raise ValueError('Qwen rewrite provenance changed')
    elif (instruction != original or config.get('qwen_pe_predictions_sha256') is not None
          or any(k.startswith('qwen_pe_') or k == 'rewritten_instruction_sha256' for k in selection)):
        raise ValueError('Disabled Qwen-PE must use the original instruction verbatim')
