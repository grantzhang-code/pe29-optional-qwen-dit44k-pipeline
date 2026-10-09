"""Use verified GPT-2.5 target dimensions with the historical native resize."""
import math

EDITOR_CANVAS_POLICY = 'gpt_image_2_5_target_dimensions_max_1048576_patch32_v1'


def editor_canvas_size(binding):
    size = binding['target_canvas_size']
    selection = binding['selection']
    if size != {'width': selection['gpt_output_width'], 'height': selection['gpt_output_height']}:
        raise ValueError('Target width and height must match the verified GPT target PNG')
    width, height = size['width'], size['height']
    if width <= 0 or height <= 0:
        raise ValueError('Invalid GPT target dimensions')
    scale = min(1.0, math.sqrt(1048576 / (width * height)))
    w = max(32, max(1, round(width * scale)) // 32 * 32)
    h = max(32, max(1, round(height * scale)) // 32 * 32)
    while w * h > 1048576:
        if w >= h and w > 32:
            w -= 32
        elif h > 32:
            h -= 32
        else:
            raise ValueError('Native target exceeds the 1MP cap')
    return w, h
