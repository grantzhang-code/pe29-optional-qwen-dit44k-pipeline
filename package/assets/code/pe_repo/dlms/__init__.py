"""design-layout-megatron-swift: image->JSON full-FT + MTP on megatron-swift, B300.

Task-specific additions layered on top of the agent-sft (megatron-swift) base:
  - dlms.patches : operator-correctness fixes for the packing/varlen path on B300
  - dlms.data    : image->JSON -> ms-swift megatron dataset converter (plain SFT)
"""
