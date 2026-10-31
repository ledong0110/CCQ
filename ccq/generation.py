"""Version-robust generation input preparation for the CCQ models.

Transformers 5 hands ``prepare_inputs_for_generation`` a pre-allocated ``DynamicCache`` and no
``cache_position`` on the first step. ``len(cache)`` is then already positive although no token
has been written, so a length-based check (as in fla <= 0.5.2) drops the whole prompt and
generation starts from the last prompt token only. The CCQ recurrent cache knows how many
tokens it holds, so slicing is decided by that count instead:

* nothing seen yet  -> feed the full prompt (or ``inputs_embeds``);
* ``seen`` tokens   -> feed only the tokens after position ``seen`` (normally the newest one).

Works with Transformers 4.45 - 5.x and flash-linear-attention 0.4 - 0.5.
"""
from __future__ import annotations


def cache_seen_tokens(cache) -> int:
    """Number of tokens already written to ``cache`` (0 for an empty or foreign cache object)."""
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:  # objects without a usable length report
        return 0


class CCQGenerationMixin:
    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=True,
        logits_to_keep=None,
        cache_position=None,
        **kwargs,
    ):
        seen = cache_seen_tokens(past_key_values)
        inputs = {}
        if inputs_embeds is not None and seen == 0:
            inputs['inputs_embeds'] = inputs_embeds
        else:
            if seen > 0 and input_ids is not None and input_ids.shape[1] > 1:
                input_ids = input_ids[:, seen:] if seen < input_ids.shape[1] else input_ids[:, -1:]
            inputs['input_ids'] = input_ids.contiguous() if input_ids is not None else None
        if logits_to_keep is not None:
            inputs['logits_to_keep'] = logits_to_keep
        inputs.update(past_key_values=past_key_values, use_cache=use_cache, attention_mask=attention_mask)
        return inputs
