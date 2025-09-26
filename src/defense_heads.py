"""
Mechanistic interpretability (entrainment head gating) - SKELETON.

Integrate with the companion repo to:
  1) Discover 'entrainment heads' (layer -> head indices).
  2) Register forward hooks to scale or zero those heads' attention during inference.

This skeleton exposes a uniform interface without tying to a specific HF architecture.
"""

from typing import Dict, List, Optional

'''
从可解释性出发，在模型内部（attention/head）层面做干预：当检测到说服模式时，通过“门控”抑制一部分注意力头或特征，使得模型无法被特定模式激活。
后续实现参考论文：Llama See Llama Do
当前为预留接口，未验证
'''
class HeadGate:
    def __init__(self, layer2heads: Dict[int, List[int]], scale: float = 0.0):
        """
        layer2heads: e.g., {12: [3,7], 13: [2]}
        scale: 0.0 = hard zero (drop); 0.3/0.5 = attenuate
        """
        self.map = layer2heads
        self.scale = scale
        self._hooks = []

    def register(self, hf_model) -> None:
        """
        Register forward hooks on attention modules.
        NOTE: You must adapt this to your exact model class
        (e.g., LlamaAttention) to intercept attention probs/scores.
        """
        # Example (pseudo):
        # for layer_idx, heads in self.map.items():
        #     attn_module = hf_model.model.layers[layer_idx].self_attn
        #     h = attn_module.register_forward_hook(self._make_hook(heads))
        #     self._hooks.append(h)
        raise NotImplementedError("Attach to your HF model's attention modules based on its class.")

    def _make_hook(self, heads: List[int]):
        def hook(module, inputs, outputs):
            """
            Modify attention weights for selected heads.
            outputs may be (attn_output, attn_weights) depending on model.
            """
            # Example shape assumptions (bs, heads, q_len, k_len)
            try:
                attn = outputs[1]
                attn[:, heads, :, :] *= self.scale
                return (outputs[0], attn)
            except Exception:
                return outputs
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
