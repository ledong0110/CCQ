"""CCQ: Curvature-Conditioned Query for Linear Attention.

Official implementation of the two model families from the paper:

* ``CCQGDN`` — CCQ on a Gated DeltaNet backbone (``model_type: ccq_gdn``)
* ``CCQGLA`` — CCQ on a Gated Linear Attention backbone (``model_type: ccq_gla``)

Importing this package registers both with HuggingFace ``transformers``,
so any training/inference framework that loads models through
``AutoConfig`` / ``AutoModelForCausalLM`` picks them up automatically:

    import ccq  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("configs/ccq_gdn_340M.json")
    model = AutoModelForCausalLM.from_config(config)
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from ccq.gdn.configuration import CCQGDNConfig
from ccq.gdn.modeling import CCQGDNForCausalLM, CCQGDNModel
from ccq.gla.configuration import CCQGLAConfig
from ccq.gla.modeling import CCQGLAForCausalLM, CCQGLAModel

AutoConfig.register(CCQGDNConfig.model_type, CCQGDNConfig, exist_ok=True)
AutoModel.register(CCQGDNConfig, CCQGDNModel, exist_ok=True)
AutoModelForCausalLM.register(CCQGDNConfig, CCQGDNForCausalLM, exist_ok=True)

AutoConfig.register(CCQGLAConfig.model_type, CCQGLAConfig, exist_ok=True)
AutoModel.register(CCQGLAConfig, CCQGLAModel, exist_ok=True)
AutoModelForCausalLM.register(CCQGLAConfig, CCQGLAForCausalLM, exist_ok=True)

__version__ = "1.0.0"

__all__ = [
    "CCQGDNConfig",
    "CCQGDNForCausalLM",
    "CCQGDNModel",
    "CCQGLAConfig",
    "CCQGLAForCausalLM",
    "CCQGLAModel",
]
