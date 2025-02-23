import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

if __name__ == "__main__":
    config = LlamaConfig(vocab_size=32000,
                         hidden_size=4096//2,
                         intermediate_size=11008//2,
                         num_hidden_layers=32//2,
                         num_attention_heads=32//2,
                         max_position_embeddings=2048//2)
    model = LlamaModel(config=config)
    print(model)

    inputs_ids = torch.randint(low=0, high=config.vocab_size, size=(4, 30))
    for i in range(1000):
        res = model(inputs_ids)

    # res is BaseModelOutputWithPast type
    print(f"{res.last_hidden_state.shape=}")