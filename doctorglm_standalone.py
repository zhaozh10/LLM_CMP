import torch
import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from arguments import ModelArguments
import argparse

def doctorInit():
    weight_ptuning = "checkpoint-20000"
    model_args = ModelArguments(model_name_or_path="pretrain_model",
                                    ptuning_checkpoint= weight_ptuning,
                                    pre_seq_len=128,
                                    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model = model.half()
    model.transformer.prefix_encoder.float()
    model = model.cuda()
    model = model.eval()
    return tokenizer,model





    
