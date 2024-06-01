import torch
import argparse

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='Viet-Mistral/Vistral-7B-Chat')
    parser.add_argument('--qlora_model_path', type=str, default='outputs/iter0-ckpt/checkpoint-5904')
    parser.add_argument('--output_dir', type=str, default='iter0_fullmodel')
    return parser.parse_args()

def main():
    args = setup_arg_parser()
    model_for_merge = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    full_model = PeftModel.from_pretrained(model_for_merge,
                                            args.qlora_model_path,
                                        )
    full_model = full_model.base_model.merge_and_unload()  
    full_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()