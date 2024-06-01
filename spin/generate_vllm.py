from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings

import warnings
warnings.filterwarnings("ignore")
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--world_size', type=int, default=8) # controls the number of gpus vLLM is allowed to use
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=768)

    # load data
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    else:
        data = data[:]['real']

    prompts_all = []
    prompts_old = []
    corrects_all = []
    prompts_system = []
    for idx in range(len(data)):
        if data[idx][0]['role'] == 'system':
            conversation = [{"role": "system", "content": data[idx][0]['content'] }]
            conversation.append({"role": "user", "content": data[idx][1]['content'] })
            prompts_system.append(data[idx][0]['content'])
            prompts_old.append(data[idx][1]['content'])
            corrects_all.append(data[idx][2]['content'])
        else:
            conversation = [{"role": "user", "content": data[idx][0]['content'] }]
            prompts_system.append("")
            prompts_old.append(data[idx][0]['content'])
            corrects_all.append(data[idx][1]['content'])
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompts_all.append(prompt)


    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    results = [r.replace("</s>","").lstrip() for r in results_gathered]

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in range(len(corrects_all)):
        if prompts_system[idx] == "":
            d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
        else:
            d = {"real": [{"role": "system", "content": prompts_system[idx]}, {"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "system", "content": prompts_system[idx]}, {"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
        #print(d)
        if args.split == 'test':
            filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl"
        else:
            filename = f"{args.output_dir}/loser_{data_frac}.jsonl"
        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')


if __name__ == "__main__":
    main()
