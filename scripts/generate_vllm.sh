python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 0 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 1 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 2 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 3 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 4 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 5 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 6 --world_size 1 --output_dir generated/iter0
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 7 --world_size 1 --output_dir generated/iter0

# Generate for the test split as well
python3 spin/generate_vllm.py --model Viet-Mistral/Vistral-7B-Chat --input_dir UCLA-AGI/SPIN_iter0 --frac_len 3000 --data_frac 0 --world_size 1 --output_dir generated/iter0 --split test