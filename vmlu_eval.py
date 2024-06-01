import os
import json
import tqdm
import pandas as pd
import time
import re
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    data = []
    with open('vmlu_v2/test.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    # Loading the model
    llm = LLM(model='Viet-Mistral/Vistral-7B-Chat')  # Name or path of your model
    sampling_params = SamplingParams(temperature=0.0, max_tokens=768)

    tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat')

    #data = data[:3] # for debuging only
    all_messages = []
    all_res = []

    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn là đáp án đúng nhất cho một câu hỏi trắc nghiệm, gồm chỉ một trong các chữ cái (A, B, C, D hoặc E) và không nên chứa bất kỳ nội dung giải thích thêm nào."
    for idx, doc in enumerate(tqdm.tqdm(data[:])):
        text_choice = '\n'.join(doc['choices'])
        prompt = doc["question"] \
                + "\n\n" \
                + text_choice \
                + "\n" \
                + "Đưa ra chữ cái đứng trước câu trả lời đúng nhất (A, B, C, D hoặc E) của câu hỏi trắc nghiệm trên."

        messages = [{"role": "system", "content": system_prompt }]
        messages.append({"role": "user", "content": prompt})
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_messages.append(messages)
    
    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(all_messages, sampling_params)))

    responses = [r.replace("</s>","").lstrip() for r in results_gathered]

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    for idx, doc in enumerate(tqdm.tqdm(data[:])):
        all_res.append({
            "id": doc['id'],
            "prompt": prompt,
            "question": doc["question"],
            "answer": responses[idx]
        })


    result_folder = "all_res/gpt_result"
    os.makedirs(result_folder, exist_ok=True)
    
    if idx % 100 == 0:
        pd.DataFrame(all_res).to_csv(f"all_res/gpt_result/raw_result_{len(all_res)}.csv", index=False)
    
    df = pd.DataFrame(all_res)
    df['answer'] = df.answer.map(lambda x: x[0].lower())
    df['answer'] = df['answer'].map(lambda x: re.sub(r'[^abcde]', '', x))
    submission_csv = df[['id', 'answer']].to_csv('submission.csv', index=None)