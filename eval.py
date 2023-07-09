from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import time
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import csv
import os.path as osp
import os
from prompt import prompt_en, prompt_zh

prompt_dict={'en':prompt_en,'zh':prompt_zh}



def is_chinese_sentence(sentence:str):
    # 统计句子中中文字符的数量
    chinese_char_count = sum(1 for char in sentence if '\u4e00' <= char <= '\u9fff')

    # 判断中文字符数量是否占总字符数的一定比例（这里假设中文字符占比超过50%）
    if chinese_char_count / len(sentence) > 0.5:
        return True
    else:
        return False


class ChatBot:
    def __init__(self, model, tokenizer, generation_config, instruction, args):
        self.device=args.device
        self.model = model
        self.model=self.model.to(self.device)
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.instruction=instruction
        self.language=args.language

        

    def generate_prompt(self, input_text=None):
        if self.language=='zh':
            return f"""以下是一条指令，其描述了一个任务并附带提供了更多的上下文信息。请编写一条回复来恰当地完成任务所提出的要求。\n\n### 指令：\n\n{self.instruction}\n\n### 输入：{input_text}\n\n### 回复："""
        else:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Input:{input_text}\n\n### Response:"""
        

        # if input_text:
        #     return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Input:{input_text}\n\n### Response:"""
        # else:
        #     return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Response:"""

    def eval(self, input_text):
        start=time.time()
        prompt = self.generate_prompt(input_text)
        print(prompt)
        print('\n')
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        
        output=self.tokenizer.decode(generation_output.sequences[0])
        # print("Response:", output.split("### Response:")[1].strip())
        if self.language=='en':
            return output.split("### Response:")[1].strip()
        else:
            return output.split("### 回复：")[1].strip()


def evalImpression(bot: ChatBot,info:pd.DataFrame, args):
    step=100
    csv_file=osp.join('./results',osp.join(args.tgt_dir,args.file))
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f"results save at {csv_file}")
    res=[]
    header = list(info.keys())+['pseudo_impression']
    # header = ['subject_id', 'study_id', 'findings', 'gt_impression','pseudo_impression']
    for index, row in tqdm(info.iterrows(),total=len(info), desc='Processing'):
        # 提取findings和impression属性
        findings = row['findings']
        gt_impression = row['impression']
        pseudo_impression=bot.eval(findings)
        ret=[row[elem] for elem in header[:-3]]
        ret+=[findings, gt_impression,pseudo_impression]
        res.append(ret)
        # res.append([row['subject_id'],row['study_id'],findings, gt_impression, pseudo_impression])
        if (index+1)%step==0:
            # 打开CSV文件并写入数据
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # 写入header行
                writer.writerow(header)
                # 写入数据行
                writer.writerows(res)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入header行
        writer.writerow(header)
        # 写入数据行
        writer.writerows(res)


def main(args):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.device=device
    if device=='cpu':
        print("[WARING] CPU only!")
    # English or Chinese
    if args.file.split('-')[-1].startswith('ZH'):
        language='zh'
    else:
        language='en'
    print(f"language: {language}")
    args.language=language
    # Init the task-specific instruction
    instruction=prompt_dict[language][args.task]
    info=pd.read_csv(osp.join(args.task,args.file))[:10]

    # prepare your tokenizer and LLM here
    tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama-7b")
    model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-7b")
    model = PeftModel.from_pretrained(model, "/public_bme/data/llm/luotuo-lora-7b-0.3")
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
        num_return_sequences=1,
    )

    bot=ChatBot(model,tokenizer,generation_config, instruction, args)
    if args.task=="ImpressionGPT":
        evalImpression(bot, info, args)
    elif args.task=="RadQNLI":
        pass
    elif args.task=="DeID":
        pass
    else:
        print("****** Error! Unknown task ******")
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--task", default="ImpressionGPT", choices=['ImpressionGPT', 'DeID', 'RadQNLI'], help="task to be evaluated")
    parser.add_argument("--file", default="MIMIC-EN.csv")
    parser.add_argument("--save", type=bool,default=True)
    parser.add_argument("--tgt_dir",default='luotuo-7b')

    args = parser.parse_args()
    main(args)

