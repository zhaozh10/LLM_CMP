from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from prompt import prob2text
import time
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import csv
import os.path as osp
import os
from prompt import prompt_en, prompt_zh
from doctorglm_standalone import doctorInit
import json
# from search_engine.src import unit,dataloader
# from search_engine.mpg.log import info,warn

fivedisease={
        "Cardiomegaly":0,
        "Edema":1,
        "Consolidation":2,
        "Atelectasis":3,
        "Pleural Effusion":4,
           }

def query_prompt(path_list: list)-> str:
    report_en=json.load(open('./report_en_dict.json'))
    prompt=""
    for i,path in enumerate(path_list):
        ex=f"Example {i}:\n"+report_en[path].replace('Findings:\n','').replace('Impression:\n','')
        prompt+=ex+'\n'
    return prompt

class ChatBot:
    def __init__(self, model, tokenizer, generation_config, instruction, args):
        self.device=args.device
        self.model = model
        self.model=self.model.to(self.device)
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.instruction=instruction

    def generate_prompt(self, input_text=None):
        
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Input:{input_text}\n\n### Response:"""

    def eval(self, input_text):
        # prompt = self.generate_prompt(input_text)
        prompt=input_text
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        # generation_output = self.model.generate(
        #     input_ids=input_ids,
        #     generation_config=self.generation_config,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=512
        # )
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        
        output=self.tokenizer.decode(generation_output.sequences[0])
        
        return output.split("### Response:")[1].strip()
        # return output
        
def evalChatcad(bot: ChatBot,info:list, args):
    res=[]
    freq=10
    for i, elem in enumerate(tqdm(info)):
        prob=elem['prob']
        converter=prob2text(prob,fivedisease)
        ref_list=elem['reference']
        res_prompt=converter.promptB()
        text_report=elem['raw']
        ref_prompt=query_prompt(ref_list[:3])
        # awesome_prompt=f"\nRevise the report of Network B based on Network A." 
        input_text=res_prompt+" "+"Network B generate a report："+text_report
        # instruction="Revise the report of Network B based on Network A. Please do not mention Network A, Network B, and any provided results. Answer it in english. Suppose you are a doctor writing findings for a chest x-ray report."
        # prompt_report=prompt_report+" "+awesome_prompt+" "+constraints+ref_prompt
        instruction=f"Revise the report of Network B based on Network A. Please do not mention Network A, Network B, and any provided results.Following the style of these examples:\n{ref_prompt} Answer it in english. Suppose you are a doctor writing findings for a chest x-ray report."
        
        prompt_report=f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{instruction}\n\n### Input: {input_text}\n\n### Response:"

        

        message=bot.eval(prompt_report)
        while message == "</s>" or message == "":
            message = bot.eval(prompt_report)
        # # 找到第一个冒号的索引
        # colon_index = message.find(':')
        # # 提取冒号之后的内容
        # message = message[colon_index + 1:].strip()
        res.append(message)
        if (i+1) % freq == 0:
            with open(f'real-plus/{args.tgt_dir}_res.json', 'w') as file:
                json.dump(res, file, indent=4)
                print(f"save at step {i+1}")

    with open(f'real-plus/{args.tgt_dir}_res.json', 'w') as file:
        json.dump(res, file, indent=4)

    csv_res= []
    csv_file=f'real-plus/{args.tgt_dir}_res.csv'
    for elem in res:
        csv_res.append([elem])
    header = ["Report Impression"]
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入header行
        writer.writerow(header)
        # 写入数据行
        writer.writerows(csv_res)

def main(args):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.device=device
    if device=='cpu':
        print("[WARING] CPU only!")
    instruction="Please act as a radiologist, revise the report generated by Network B based on results from Network A. Please combine with your clinical knowledge to ensure factual correction because Network A is not always correct."
    # info=json.load(open('exp_res.json'))
    info=json.load(open('exp_res_reference.json'))

    # prepare your tokenizer and LLM here
    if args.tgt_dir=='luotuo-7b':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama-7b")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-7b")
        model = PeftModel.from_pretrained(model, "/public_bme/data/llm/luotuo-lora-7b-0.3")
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='Ziya-13B':
        ## Ziya-13B
        # tokenizer=None
        # model=None
        tokenizer = AutoTokenizer.from_pretrained('/public_bme/data/llm/Ziya-LLaMA-13B', use_fast=False)
        model = LlamaForCausalLM.from_pretrained('/public_bme/data/llm/Ziya-LLaMA-13B', torch_dtype=torch.float16, device_map='auto')
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='llama':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama-7b")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-7b")
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='llama-13b':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama-13b")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-13b")
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='llama2-13b':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama2-13b-chat")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama2-13b-chat")
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='llama2':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/Llama-2-7b-chat-hf")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//Llama-2-7b-chat-hf")
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='medalpaca':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/medAlpaca")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//medAlpaca")
        generation_config = GenerationConfig(
            num_return_sequences=1,
            top_p = 0.85, 
            temperature = 1.0, 
            repetition_penalty=1., 
            max_new_tokens=1024, 
            do_sample = True,  
            eos_token_id=2, 
            bos_token_id=1, 
            pad_token_id=0
        )
    elif args.tgt_dir=='DoctorGLM':
        tokenizer,model=doctorInit()
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=1024,
        )
    elif args.tgt_dir=='HuaTuoGPT':
        tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/Baichuan",trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("/public_bme/data/llm/HuatuoGPT-7B",torch_dtype=torch.float16, trust_remote_code=False)
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='Baichuan':
        tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/Baichuan2-13B-Chat-v2",use_fast=False,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("/public_bme/data/llm/Baichuan2-13B-Chat-v2",device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        # generation_config = GenerationConfig(
        #     temperature=0.9,
        #     top_p=0.9,
        #     top_k=40,
        #     num_beams=4,
        #     num_return_sequences=1,
        #     max_new_tokens=512,
        # )
        generation_config = GenerationConfig.from_pretrained("/public_bme/data/llm/Baichuan2-13B-Chat-v2")
    elif args.tgt_dir=='chatdoctor':
        tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/chatdoctor")
        model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm/chatdoctor")
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='Pulse':
    # tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/HuatuoGPT-7B", trust_remote_code=True)
        ## HuaTuoGPT-7B
        tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/PULSE-7b")
        model = AutoModelForCausalLM.from_pretrained("/public_bme/data/llm/PULSE-7b",device_map="auto").eval()
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='vicuna':
        tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/vicuna_v1.5")
        model = AutoModelForCausalLM.from_pretrained("/public_bme/data/llm/vicuna_v1.5").eval()
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='mistral':
        tokenizer = AutoTokenizer.from_pretrained("/public_bme/data/llm/Mistral")
        model = AutoModelForCausalLM.from_pretrained("/public_bme/data/llm/Mistral").eval()
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='PMC':
        tokenizer = LlamaTokenizer.from_pretrained('/public_bme/data/llm/PMC-LLaMA')
        model = LlamaForCausalLM.from_pretrained('/public_bme/data/llm/PMC-LLaMA')
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=512,
        )
    elif args.tgt_dir=='ChatGLM-6B':
        tokenizer=AutoTokenizer.from_pretrained("/public_bme/data/llm/chatGLM-6b",trust_remote_code=True)
        model = AutoModel.from_pretrained("/public_bme/data/llm/chatGLM-6b",trust_remote_code=True).half()
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=1024,
        )
    elif args.tgt_dir=='ChatGLM2-6B':
        tokenizer=AutoTokenizer.from_pretrained("/public_bme/data/llm/ChatGLM2-6B",trust_remote_code=True)
        model = AutoModel.from_pretrained("/public_bme/data/llm/ChatGLM2-6B",trust_remote_code=True)
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            num_return_sequences=1,
            max_new_tokens=1024,
        )
    else:
        print("Error! Unknown model!")


    bot=ChatBot(model,tokenizer,generation_config, instruction, args)
    # bot=None
    evalChatcad(bot, info, args)

    print("****** Error! Unknown task ******")
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--save", type=bool,default=True)
    parser.add_argument("--tgt_dir",default='Ziya-13B')
    args = parser.parse_args()
    main(args)

