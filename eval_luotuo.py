from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import time
import torch

class ChatBot:
    def __init__(self, model, tokenizer, generation_config, device, instruction):
        self.device=device
        self.model = model
        self.model=self.model.to(self.device)
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.instruction=instruction
        

    def generate_prompt(self, input_text=None):
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Input:{input_text}\n\n### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{self.instruction}\n\n### Response:"""

    def eval(self, input_text):
        start=time.time()
        prompt = self.generate_prompt(input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        print(len(generation_output.sequences))
        output=self.tokenizer.decode(generation_output.sequences[0])
        print(f"time: {time.time()-start}s")
        print("Response:", output.split("### Response:")[1].strip())
        return output.split("### Response:")[1].strip()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
instruction="作为一个conversational chatbot, 提供友好的回复"
input_text="请锐评漩涡鸣人"


tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm/llama-7b")
model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-7b")
model = PeftModel.from_pretrained(model, "/public_bme/data/llm/luotuo-lora-7b-0.3")
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
    num_return_sequences=1,
)
bot=ChatBot(model,tokenizer,generation_config,device, instruction)
ret=bot.eval(input_text)
print(f"input: {input_text}\noutput: {ret}")
