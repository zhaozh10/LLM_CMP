from dis import Instruction
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig



def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())

def generate_prompt(instruction, input_text=None):
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# instruction="请锐评漩涡鸣人"
instruction="作为一个conversational chatbot, 提供友好的回复"
input_text="请锐评漩涡鸣人"
prompt = generate_prompt(instruction, input_text=input_text)
print("hold")

tokenizer = LlamaTokenizer.from_pretrained("/public_bme/data/llm//llama-7b")
model = LlamaForCausalLM.from_pretrained(
    "/public_bme/data/llm//llama-7b",
    device_map="auto",
)
model = LlamaForCausalLM.from_pretrained("/public_bme/data/llm//llama-7b")
model = PeftModel.from_pretrained(model, "silk-road/luotuo-lora-7b-0.3")
     
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

