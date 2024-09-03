from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
def initialize_model(config):
    model_name = config['MODEL_CONFIG']['model_name']
    device_map = {"": int(config['MODEL_CONFIG']['device_map'])}

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.getboolean('QUANTIZATION_CONFIG', 'load_in_4bit'),
        bnb_4bit_quant_type=config['QUANTIZATION_CONFIG']['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=config.getboolean('QUANTIZATION_CONFIG', 'bnb_4bit_use_double_quant'),
        bnb_4bit_compute_dtype=getattr(torch, config['QUANTIZATION_CONFIG']['bnb_4bit_compute_dtype']),
    )

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = int(config['TOKENIZER_CONFIG']['pad_token_id'])
    tokenizer.padding_side = config['TOKENIZER_CONFIG']['padding_side']
    
    pipe = pipeline(task="text-generation", model=loaded_model, tokenizer=tokenizer)
    return pipe
    
def generate_response(pipe, prompt):
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given the following natural language question:

{{question_placeholder}}

Convert the question into the corresponding JSON format:
{{
  "causal_problem": ["{{classification_placeholder}}"],
  "dataset": ["{{dataset_placeholder}}"],
  "nodes/treatment/response/mediator/condition": ["{{appropriate_variable_names_placeholder}}"]
}}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    result = pipe(formatted_prompt)
    output = result[0]['generated_text']
    output = output.rsplit('<|end_header_id|>', 1)[-1].strip()
    return output

