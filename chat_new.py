import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

LLAMA2_PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""

def generate_response(prompt, model_path="Qwen/Qwen2-1.5B-Instruct", max_tokens=500):
    """
    Generate a response from the LLM based on the provided prompt.
    
    Args:
        prompt (str): The user prompt to generate a response for
        model_path (str): Path to the model or HuggingFace repo ID
        max_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        tuple: (generated text, inference time in seconds)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Using float16 for better performance
        low_cpu_mem_usage=True
    )

    # Move model to XPU if available, else use CPU
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Format the prompt
    formatted_prompt = LLAMA2_PROMPT_FORMAT.format(prompt=prompt)
    
    # Generate predicted tokens
    with torch.inference_mode():
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Start inference
        st = time.time()
        output = model.generate(
            **input_ids,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=max_tokens
        )
        
        # Synchronize if using XPU
        if device == 'xpu':
            torch.xpu.synchronize()
            
        end = time.time()
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text, end-st

def main():
    print("LLM Inference Application")
    print("-----------------------")
    
    # Get user inputs
    model_path = input("Enter model path or HuggingFace repo ID (default: Qwen/Qwen2-1.5B-Instruct): ").strip()
    if not model_path:
        model_path = "Qwen/Qwen2-1.5B-Instruct"
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
        if prompt.lower() == 'quit':
            break
        
        try:
            max_tokens = int(input("Max tokens to generate (default: 500): ") or "500")
        except ValueError:
            max_tokens = 500
            
        print("\nGenerating response...\n")
        
        try:
            _, inference_time = generate_response(prompt, model_path, max_tokens)
            print(f"\nInference time: {inference_time:.2f} seconds")
        except Exception as e:
            print(f"Error during generation: {e}")
        
        print("\n" + "-"*50)

if __name__ == '__main__':
    main()