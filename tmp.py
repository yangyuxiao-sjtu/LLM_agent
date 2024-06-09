from vllm import LLM, SamplingParams

# Sample prompts.
prompts = "what is the answer of 1+1"
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, n=3)

# Create an LLM.
llm = LLM(
    model="../meta-llama4b/Meta-Llama-3-8Bawq",
    quantization="AWQ",
    gpu_memory_utilization=0.9,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
res = []
for output in outputs:
    print(output.prompt_token_ids)
    for ans in output.outputs:
        res.append(ans)
print(res)
