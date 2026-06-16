from llm.llm import ask_llm

context = " GPT-Rosalind is a large language model developed by NVIDIA, designed for natural language processing tasks. It is based on the transformer architecture and is trained on a diverse range of text data to generate human-like responses to various prompts and questions."

question = "What is GPT-Rosalind?"

answer = ask_llm(context, question)

print("Answer from LLM:")
print(answer)
