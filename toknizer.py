import tiktoken

enc2 = tiktoken.get_encoding("gpt2")
print(f"Vocab size of gpt2: {enc2.n_vocab}")

enc = tiktoken.get_encoding("o200k_base")
print(f"Vocab size of o200k_base: {enc.n_vocab}")

tokens = enc.encode("Hello this is tanish")
print(tokens)