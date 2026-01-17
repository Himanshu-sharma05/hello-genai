import tiktoken

enc = tiktoken.encoding_for_model('gpt-4o')

text = "Hello im learning gen ai and my name is himanshu"
# Tokens [13225, 770, 7524, 3645, 8440, 326, 922, 1308, 382, 2395, 616, 6916]
tokens = enc.encode(text)
print("Tokens",tokens)

decoded = enc.decode([13225, 770, 7524, 3645, 8440, 326, 922, 1308, 382, 2395, 616, 6916])
print("Decoded",decoded)