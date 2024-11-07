

text = """The Unicode Consortium’s stated goal is “enabling people around the world to use computers in any language”. And as you might imagine, the diversity of written languages is immense! To date, Unicode supports 135 different scripts, covering some 1100 languages, and there’s still a long tail of over 100 unsupported scripts, both modern and historical, which people are still working to add.  Given this enormous diversity, it’s inevitable that representing it is a complicated project. Unicode embraces that diversity, and accepts the complexity inherent in its mission to include all human writing systems. It doesn’t make a lot of trade-offs in the name of simplification, and it makes exceptions to its own rules where necessary to further its mission.  Moreover, Unicode is committed not just to supporting texts in any single language, but also to letting multiple languages coexist within one text—which introduces even more complexity."""
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
# print('---')
# print(text)
# print("length:", len(text)) # length: 533
# print('---')
# print(tokens)
# print("length:", len(tokens))  # length: 616




# 按照 bpe 的思想，统计每个2元组出现次数
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
stats = get_stats(tokens)
# print(stats)
# print(sorted(((v, k) for k, v in stats.items()), reverse=True))
top_pair = max(stats, key=stats.get)
print(top_pair)  # (101, 32)


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99)) # [5, 6, 99, 9, 1]

# tokens2 = merge(tokens, top_pair, 256)  # 将出现次数最多的2元组替换为256
# print(tokens2)
# print("length:", len(tokens2))


vocab_size = 276  # 超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens)
# print(ids)


merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    print(pair)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx


print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")



# decoding
"""
vocab = {idx: bytes([idx]) for idx in range(256)} :
这行代码创建了一个字典 vocab，其键是从 0 到 255 的整数索引，值是对应的字节表示。具体来说：
idx 是一个循环变量，遍历从 0 到 255 的整数。
bytes([idx]) 将每个整数 idx 转换为一个包含该整数的单个字节。
{
    0: b'\x00',
    1: b'\x01',
    2: b'\x02',
    ...,
    255: b'\xff'
}
"""
vocab = {idx: bytes([idx]) for idx in range(256)}

for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
# print(vocab)


"""
tokens = b"".join(vocab[idx] for idx in ids):
这段代码的作用是将一个字节序列 tokens 生成出来，该序列是根据 ids 列表中的索引，通过 vocab 字典映射到对应的字节值后拼接而成。
tokens = b"\x00" + b"\x01" + b"\x02"  # 结果是 b'\x00\x01\x02'
"""
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    # print(tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text
print(decode([65, 32, 80, 114, 111, 103, 114, 97, 109, 109, 260, 263, 153, 258, 73, 110, 116, 114, 111, 100, 117, 99, 116, 105, 111, 110, 32, 116, 111, 32, 85, 110, 105, 271, 101,]))


# encoding
def encode(text):
    tokens = text.encode("utf-8")
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]  # 取出替代后的值
        tokens = merge(tokens, pair, idx)
    return tokens
print(encode("A Programmer’s Introduction to Unicode"))
# [65, 32, 80, 114, 111, 103, 114, 97, 109, 109, 261, 272, 153, 256, 73, 110, 116, 114, 111, 100, 117, 99, 275, 271, 260, 270, 85, 110, 105, 263, 100, 101]


print(decode(encode("hello world")))
