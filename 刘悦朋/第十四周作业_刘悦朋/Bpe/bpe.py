import regex as re
from collections import defaultdict

from config import Config

words = []

with open(Config['path'], encoding='utf-8') as f:
    for line in f:
        chars = re.findall(r'\p{Han}', line)
        words += chars

    words = list(set(words))
    length = len(words)
    print('\n已统计所有单字, 共%d字' % length)

    record = []

    while len(words) < length + Config['limitation']:

        f.seek(0)
        dictionary = defaultdict(int)

        for line in f:
            line = re.sub(r'\n', '', line)
            line = re.sub(r' ', '', line)
            phrases = re.findall(r'[^\p{P}]+', line)
            for phrase in phrases:
                while len(phrase) > 1:
                    # TODO: Find token_1.
                    i = 1
                    if phrase[0: i + 1] in words:
                        i += 1
                    token_1 = phrase[0: i]

                    # TODO: Find token_2.
                    j = i + 1
                    if phrase[i: j + 1] in words:
                        j += 1
                    token_2 = phrase[i: j]

                    token = token_1 + token_2

                    if token not in words:
                        dictionary[token] += 1

                    phrase = phrase[i:]

        max_times = max(list(dictionary.values()))

        keys = [key for key, value in dictionary.items() if value == max_times]

        words += keys
        record += keys

        print('\n当前进度 %d / %d' % (len(words) - length, Config['limitation']))

        print('本轮token最大出现次数 - %d' % max_times)

        print('本轮添加的token - %s' % '、'.join(keys))

    print('\n本次bpe分词共添加下列token\n------------------------\n%s\n------------------------' % '\n'.join(record))

# TODO: Encode a sentence
sen = '春风得意马蹄疾，一日看尽长安花。'
sens = re.findall(r'[^\p{P}]+', sen)

# words = ['春风', '得意', '马蹄', '疾', '一日', '看尽', '长安花']

lst = []

for sen in sens:
    i = 1
    while sen:
        if sen[0: i + 1] in words:
            i += 1

        token = sen[0: i]

        lst.append(words.index(token))

        sen = sen[i:]

print(lst)

# TODO: Decode a sentence

text = []

for idx in lst:
    text.append(words[idx])

print(' '.join(text))
