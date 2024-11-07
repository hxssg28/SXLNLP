# coding: utf-8
import os
import glob
'''
基于bpe创建utf-8编码格式的词表
'''
class Tokenizer_BPE(object):
    def __init__(self, vocab_path, vocab_size=4500):
        self.vocab_path = vocab_path
        self.vocab_sentence = ''
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab_to_bytes = {}
        self.bytes_to_vocab = {}
        self.creat_vocab()

    def creat_vocab(self):
        # 首先将要转化的字符等收集到一起
        self.get_str()
        # 将这些字符转化为utf-8编码
        self.str_to_utf8()
        # 将编码后的字符按照定好的大小进行bpe编码
        self.bpe()
        # 将编码后的字符进行合并
        self.trans_bpe()
        self.vocab = sorted(list(set(self.vocab_utf8_list)), reverse=False)
        self.get_vocab()

    def get_vocab(self):
        for index in self.vocab:
            if index not in self.merges:
                self.vocab_to_bytes[index] = bytes([index])
                self.bytes_to_vocab[bytes([index])] = index
        for (pair1, pair2), index in self.merges.items():
            self.vocab_to_bytes[index] = bytes([pair1, pair2])
            self.bytes_to_vocab[bytes([pair1, pair2])] = index

    def get_str(self):
        file_path = os.path.join(self.vocab_path, '*')
        for file in glob.glob(file_path):
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            self.vocab_sentence += line
            else:
                self.get_str(file)

    def str_to_utf8(self):
        self.vocab_utf8 = self.vocab_sentence[:10000].encode('utf-8')
        self.vocab_utf8_list = [_chr for _chr in self.vocab_utf8]

    def bpe(self):
        self.count = {}
        for pair in zip(self.vocab_utf8_list, self.vocab_utf8_list[1:]):
            self.count[pair] = self.count.get(pair, 0) + 1
        self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True))
        self.count_pair = dict([(v, k) for k, v in self.count.items()])

    def trans_bpe(self):
        # 合并操作
        for i in range(10):
            self.new_list = []
            max_num = max(self.count_pair)
            max_pair = self.count_pair[max_num]
            num1 = len(self.vocab_utf8_list)
            self.new_list = self.merge(self.vocab_utf8_list, max_pair)
            self.vocab_utf8_list = self.new_list
            print(f"merge pairs{max_pair} into a new token {max(self.vocab_utf8_list) + 1}", end="   ")
            self.merges[max_pair] = max(self.vocab_utf8_list) + 1
            print("列表长度：", len(self.vocab_utf8_list), max_pair,"最大的数字：",max(self.vocab_utf8_list) + 1, end="   ")
            self.bpe()
            print("前后列表长度差值：", num1 - len(self.vocab_utf8_list), "原本列表出现最大的pair的次数：", max_num)

    def merge(self, vocab_list, pairs, max_num=None):
        j = 0
        new_list = []
        if max_num:
            trans_token = max_num
        else:
            trans_token = max(vocab_list) + 1
        while j < len(vocab_list):
            if j < len(vocab_list) - 1 and vocab_list[j] == pairs[0] and vocab_list[j+1] == pairs[1]:
                new_list.append(trans_token)
                j += 2
            else:
                new_list.append(vocab_list[j])
                j += 1
        return new_list

    def encoder(self, sentence):
        # 将一句话用我们创建好的词表进行表示
        sentence_utf8 = [char for char in sentence.encode('utf-8')]
        print(sentence_utf8)
        for _pair in zip(sentence_utf8, sentence_utf8[1:]):
            if self.merges.get(_pair, None):
                new_list = self.merge(sentence_utf8, _pair, self.merges[_pair])
                sentence_utf8 = new_list
                print(f"{_pair}合并为{self.merges[_pair]}")
        return sentence_utf8
    
    def decoder(self, _utf8_code):
        # 将词表表示的句子还原为原始句子
        sentence = ""
        text = b''.join(self.vocab_to_bytes[ch] for ch in _utf8_code)
        sentence = text.decode('utf-8', errors="replace")
        return sentence

if __name__ == '__main__':
    bpe_folder_path = r'E:\badouai\ai\第十四周 大语言模型RAG\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes'
    bpe = Tokenizer_BPE(bpe_folder_path)
    sentence_utf8 = bpe.encoder(bpe.vocab_sentence[:100])
    print(sentence_utf8)
    print(bpe.decoder(sentence_utf8))