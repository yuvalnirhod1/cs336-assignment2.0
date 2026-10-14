import pickle
import cProfile
import tqdm
from typing import List, Tuple, Dict
import numpy as np
import os
import itertools

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.opposite_vocab = self.calculate_opposite_vocab(vocab)
        self.merges = self.process_merges(merges)
        if not special_tokens:
            special_tokens = []
        self.special_tokens = special_tokens
        self.special_tokens.sort(key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def process_merges(self, merges: List[Tuple[bytes, bytes]]) -> Dict[bytes, bytes]:
        merges_dict = {}
        for index, merge in enumerate(merges):
            merges_dict[merge] = index, merge[0] + merge[1]
        return merges_dict

    def calculate_opposite_vocab(self, vocab):
        return {v: k for k, v in vocab.items()}

    def look_up_token(self, token):
        return self.opposite_vocab[token]

    def _encode_pretokenized_word(self, word: bytes) -> List[int]:
        word = list(word)

        while True:
            best_merge = None
            best_merge_index = 0
            for index in range(len(word) - 1):
                merge = (word[index], word[index + 1])
                if merge != best_merge and merge in self.merges:
                    if best_merge is None or self.merges[best_merge][0] > self.merges[merge][0]:
                        best_merge = merge
                        best_merge_index = index

            if best_merge is None:
                break

            word[best_merge_index] = self.merges[best_merge][1]
            word.pop(best_merge_index + 1)

            
        for token in word:
            yield self.look_up_token(token)

    def _encode_pretokenized_words(self, pretokenized_words: List[bytes]) -> List[int]:
        for word in pretokenized_words:
            yield from self._encode_pretokenized_word(word)

    def encode(self, text: str) -> List[int]:
        words = encode_pre_tokenize(text, self.special_tokens)
        return list(self._encode_pretokenized_words(words))

    def encode_iterable(self, iterable):
        if self.special_tokens:
            max_special_token_length = max([len(token) for token in self.special_tokens])
        else:
            max_special_token_length = 0

        leftover_chunk = ""
        
        for chunk in tqdm.tqdm(iterable):
            leftover_chunk += chunk

            if len(leftover_chunk) <= max_special_token_length:
                continue

            if max_special_token_length:
                working_chunk = leftover_chunk[:-max_special_token_length]
                leftover_chunk = leftover_chunk[-max_special_token_length:]
            else:
                working_chunk = leftover_chunk
                leftover_chunk = ""

            pre_tokenized_words = encode_pre_tokenize(working_chunk, self.special_tokens)

            last_words_tokens = [] 
            last_word = None
            for word in pre_tokenized_words:
                if last_word is not None:
                    last_words_tokens = self._encode_pretokenized_word(last_word)
                    yield from last_words_tokens
                last_word = word

            if last_word is not None:
                leftover_chunk = b"".join(last_word).decode("utf-8") + leftover_chunk

        pre_tokenized_words = encode_pre_tokenize(leftover_chunk, self.special_tokens)
        yield from self._encode_pretokenized_words(pre_tokenized_words)

    def decode(self, ids: List[int]) -> str:
        text = []
        for id in ids:
            text.append(self.vocab[id.item()])
        return b"".join(text).decode("utf-8", errors="replace")

def tokenize_file(path: str, output_path: str):
    f = open(path, 'r')
    tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    np.save(output_path, np.array([0], dtype=np.uint16))

    iterable = tokenizer.encode_iterable(f)
    chunk_size = 10000000
    temp_file = output_path + "_temp"
    # with open(temp_file, 'wb') as f:
    #     while True:
    #         chunk = list(itertools.islice(iterable, chunk_size))
    #         if not chunk:
    #             break

    #         print("Saving chunk!")
    #         np.array(chunk, dtype=np.uint16).tofile(f)
    
    disk_array = np.memmap(temp_file, dtype=np.uint16, mode='r')

    np.save(output_path, disk_array)
    os.remove(temp_file)

def main():
    # text = "The big brown fox jumped the fence jokepoerasfdck\nhello there my friends asdf"
    # text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # text = "Four score and seven years ago our fathers<|endoftext|>"
    # print(tokenize_file())
    # text = "hello"
    # tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    # encoded = list(tokenizer.encode(text))
    # print(encoded)
    # cProfile.run("tokenize_file()", sort="cumtime")
    # encoded = tokenizer.encode(text)
    # print(encoded)
    # decoded = tokenizer.decode(encoded)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded]
    # print(tokenized_string)

    for file_path in os.listdir("tokenized_data"):
        cmdline = f'tokenize_file("{os.path.join("data", file_path)}", "{file_path.split(".")[0] + "_tokenized"}")'
        print(cmdline)
        eval(cmdline)

        # cProfile.run(cmdline, sort="cumtime")
        # break

if __name__ == "__main__":
    from pre_tokenizer import encode_pre_tokenize
    main()
else:
    from .pre_tokenizer import encode_pre_tokenize
