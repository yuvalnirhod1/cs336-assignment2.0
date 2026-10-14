from typing import Tuple, List, Dict
import pickle
from sortedcontainers import SortedDict
import cProfile
import tqdm


class Position:
    def __init__(self, pair, count, prev = None, next = None):
        self.count = count
        self.pair = pair
        self.prev = prev
        self.next = next

    def pop(self):
        if self.next is not None:
            self.next.prev = self.prev
        if self.prev is not None:
            self.prev.next = self.next

        self.pair = None

    def __repr__(self) -> str:
        return f"Position(count = {self.count}, first = {None == self.prev}, last = {None == self.next})"

class Pair:
    def __init__(self, pair):
        self.pair = pair
        self.positions = []

    def __repr__(self) -> str:
        return f"PairData(pair={self.pair}, positions={len(self.positions), self.positions})"

    def __eq__(self, other) -> bool:
        if other is None or self is None:
            return False
        return self.pair == other.pair

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self.pair.__hash__()

    def __le__(self, other):
        return self.pair <= other.pair

    def __lt__(self, other):
        return self.pair < other.pair

class Count:
    def __init__(self, pair, count):
        self.pair = pair
        self.count = count

    def __le__(self, other):
        if self.count == other.count:
            return self.pair <= other.pair
        return self.count <= other.count

    def __lt__(self, other):
        if self.count == other.count:
            return self.pair < other.pair
        return self.count < other.count

    def __repr__(self) -> str:
        return f"Count: {self.count}, Pair: {self.pair}"

class Pairs:
    def __init__(self, word_count):
        self.word_count = word_count
        self._init_pairs(word_count)

    def _insert_pair(self, pair, position):

        if pair in self.pairs:
            pair_count = self.pairs[pair]
            pair = self.pairs_count.pop(pair_count)
            pair_count.count += position.count
        else:
            pair_count = Count(pair, position.count)

        position.pair = pair
        pair.positions.append(position)
        self.pairs[pair] = pair_count
        self.pairs_count[pair_count] = pair

    def _init_pairs(self, word_count):
        self.pairs = {}
        self.pairs_count = SortedDict({})

        for word, count in word_count.items():

            # Does not count
            if len(word) == 1 or len(word) == 0:
                continue

            # Gather all pairs, and way they are
            last_position = None
            for i in range(len(word) - 1):
                pair = Pair((word[i].to_bytes(1), word[i + 1].to_bytes(1)))
                position = Position(None, count)

                if last_position is not None:
                    last_position.next = position
                    position.prev = last_position
                last_position = position

                self._insert_pair(pair, position)

    def merge_max_pair(self):
        if len(self.pairs_count) > 0:
            max_pair = self.pairs_count.popitem(-1)[1]
            return self._merge_pair(max_pair)
        return None, None

    def _merge_pair(self, pair):
        new_token = b"".join(pair.pair)
        self.pairs.pop(pair)

        while len(pair.positions) > 0:
            position = pair.positions[0]

            if position.prev and position.prev.pair != pair:
                # Handling the old pair
                prev_count = self.pairs[position.prev.pair]
                self.pairs_count.pop(prev_count)
                prev_count.count -= position.prev.count

                assert prev_count.count >= 0, "Counting went wrong!"

                position.prev.pair.positions.remove(position.prev)
                if prev_count.count == 0:
                    self.pairs.pop(position.prev.pair)
                else:
                    self.pairs_count[prev_count] = position.prev.pair

                # Adding the new created pair
                position.prev.pair = Pair((position.prev.pair.pair[0], new_token))
                self._insert_pair(position.prev.pair, position.prev)

            if position.next and position.next.pair != pair:
                next_count = self.pairs[position.next.pair]
                self.pairs_count.pop(next_count)
                next_count.count -= position.next.count

                assert next_count.count >= 0, "Counting went wrong!"

                position.next.pair.positions.remove(position.next)
                if next_count.count == 0:
                    self.pairs.pop(position.next.pair)
                else:
                    self.pairs_count[next_count] = position.next.pair

                position.next.pair = Pair((new_token, position.next.pair.pair[1]))
                self._insert_pair(position.next.pair, position.next)

            position.pop()
            pair.positions.pop(0)
        return pair.pair, new_token

    @property
    def max_pair(self):
        if len(self.pairs_count) == 0:
            return None
        return self.pairs_count.peekitem()[1]

def build_vocab():
    printable_list = []
    non_printable_list = []

    for i in range(256):
        try:
            # Decode as latin1 or utf-8 to get the character, or check if chr(i) is printable
            # Note: bytes([i]).decode('latin1') maps 0-255 directly to Unicode code points U+0000 to U+00FF
            char = chr(i)
            value = i.to_bytes(1)
            if char.isprintable() and char != " ":
                printable_list.append(value)
            else:
                non_printable_list.append(value)
        except Exception:
            non_printable_list.append(value)

    # Combine them with printable ones at the start
    return printable_list + non_printable_list

class TokenizerTrainer:
    def train(self, input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        word_count = pre_tokenize(input_path, special_tokens)
        vocabulary = []
        for token in special_tokens:
            vocabulary.append(token.encode("utf-8"))
        vocabulary.extend(build_vocab())
        print(vocabulary)

        assert vocab_size > len(vocabulary), "Not enough size for vocabulary"

        pairs = Pairs(word_count)

        merges = []
        for i in tqdm.tqdm(range(vocab_size - len(vocabulary))):
            merge, new_token = pairs.merge_max_pair()
            if not new_token:
                break

            merges.append((merge[0], merge[1]))
            vocabulary.append(new_token)

        final_vocab = {i: vocabulary for i, vocabulary in enumerate(vocabulary)}

        self.vocab = final_vocab
        self.special_tokens = special_tokens
        return final_vocab, merges

if __name__ == "__main__":
    from pre_tokenizer import pre_tokenize
    tokenizer_trainer = TokenizerTrainer()
    cmd = 'tokenizer_trainer.train("./data/TinyStoriesV2-GPT4-valid.txt", 5000, special_tokens=["<|endoftext|>"])'
    vocab, merges = eval(cmd)
    with open("vocab.pkl", "wb")  as f:
        pickle.dump(vocab, f)

    with open("merges.pkl", "wb")  as f:
        pickle.dump(merges, f)
    # cProfile.run(cmd, sort="cumtime")
else:
    from .pre_tokenizer import pre_tokenize
