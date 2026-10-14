import os
from typing import BinaryIO, Dict, List
import regex as re
from multiprocessing import Pool

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 16

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(filepath: str, start: int, end: int, special_tokens: List[str]) -> Dict:
    with open(filepath, "rb") as f:
        f.seek(start)
        super_chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_count = {}

    for chunk in re.split("|".join([re.escape(token) for token in special_tokens]), super_chunk):
        for reg_match in re.finditer(SPLIT_PATTERN, chunk):
            word = reg_match.group().encode("utf-8")
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count

def processs_pre_tokenize_chunk_wrapper(params):
    return pre_tokenize_chunk(*params)

## Usage
def pre_tokenize(filepath: str, special_tokens) -> Dict[str, int]:
    num_processes = NUM_PROCESSES
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        params = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            params.append((filepath, start, end, special_tokens))

    with Pool(num_processes) as p:
        word_counts = p.map(processs_pre_tokenize_chunk_wrapper, params)
    
    final_word_count = {}
    for word_count in word_counts:
        for word, count in word_count.items():
            word = tuple(word)
            if word in final_word_count:
                final_word_count[word] += count
            else:
                final_word_count[word] = count

    return final_word_count

def encode_pre_tokenize(text: str, special_tokens):
    special_tokens_split_pattern = "|".join([re.escape(token) for token in special_tokens])
    or_pattern = r"^(?:{})".format(special_tokens_split_pattern)

    chunks = re.split(special_tokens_split_pattern, text) if len(special_tokens) > 0 else [text]

    current_index = 0
    for chunk in chunks:
        for reg_match in re.finditer(SPLIT_PATTERN, chunk):
            word = reg_match.group().encode("utf-8")
            yield tuple([c.to_bytes(1) for c in word])

        if len(special_tokens) > 0:
            current_index += len(chunk)
            if current_index < len(text):
                match = re.match(or_pattern, text[current_index:])
                assert match, f"No special token at the end of chunk:\n{chunk}\n{text[current_index:]}\ncurrent index: {current_index}\n{or_pattern}"
                seperator = match.group(0)
                yield (seperator.encode("utf-8"),)
                current_index += len(seperator)

if __name__ == "__main__":
    print(pre_tokenize("tests/test_file.txt", NUM_PROCESSES))