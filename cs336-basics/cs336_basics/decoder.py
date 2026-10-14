from typing import List, Dict, Tuple
import numpy as np
import torch
from .transformer import softmax, TransformerLM
from .data_utils import Checkpointing
from .config import update_config
from .tokenizer import Tokenizer
import argparse
import os 
import itertools
from tqdm import tqdm

class Decoder:
    def __init__(self, model: torch.nn.Module, stop_tokens: List[int] = None, tempature: int = 1, sampling_threshold: int = None):
        self.model = model
        self.tempature = tempature
        self.sampling_threshold = sampling_threshold
        if stop_tokens:
            self.stop_tokens = stop_tokens
        else:
            self.stop_tokens = []

    def decode_token(self, input_sequence: torch.Tensor) -> int:

        with torch.inference_mode():
            logits = self.model(input_sequence)

        if self.tempature:
            logits = logits / self.tempature
            
        probabilities = softmax(logits, -1)[-1, :]

        if self.sampling_threshold:
            torch.sort(probabilities, dim=-1, descending=True)
            sum = 0
            cutting_index = 0
            while sum < self.sampling_threshold:
                sum += probabilities[cutting_index]
                cutting_index += 1

            probabilities = probabilities[:cutting_index]
            probabilities = probabilities / torch.sum(probabilities, dim=-1)

        return torch.multinomial(probabilities, num_samples=1)

    def decode_sequence(self, input_sequence: torch.Tensor, max_generated_tokens: int) -> torch.Tensor:
        current_sequence = input_sequence
        for _ in tqdm(range(max_generated_tokens)):
            new_token = self.decode_token(current_sequence)
            current_sequence = torch.cat((current_sequence, torch.Tensor([new_token]).to(current_sequence.device))).long()
            if new_token in self.stop_tokens:
                break
        return current_sequence

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("tokenizer", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    if args.config:
        with open(args.config, "r") as f:
            user_config = eval(f.read())
    else:
        user_config = None

    param_config = update_config(user_config)
    model_config = param_config["model"]

    tokenizer = Tokenizer.from_files(os.path.join(args.tokenizer, "vocab.pkl"), os.path.join(args.tokenizer, "merges.pkl"), ["<|endoftext|>"])
    encoded_text = torch.tensor(tokenizer.encode(args.text)).to(device)

    model = TransformerLM(model_config["d_model"],
                                      model_config["num_heads"],
                                      model_config["vocab_size"],
                                      model_config["context_length"],
                                      model_config["num_layers"],
                                      d_ff=model_config["d_ff"],
                                      theta=model_config["theta"], 
                                      device=device,
                                      dtype=model_config["dtype"])
    model.eval()
    
    Checkpointing.load_checkpoint(args.checkpoint, model)

    stop_tokens = tokenizer.encode("<|endoftext|>")
    decoder = Decoder(model, tempature=0.1, sampling_threshold=0.5, stop_tokens=stop_tokens)
    print("Generating input...")
    generated = decoder.decode_sequence(encoded_text, max_generated_tokens=110)
    print(generated)
    print(tokenizer.decode(generated))

if __name__ == "__main__":
    main()
