from .data_utils import *
from .logger import create_logger
from .loss import *
from .transformer import TransformerLM
import argparse
import numpy as np
import os
from typing import Dict
import torch
import time
import shutil
from tqdm import tqdm
from .config import update_config

def training_loop(training_dataset_path: str,
                  validation_dataset_path: str,
                  checkpoint_path: str,
                  logger: str,
                  config: Dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    ### Get user dataset input
    training_dataset = np.load(training_dataset_path, mmap_mode="r")
    validation_dataset = np.load(validation_dataset_path, mmap_mode="r")

    ### Configure hyperparams
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]

    ### Configure model
    # Maybe missing context length here
    model = TransformerLM(model_config["d_model"],
                                      model_config["num_heads"],
                                      model_config["vocab_size"],
                                      model_config["context_length"],
                                      model_config["num_layers"],
                                      d_ff=model_config["d_ff"],
                                      theta=model_config["theta"], 
                                      device=device,
                                      dtype=model_config["dtype"])

    ### Configure optimizer
    lr = optimizer_config["lr"] # 
    optimizer = AdamW(model.parameters(),
                        lr,
                        optimizer_config["weight_decay"],
                        betas=optimizer_config["betas"],
                        eps=optimizer_config["eps"])

    ### Training loop
    for iteration in tqdm(range(1, training_config["number_of_iterations"] + 1)):
        # print(f" *** Starting iteration {iteration} *** ")
        optimizer.zero_grad(set_to_none=True)
        data, labels = get_batch(training_dataset, training_config["batch_size"], model_config["context_length"], device)
        guessed_labels = model(data)

        guessed_labels = rearrange(guessed_labels, "batch_size sequence_length vocab -> (batch_size sequence_length) vocab")
        labels = rearrange(labels, "batch_size sequence_length -> (batch_size sequence_length)")

        batch_loss = cross_entropy(guessed_labels, labels)
        batch_loss.backward()
        gradient_clipping(model.parameters(), optimizer_config["max_l2_norm"])

        lr = calculate_cosine_lr(iteration,
                            optimizer_config["lr"],
                            optimizer_config["lr"] / 10,
                            training_config["number_of_iterations"] * 0.05,
                            training_config["number_of_iterations"])

        optimizer.update_lr(lr)
        optimizer.step()

        del data, labels, guessed_labels

        with torch.no_grad():
            validation_data, validation_labels = get_batch(validation_dataset, training_config["batch_size"], model_config["context_length"], device)
            guessed_validation_labels = model(validation_data)

            validation_loss = cross_entropy(guessed_validation_labels, validation_labels)
            validation_perplexity = perplexity(validation_loss)

        del validation_data, validation_labels, guessed_validation_labels 

        logger.add_scalar("Loss/train", batch_loss.item(), global_step=iteration)
        logger.add_scalar("Loss/validation", validation_loss.item(), global_step=iteration)
        logger.add_scalar("Perplexity/validation", validation_perplexity.item(), global_step=iteration)

        del validation_loss, validation_perplexity, batch_loss
        
        if iteration % training_config["save_iteration_number"] == 0 :
            Checkpointing.save_checkpoint(model, optimizer, iteration, os.path.join(checkpoint_path, f"checkpoint_{iteration}.chpt"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dataset_path", type=str)
    parser.add_argument("validation_dataset_path", type=str)
    parser.add_argument("checkpointing_directory", type=str)
    parser.add_argument("logging_directory", type=str)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def run_with_default_config(training_dataset_path, validation_dataset_path, checkpointing_directory, logging_directory, user_config_path):
    shutil.rmtree(logging_directory)
    if user_config_path:
        with open(user_config_path, "r") as f:
            user_config = eval(f.read())
    else:
        user_config = None

    param_config = update_config(user_config)
    logger = create_logger(logging_directory, user_config if user_config is not None else {}) 
    training_loop(training_dataset_path, validation_dataset_path, checkpointing_directory, logger, param_config) 

def main():
    args = get_args()
    run_training_loop_with_params = lambda: run_with_default_config(args.training_dataset_path, args.validation_dataset_path, args.checkpointing_directory, args.logging_directory, args.config)
    run_training_loop_with_params()

    # job_id = TrainerClient().train(
    #     trainer=CustomTrainer(
    #         # func=run_training_loop_with_params
    #         func=dummy_loop,
    #         num_nodes=2,
    #         resources_per_node={
    #             "cpu": 1,
    #             "memory": "4Gi",
    #             # "cpu": 4,
    #             # "memory": "16Gi",
    #             # "nvidia.com/mig-1g.5gb": 1
    #         },
    #     ),
    #     runtime=TrainerClient().get_runtime("torch-distributed")
    # )

    # logs = TrainerClient().get_job_logs(name=job_id)
    # print(logs["node-0"])

if __name__ == "__main__":
    main()
