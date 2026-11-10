import argparse
from rfdetr import RFDETRBase, RFDETRNano
import torch


DATASET_PATH = "data/100k"
IMAGE_DIR = "data/100k/train" 
PRED_PATH = "data/train_predictions.json"
ATTR_PATH = "data/BDD_attributes.json"
OUTPUT_DIR = "outputs"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", default=False, help="Enable logging")
    parser.add_argument("--dataset-dir", default="data/100k", help="Root of the dataset (contains train/valid/test splits). Default: %(default)s")
    parser.add_argument("--pred-path", default="data/train_predictions.json", help="Path to the training predictions JSON. Default: %(default)s")
    parser.add_argument("--attr-path", default="data/BDD_attributes.json",help="Path to the BDD attributes JSON. Default: %(default)s")
    parser.add_argument("--output-dir", default="outputs", help="Where checkpoints and logs are written. Default: %(default)s")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")

    model = RFDETRNano(device=device)
    model_name = model.__class__.__name__

    model.train(
        dataset_dir=args.dataset_dir,
        epochs=3,
        batch_size=16,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=args.output_dir,
        progress_bar=True,
        wandb=True,
        project="jetbdd",
        run=f"{model_name}-test",
        train_log_on_step=True,
    )

if __name__ == "__main__":
    main()