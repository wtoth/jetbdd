import os
import argparse
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
import torch


DATASET_PATH = "data/100k"
IMAGE_DIR = "data/100k/train" 
PRED_PATH = "data/train_predictions.json"
ATTR_PATH = "data/BDD_attributes.json"
OUTPUT_DIR = "outputs"
MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="nano", help="RF-DETR model size. Default: %(default)s")
    parser.add_argument("--log", action="store_true", default=False, help="Enable logging")
    parser.add_argument("--dataset-dir", default="data/100k", help="Root of the dataset (contains train/valid/test splits). Default: %(default)s")
    parser.add_argument("--pred-path", default="data/train_predictions.json", help="Path to the training predictions JSON. Default: %(default)s")
    parser.add_argument("--attr-path", default="data/BDD_attributes.json",help="Path to the BDD attributes JSON. Default: %(default)s")
    parser.add_argument("--output-dir", default="outputs", help="Where checkpoints and logs are written. Default: %(default)s")
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Activate gradient checkpointing reducing on-gpu memory requirements")
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

    model = MODELS[args.model](device=device, gradient_checkpointing=args.gradient_checkpointing)
    model_name = model.__class__.__name__

    # the best checkpoint is stored in "outputs/checkpoint_best_total.pth"
    last_ckpt = os.path.join(args.output_dir, "last.ckpt")
    resume = last_ckpt if os.path.isfile(last_ckpt) else None
    print(f"Resuming from {resume}" if resume else "No checkpoint found, starting fresh")
 
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        resume=resume,
        progress_bar=True,
        wandb=args.wandb,
        project="jetbdd",
        run=f"{model_name}",
        train_log_on_step=True,
        devices="auto"
    )

if __name__ == "__main__":
    main()