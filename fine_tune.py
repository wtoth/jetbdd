from rfdetr import RFDETRBase, RFDETRNano
import torch


DATASET_PATH = "data/100k"
IMAGE_DIR = "data/100k/train" 
PRED_PATH = "data/train_predictions.json"
ATTR_PATH = "data/BDD_attributes.json"
OUTPUT_DIR = "outputs"

def main():
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
        dataset_dir=DATASET_PATH,
        epochs=3,
        batch_size=16,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=OUTPUT_DIR,
        progress_bar=True,
        wandb=True,
        project="jetbdd",
        run=f"{model_name}-test",
        train_log_on_step=True,
    )

if __name__ == "__main__":
    main()