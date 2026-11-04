from rfdetr import RFDETRBase, RFDETRNano


DATASET_PATH = "data/100k"
IMAGE_DIR = "data/100k/train" 
PRED_PATH = "data/train_predictions.json"
ATTR_PATH = "data/BDD_attributes.json"
OUTPUT_DIR = "outputs"

def main():
    model = RFDETRNano(device="mps")

    model.train(
        dataset_dir=DATASET_PATH,
        epochs=1,
        batch_size=16,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=OUTPUT_DIR,
        progress_bar=True,
        # resume="checkpoints/<checkpoint-file>",
    )


if __name__ == "__main__":
    main()