import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import time
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import torch
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import logging
from .config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TransformerVulnerabilityModel(torch.nn.Module):
    def __init__(self, model_name="microsoft/graphcodebert-base", dropout=0.3):
        super(TransformerVulnerabilityModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder.config.hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        if return_embedding:
            return logits, cls_embedding
        else:
            return logits, None

class CodeSequenceDataset(TorchDataset):
    def __init__(self, data_infos, tokenizer, code_root_dir, max_length=512):
        self.tokenizer = tokenizer
        self.code_root = Path(code_root_dir)
        self.max_length = max_length
        self.samples = []
        processed_files = set()

        for info in tqdm(data_infos, desc="Loading and deduplicating data"):
            c_file_name, label = info[1], info[2]
            if c_file_name in processed_files:
                continue
            processed_files.add(c_file_name)

            code_path = self.code_root / c_file_name
            if code_path.exists():
                try:
                    with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    self.samples.append({'code': code, 'label': label})
                except Exception as e:
                    logging.warning(f"Failed to read file {code_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        code = sample['code']
        label = sample['label']

        inputs = self.tokenizer.encode_plus(
            code,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, is_test=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    desc = "Testing" if is_test else "Evaluating"
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return all_labels, all_preds, all_probs

def main():
    parser = argparse.ArgumentParser(description="Transformer Vulnerability Detection Model")
    parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    parser.add_argument('--experiment_name', type=str, default="Transformer_FineTune", help="Experiment name")
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (common value for fine-tuning)')
    parser.add_argument('--epochs', type=int, default=5, help='Maximum number of epochs (fine-tuning usually does not require many epochs)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (adjust according to VRAM)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--project_root', type=str, default="", help="Project root directory")

    args = parser.parse_args()
    config.load_from_args(args)
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = config.project_root / "src" / "outputs" / f"{timestamp}_{args.experiment_name}_{config.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(output_dir / 'training.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    logging.info(f"All outputs will be saved to: {output_dir}")
    logging.info(f"Current configuration: {vars(args)}")

    logging.info("\n--- [1/5] Loading tokenizer and dataset ---")
    tokenizer = RobertaTokenizer.from_pretrained(config.codebert_model)
    code_root_dir = config.data_paths['functions']

    with open(config.data_paths['train'], 'r') as f:
        train_infos = json.load(f)
    with open(config.data_paths['val'], 'r') as f:
        val_infos = json.load(f)
    with open(config.data_paths['test'], 'r') as f:
        test_infos = json.load(f)

    train_dataset = CodeSequenceDataset(train_infos, tokenizer, code_root_dir)
    val_dataset = CodeSequenceDataset(val_infos, tokenizer, code_root_dir)
    test_dataset = CodeSequenceDataset(test_infos, tokenizer, code_root_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logging.info(f"  - Training set size: {len(train_dataset)}")
    logging.info(f"  - Validation set size: {len(val_dataset)}")
    logging.info(f"  - Test set size: {len(test_dataset)}")

    logging.info("\n--- [2/5] Initializing model, optimizer, loss function ---")
    model = TransformerVulnerabilityModel(model_name=config.codebert_model).to(device)

    train_labels = [s['label'] for s in train_dataset.samples]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([sum(class_counts) / (2 * c) if c > 0 else 1.0 for c in class_counts],
                                 dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    logging.info(f"Calculated class weights: {class_weights}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    logging.info("\n--- [3/5] Starting training ---")
    best_val_mcc = -1
    patience_counter = 0
    best_model_path = output_dir / "best_transformer_model.pt"

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        labels, preds, probs = evaluate(model, val_loader)

        val_mcc = matthews_corrcoef(labels, preds)
        val_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        val_f1 = f1_score(labels, preds, zero_division=0)

        logging.info(
            f"Epoch {epoch:02d}/{args.epochs} | Loss: {avg_loss:.4f} | Val MCC: {val_mcc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Validation MCC improved to {best_val_mcc:.4f}. Saving model to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"EarlyStopping counter: {patience_counter} out of {args.patience}")
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered, ending training.")
                break

    logging.info("\n--- [4/5] Training complete, performing final evaluation on the test set ---")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        test_labels, test_preds, test_probs = evaluate(model, test_loader, is_test=True)

        final_metrics = {
            "accuracy": accuracy_score(test_labels, test_preds),
            "precision": precision_score(test_labels, test_preds, zero_division=0),
            "recall": recall_score(test_labels, test_preds, zero_division=0),
            "f1_score": f1_score(test_labels, test_preds, zero_division=0),
            "auc": roc_auc_score(test_labels, test_probs),
            "mcc": matthews_corrcoef(test_labels, test_preds)
        }
        logging.info("\n--- Final Test Set Evaluation Results ---")
        for metric, value in final_metrics.items():
            logging.info(f"{metric.capitalize():<12}: {value:.4f}")

        results_file = output_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump({"args": vars(args), "final_metrics": final_metrics}, f, indent=4)
        logging.info(f"\nEvaluation results and configuration saved to: {results_file}")
    else:
        logging.warning("No best model was saved, skipping final evaluation.")

if __name__ == "__main__":
    main()