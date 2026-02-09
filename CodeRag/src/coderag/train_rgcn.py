import os
import json
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
import argparse
import time
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.nn import RGCNConv, global_mean_pool
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from functools import partial
import logging
from collections import defaultdict
from .config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_empty_bag():
    return {'label': 0, 'paths': []}

class MILDataset(TorchDataset):
    def __init__(self, processed_dir, graph_infos):
        self.root_dir = Path(processed_dir)
        self.bags = defaultdict(create_empty_bag)
        for info in graph_infos:
            c_file_name, label, graph_path_rel = info[1], info[2], info[3]
            self.bags[c_file_name]['paths'].append(graph_path_rel)
            if label == 1: self.bags[c_file_name]['label'] = 1
        self.c_file_names = sorted(list(self.bags.keys()))

    def __len__(self):
        return len(self.c_file_names)

    def __getitem__(self, idx):
        c_file_name = self.c_file_names[idx]
        bag_info = self.bags[c_file_name]
        bag_label = torch.tensor(bag_info['label'], dtype=torch.long)
        data_list = []
        for graph_path_rel in bag_info['paths']:
            safe_filename = graph_path_rel.replace('/', '_').replace('\\', '_') + ".pt"
            file_path = self.root_dir / safe_filename
            if file_path.exists():
                try:
                    data_list.append(torch.load(file_path, weights_only=False))
                except Exception as e:
                    logging.warning(f"Failed to load and skip file {file_path}: {e}")
        return data_list, bag_label

def mil_collate_fn(batch_of_bags):
    all_graphs, bag_labels, bag_ptr = [], [], [0]
    for bag_data_list, bag_label in batch_of_bags:
        bag_labels.append(bag_label)
        if bag_data_list:
            all_graphs.extend(bag_data_list)
        bag_ptr.append(len(all_graphs))
    super_batch = Batch.from_data_list(all_graphs) if all_graphs else Batch()
    return super_batch, torch.stack(bag_labels), torch.tensor(bag_ptr)

class RGCNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_relations, num_graph_features, hidden_dim=512, dropout=0.4):
        super(RGCNModel, self).__init__()
        self.use_graph_features = num_graph_features > 0
        self.dropout_rate = dropout
        self.in_lin = torch.nn.Linear(num_node_features, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        classifier_input_dim = hidden_dim
        if self.use_graph_features:
            self.graph_feature_processor = torch.nn.Sequential(
                torch.nn.Linear(num_graph_features, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
            processed_feature_dim = 32
            self.gate_nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + processed_feature_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid())
            classifier_input_dim += processed_feature_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.5), torch.nn.Linear(256, 2))

    def forward(self, data, return_embedding=False):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        x, edge_index, edge_type, batch = x.to(device), edge_index.to(device), edge_type.to(device), batch.to(device)
        x_init = self.in_lin(x)

        x1 = self.conv1(x_init, edge_index, edge_type)
        x1 = F.relu(self.bn1(x1))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_type) + x_init
        x2 = F.relu(self.bn2(x2))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_type) + x1
        x_final = F.relu(self.bn3(x3))

        graph_embedding = global_mean_pool(x_final, batch)

        if self.use_graph_features and hasattr(data, 'graph_level_features'):
            num_graphs_in_batch = int(batch.max()) + 1
            graph_level_features = data.graph_level_features.view(num_graphs_in_batch, -1).to(device)
            processed_graph_features = self.graph_feature_processor(graph_level_features)

            gate_input = torch.cat([graph_embedding.detach(), processed_graph_features.detach()], dim=1)
            gate_value = self.gate_nn(gate_input)

            gated_graph_embedding = graph_embedding * gate_value
            gated_processed_features = processed_graph_features * (1 - gate_value)
            combined_features = torch.cat([gated_graph_embedding, gated_processed_features], dim=1)
        else:
            combined_features = graph_embedding

        logits = self.classifier(combined_features)
        if return_embedding:
            return logits, combined_features
        return logits

worker_models = {}

def init_worker():
    worker_id = multiprocessing.current_process().pid
    if worker_id not in worker_models:
        model_name = config.codebert_model
        worker_models['tokenizer'] = RobertaTokenizer.from_pretrained(model_name)
        worker_models['codebert'] = RobertaModel.from_pretrained(model_name).to(device)
        worker_models['codebert'].eval()

def extract_node_semantic_features(graph, tokenizer, model):
    sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: int(x[0]))
    node_codes = [data.get('code', data.get('labelV', '')) for _, data in sorted_nodes]
    if not node_codes: return torch.empty((0, model.config.hidden_size))
    inputs = tokenizer(node_codes, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad(): outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def extract_graph_level_features(graph):
    try:
        features = []
        num_nodes = graph.number_of_nodes()
        if num_nodes == 0: return [0.0] * 12
        num_edges = graph.number_of_edges()
        features.extend([float(num_nodes), float(num_edges), float(nx.density(graph)) if num_nodes > 1 else 0.0])
        degrees = [d for _, d in graph.degree()]
        features.extend(
            [float(np.mean(degrees)), float(np.std(degrees)), float(np.max(degrees))] if degrees else [0.0] * 3)
        undirected_graph = graph.to_undirected(as_view=True)
        features.append(float(nx.average_clustering(undirected_graph)))
        if nx.is_connected(undirected_graph):
            avg_shortest_path = float(nx.average_shortest_path_length(undirected_graph))
        else:
            largest_cc = max(nx.connected_components(undirected_graph), key=len)
            subgraph = undirected_graph.subgraph(largest_cc)
            avg_shortest_path = float(
                nx.average_shortest_path_length(subgraph)) if subgraph.number_of_nodes() > 1 else 0.0
        features.append(avg_shortest_path)
        if num_nodes > 1:
            betweenness = list(nx.betweenness_centrality(graph, normalized=True, endpoints=False).values())
            features.extend([float(np.mean(betweenness)), float(np.max(betweenness))])
        else:
            features.extend([0.0, 0.0])
        while len(features) < 12: features.append(0.0)
        return features[:12]
    except Exception:
        return [0.0] * 12

def process_and_save_single_graph(item, project_root, output_dir, node_type_map, edge_type_map, use_graph_features):
    _, c_file_name, target, graph_path_rel = item
    safe_filename = graph_path_rel.replace('/', '_').replace('\\', '_') + ".pt"
    output_file = output_dir / safe_filename
    if output_file.exists(): return
    try:
        graph_path_abs = project_root / graph_path_rel
        if not graph_path_abs.exists(): return
        graph_nx = nx.read_graphml(graph_path_abs)
        if graph_nx.number_of_nodes() == 0: return
        tokenizer = worker_models['tokenizer']
        codebert_model = worker_models['codebert']
        node_labels = [data.get('labelV', 'UNKNOWN') for _, data in graph_nx.nodes(data=True)]
        node_type_indices = torch.tensor([node_type_map.get(label, 0) for label in node_labels], dtype=torch.long)
        node_type_one_hot = F.one_hot(node_type_indices, num_classes=len(node_type_map)).float()
        node_mapping = {node_id: i for i, node_id in enumerate(graph_nx.nodes())}
        graph_nx = nx.relabel_nodes(graph_nx, node_mapping)
        semantic_features = extract_node_semantic_features(graph_nx, tokenizer, codebert_model)
        if semantic_features.shape[0] != node_type_one_hot.shape[0]: return
        combined_features = torch.cat([semantic_features, node_type_one_hot], dim=1)
        edge_index_list = list(graph_nx.edges(data=True))
        edge_index = torch.tensor([[u, v] for u, v, _ in edge_index_list], dtype=torch.long).t().contiguous()
        edge_types = torch.tensor(
            [edge_type_map.get(data.get('labelE', 'UNKNOWN'), 0) for _, _, data in edge_index_list], dtype=torch.long)
        data = Data(x=combined_features, edge_index=edge_index, edge_type=edge_types,
                    y=torch.tensor(target, dtype=torch.long))
        if use_graph_features: data.graph_level_features = torch.tensor(extract_graph_level_features(graph_nx),
                                                                        dtype=torch.float)
        torch.save(data, output_file)
    except Exception as e:
        logging.warning(f"Error processing and skipping {graph_path_rel}: {e}")
        return

def preprocess_and_save_split(project_root, split_name, use_graph_features, num_workers, sample_percentage=100.0,
                              node_type_map=None, edge_type_map=None):
    split_file = config.data_paths[split_name]
    cache_suffix = f"_sample{int(sample_percentage)}" if sample_percentage < 100 else ""
    cache_dir_name = f"processed_{split_name}_{'withGraphFeat' if use_graph_features else 'noGraphFeat'}_graphcodebert{cache_suffix}"
    output_dir = config.data_paths['cache'] / cache_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'r', encoding='utf-8') as f:
        graph_infos = json.load(f)
    if sample_percentage < 100:
        original_count = len(graph_infos)
        num_to_sample = int(original_count * (sample_percentage / 100.0))
        if num_to_sample == 0 and original_count > 0: num_to_sample = 1
        graph_infos.sort(key=lambda x: x[0])
        graph_infos = graph_infos[:num_to_sample]
        logging.info(f"Deterministic sampling enabled: selecting first {len(graph_infos)} of {original_count} graphs ({sample_percentage}%) for {split_name} set.")
    is_training_split = (split_name == 'train')
    if is_training_split:
        vocab_file = output_dir / "_vocab.json"
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            node_type_map, edge_type_map = vocab['node_type_map'], vocab['edge_type_map']
        else:
            all_node_labels, all_edge_labels = set(), set()
            for item in tqdm(graph_infos, desc=f"Scanning {split_name} types"):
                try:
                    graph_path_abs = project_root / item[3]
                    if not graph_path_abs.exists(): continue
                    graph_nx = nx.read_graphml(graph_path_abs)
                    for _, data in graph_nx.nodes(data=True): all_node_labels.add(data.get('labelV', 'UNKNOWN'))
                    for _, _, data in graph_nx.edges(data=True): all_edge_labels.add(data.get('labelE', 'UNKNOWN'))
                except Exception:
                    pass
            node_type_map = {label: i for i, label in enumerate(sorted(list(all_node_labels)))}
            edge_type_map = {label: i for i, label in enumerate(sorted(list(all_edge_labels)))}
            with open(vocab_file, 'w') as f:
                json.dump({'node_type_map': node_type_map, 'edge_type_map': edge_type_map}, f)
    if node_type_map is None or edge_type_map is None: raise ValueError("Vocabulary not found")
    worker_func = partial(process_and_save_single_graph, project_root=project_root, output_dir=output_dir,
                          node_type_map=node_type_map, edge_type_map=edge_type_map,
                          use_graph_features=use_graph_features)
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        list(tqdm(pool.imap_unordered(worker_func, graph_infos, chunksize=100), total=len(graph_infos),
                  desc=f"Extracting features for {split_name} set"))
    if is_training_split:
        return str(output_dir), node_type_map, edge_type_map, graph_infos
    else:
        return str(output_dir), graph_infos

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, path='checkpoint.pt'):
        self.patience, self.min_delta, self.path = patience, min_delta, path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.save_checkpoint(model)
            logging.info(f'Validation score improved ({self.best_score:.4f}). Saving model to {self.path}')
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)

def train_epoch_mil(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_bags_processed = 0, 0
    for super_bag, bag_labels, bag_ptr in tqdm(loader, desc="Training"):
        super_bag, bag_labels = super_bag.to(device), bag_labels.to(device)
        if not hasattr(super_bag, 'x') or super_bag.x is None: continue
        optimizer.zero_grad()
        graph_logits = model(super_bag)
        logits_for_non_empty_bags, labels_for_non_empty_bags = [], []
        num_bags_in_batch = len(bag_ptr) - 1
        for i in range(num_bags_in_batch):
            start_idx, end_idx = bag_ptr[i], bag_ptr[i + 1]
            if start_idx != end_idx:
                positive_logits_in_bag = graph_logits[start_idx:end_idx, 1]
                aggregated_logit, _ = torch.max(positive_logits_in_bag, dim=0)
                logits_for_non_empty_bags.append(aggregated_logit)
                labels_for_non_empty_bags.append(bag_labels[i])
        if not logits_for_non_empty_bags: continue
        aggregated_logits = torch.stack(logits_for_non_empty_bags)
        final_logits = torch.stack([torch.zeros_like(aggregated_logits), aggregated_logits], dim=1)
        loss = criterion(final_logits, torch.stack(labels_for_non_empty_bags))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels_for_non_empty_bags)
        total_bags_processed += len(labels_for_non_empty_bags)
    return total_loss / total_bags_processed if total_bags_processed > 0 else 0

def evaluate_mil(model, loader, is_test=False):
    model.eval()
    all_bag_preds, all_bag_labels, all_bag_probs = [], [], []
    desc = "Testing" if is_test else "Evaluating"
    with torch.no_grad():
        for super_bag, bag_labels, bag_ptr in tqdm(loader, desc=desc):
            super_bag, bag_labels = super_bag.to(device), bag_labels.to(device)
            if not hasattr(super_bag, 'x') or super_bag.x is None:
                num_bags_in_batch = len(bag_labels)
                all_bag_preds.extend([0] * num_bags_in_batch)
                all_bag_probs.extend([0.0] * num_bags_in_batch)
                all_bag_labels.extend(bag_labels.cpu().numpy())
                continue
            graph_logits = model(super_bag)
            num_bags_in_batch = len(bag_ptr) - 1
            for i in range(num_bags_in_batch):
                start_idx, end_idx = bag_ptr[i], bag_ptr[i + 1]
                bag_graph_logits = graph_logits[start_idx:end_idx]
                if bag_graph_logits.numel() == 0:
                    aggregated_prob = torch.tensor(0.0, device=device)
                else:
                    bag_graph_probs = F.softmax(bag_graph_logits, dim=1)
                    positive_probs_in_bag = bag_graph_probs[:, 1]
                    aggregated_prob, _ = torch.max(positive_probs_in_bag, dim=0)
                prediction = 1 if aggregated_prob > 0.5 else 0
                all_bag_preds.append(prediction)
                all_bag_probs.append(aggregated_prob.cpu().item())
            all_bag_labels.extend(bag_labels.cpu().numpy())
    return all_bag_labels, all_bag_preds, all_bag_probs

def main():
    parser = argparse.ArgumentParser(description="R-GCN Vulnerability Detection Model (MIL)")
    parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    parser.add_argument('--experiment_name', type=str, default="RGCN_Final_Train", help="Experiment name")
    parser.add_argument('--no_graph_features', action='store_true', help="Do not use the 12 handcrafted graph-level features")
    parser.add_argument('--preprocess_workers', type=int, default=4, help="Number of parallel processes for data preprocessing")
    parser.add_argument('--dataloader_workers', type=int, default=4, help="Number of data loading workers for DataLoader during training")
    parser.add_argument('--lr', type=float, default=0.000121, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='GNN hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.2328, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample_percentage', type=float, default=100.0, help="Percentage of data to use")
    parser.add_argument('--project_root', type=str, default="", help="Project root directory")
    args = parser.parse_args()

    config.load_from_args(args)
    args.project_root = str(config.project_root)
    args.use_graph_features = not args.no_graph_features
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = config.project_root / "src" / "outputs" / f"{timestamp}_{args.experiment_name}_sample{int(args.sample_percentage)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(output_dir / 'training.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    logging.info(f"All outputs will be saved to: {output_dir}")
    logging.info(f"Current configuration: {vars(args)}")

    logging.info("\n--- [1/5] Preprocessing dataset splits ---")
    train_dir, node_type_map, edge_type_map, train_graph_infos = preprocess_and_save_split(
        config.project_root, 'train', args.use_graph_features, args.preprocess_workers, args.sample_percentage)
    val_dir, val_graph_infos = preprocess_and_save_split(
        config.project_root, 'val', args.use_graph_features, args.preprocess_workers, args.sample_percentage,
        node_type_map, edge_type_map)
    test_dir, test_graph_infos = preprocess_and_save_split(
        config.project_root, 'test', args.use_graph_features, args.preprocess_workers, args.sample_percentage,
        node_type_map, edge_type_map)

    logging.info("\n--- [2/5] Creating on-demand loading MIL Dataset and DataLoader ---")
    train_dataset = MILDataset(train_dir, train_graph_infos)
    val_dataset = MILDataset(val_dir, val_graph_infos)
    test_dataset = MILDataset(test_dir, test_graph_infos)
    logging.info(f"  - Training set (C files) size: {len(train_dataset)}")
    logging.info(f"  - Validation set (C files) size: {len(val_dataset)}")
    logging.info(f"  - Test set (C files) size: {len(test_dataset)}")
    if len(train_dataset) == 0: logging.error("Training set is empty, cannot continue."); exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.dataloader_workers, collate_fn=mil_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.dataloader_workers, collate_fn=mil_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.dataloader_workers, collate_fn=mil_collate_fn, pin_memory=True)

    logging.info("--- [3/5] Initializing model, optimizer, loss function ---")
    num_node_features = 768 + len(node_type_map)
    num_relations = len(edge_type_map)
    num_graph_features = 12 if args.use_graph_features else 0
    model = RGCNModel(num_node_features, num_relations, num_graph_features, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    class_counts = np.bincount([bag['label'] for bag in train_dataset.bags.values()])
    if len(class_counts) < 2: class_counts = np.append(class_counts, [0] * (2 - len(class_counts)))
    class_weights = torch.tensor([sum(class_counts) / (2 * c) if c > 0 else 1.0 for c in class_counts],
                                 dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    logging.info(f"Calculated file-level class weights: {class_weights}")
    best_model_path = output_dir / "best_model_mil.pt"
    early_stopper = EarlyStopping(patience=args.patience, path=str(best_model_path))

    logging.info("--- [4/5] Starting final training ---")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch_mil(model, train_loader, optimizer, criterion)
        if len(val_dataset) == 0:
            logging.info(f"Epoch {epoch:03d}/{args.epochs} | Loss: {avg_loss:.4f} | Validation set is empty, skipping evaluation");
            continue

        labels, preds, probs = evaluate_mil(model, val_loader)
        if not labels:
            logging.warning(f"Epoch {epoch:03d}/{args.epochs} | Validation set returned empty results, skipping this round of evaluation.")
            continue

        val_mcc = matthews_corrcoef(labels, preds)
        val_acc = accuracy_score(labels, preds)
        val_pre = precision_score(labels, preds, zero_division=0)
        val_rec = recall_score(labels, preds, zero_division=0)
        val_f1 = f1_score(labels, preds, zero_division=0)
        val_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

        log_message = (
            f"Epoch {epoch:03d}/{args.epochs} | Loss: {avg_loss:.4f} | "
            f"Val MCC: {val_mcc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
        )
        logging.info(log_message)

        early_stopper(val_mcc, model)
        scheduler.step()
        if early_stopper.early_stop:
            logging.info("Early stopping triggered, training finished.")
            break

    logging.info("\n--- [5/5] Training complete, performing final evaluation on the test set ---")
    if best_model_path.exists() and len(test_dataset) > 0:
        logging.info(f"Loading best performing model on validation set: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

        test_labels, test_preds, test_probs = evaluate_mil(model, test_loader, is_test=True)

        if test_labels:
            final_metrics = {
                "accuracy": accuracy_score(test_labels, test_preds),
                "precision": precision_score(test_labels, test_preds, zero_division=0),
                "recall": recall_score(test_labels, test_preds, zero_division=0),
                "f1_score": f1_score(test_labels, test_preds, zero_division=0),
                "auc": roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.0,
                "mcc": matthews_corrcoef(test_labels, test_preds)
            }
            try:
                tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
                final_metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
            except ValueError:
                pass

            logging.info("\n--- Final Test Set File-Level Evaluation Results ---")
            for metric, value in final_metrics.items():
                logging.info(f"{metric.capitalize():<12}: {value:.4f}")

            results_file = output_dir / "final_results.json"
            args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"args": args_dict, "final_metrics": final_metrics}, f, indent=4)
            logging.info(f"\nEvaluation results and configuration saved to: {results_file}")
        else:
            logging.warning("Test set evaluation did not produce any results, skipped.")
    else:
        logging.warning("No best model was saved during training, or the test set is empty, skipping final evaluation.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    main()