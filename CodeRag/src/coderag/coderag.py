import os
import json
import time
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
import argparse
import re
from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import traceback
import logging
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)
from itertools import cycle
from pathlib import Path
from pycparser import c_parser, c_ast
from .config import config
from .train_rgcn import RGCNModel, extract_graph_level_features, extract_node_semantic_features
from .train_transformer import TransformerVulnerabilityModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def get_tokens(code: str) -> set:
    tokens = re.split(r'[^a-zA-Z0-9_]+', code)
    return set(filter(None, tokens))

def get_ast_sequence(code: str) -> List[str]:
    parser = c_parser.CParser()
    try:
        ast = parser.parse(f"void func_wrapper() {{ {code} }}", filename='<stdin>')
        node_types = []
        class NodeVisitor(c_ast.NodeVisitor):
            def visit(self, node):
                node_types.append(type(node).__name__)
                self.generic_visit(node)
        NodeVisitor().visit(ast)
        return node_types
    except Exception:
        return []

def calculate_lexical_similarity(tokens1: set, tokens2: set) -> float:
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(s1: List, s2: List) -> int:
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if not s2: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_syntactic_similarity(seq1: List[str], seq2: List[str]) -> float:
    len_sum = len(seq1) + len(seq2)
    if len_sum == 0: return 1.0
    lev_dist = levenshtein_distance(seq1, seq2)
    return (len_sum - lev_dist) / len_sum

class HybridICLExampleSelector:
    def __init__(self, context_data: List[Dict]):
        logging.info("Initializing Hybrid ICL Selector...")
        for item in tqdm(context_data, desc="Preprocessing ICL context"):
            item['tokens'] = set(item['tokens'])
        self.vuln_context = [d for d in context_data if d.get('label') == 1]
        self.safe_context = [d for d in context_data if d.get('label') == 0]
        self.vuln_semantic_embeddings = np.array(
            [d['semantic_embedding'] for d in self.vuln_context]) if self.vuln_context else np.array([])
        self.safe_semantic_embeddings = np.array(
            [d['semantic_embedding'] for d in self.safe_context]) if self.safe_context else np.array([])
        logging.info(f"Hybrid ICL Selector initialized: {len(self.vuln_context)} vulnerable samples, {len(self.safe_context)} safe samples.")

    def _select_single_best_example(self, target_semantic_embedding: np.ndarray, target_gnn_embedding: np.ndarray,
                                    target_tokens: set, target_ast_seq: list, context_pool: List[Dict],
                                    semantic_embeddings_pool: np.ndarray, k: int = 10, w_gnn: float = 0.5,
                                    w_lex: float = 0.3, w_syn: float = 0.2) -> Dict:
        if not context_pool or semantic_embeddings_pool.shape[0] == 0: return None
        similarities = cosine_similarity(target_semantic_embedding.reshape(1, -1), semantic_embeddings_pool)[0]
        top_k_indices = [idx for idx in np.argsort(similarities)[-k:][::-1] if similarities[idx] > 0]
        if not top_k_indices: return None
        best_candidate, max_mixed_score = None, -1.0
        for idx in top_k_indices:
            candidate = context_pool[idx]
            candidate_gnn_embedding = np.array(candidate['gnn_embedding']).reshape(1, -1)
            gnn_sim = cosine_similarity(target_gnn_embedding.reshape(1, -1), candidate_gnn_embedding)[0][0]
            lex_sim = calculate_lexical_similarity(target_tokens, candidate['tokens'])
            syn_sim = calculate_syntactic_similarity(target_ast_seq, candidate['ast_sequence'])
            mixed_score = w_gnn * gnn_sim + w_lex * lex_sim + w_syn * syn_sim
            if mixed_score > max_mixed_score:
                max_mixed_score = mixed_score
                best_candidate = candidate
        return best_candidate

    def select_examples(self, target_code: str, target_semantic_embedding: np.ndarray,
                        target_gnn_embedding: np.ndarray) -> List[Dict]:
        target_tokens = get_tokens(target_code)
        target_ast_seq = get_ast_sequence(target_code)
        examples = []
        best_vuln = self._select_single_best_example(target_semantic_embedding, target_gnn_embedding, target_tokens,
                                                     target_ast_seq, self.vuln_context, self.vuln_semantic_embeddings)
        if best_vuln: examples.append(best_vuln)
        best_safe = self._select_single_best_example(target_semantic_embedding, target_gnn_embedding, target_tokens,
                                                     target_ast_seq, self.safe_context, self.safe_semantic_embeddings)
        if best_safe: examples.append(best_safe)
        return examples

class CodeRagDetector:
    def __init__(self, gnn_model, transformer_model, codebert_for_nodes, tokenizer, context_data: List[Dict],
                 node_type_map, edge_type_map,
                 transformer_decision_threshold: float, transformer_rescue_threshold: float):
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        self.codebert_for_nodes = codebert_for_nodes
        self.tokenizer = tokenizer
        self.node_type_map = node_type_map
        self.edge_type_map = edge_type_map
        self.icl_selector = HybridICLExampleSelector(context_data)
        self.transformer_decision_threshold = transformer_decision_threshold
        self.transformer_rescue_threshold = transformer_rescue_threshold
        self.llm_client = None
        logging.info(f"CodeRag Detector initialized. Decision threshold: {self.transformer_decision_threshold}, Rescue threshold: {self.transformer_rescue_threshold}")

    def detect(self, graph: nx.Graph, code: str, llm_client: OpenAI) -> Dict[str, Any]:
        self.llm_client = llm_client
        inputs = self.tokenizer(code, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits, semantic_embedding_tensor = self.transformer_model(inputs['input_ids'], inputs['attention_mask'],
                                                                       return_embedding=True)
            probs = F.softmax(logits, dim=1)
            transformer_probability = probs[0][1].item()

        if transformer_probability < self.transformer_decision_threshold:
            final_prediction, llm_prediction, llm_response, prompt = 0, 0, "Transformer low confidence, safe.", ""
        elif transformer_probability > self.transformer_rescue_threshold:
            final_prediction, llm_prediction, llm_response, prompt = 1, 1, "Transformer high confidence, vulnerable.", ""
        else:
            data = self._preprocess_graph(graph)
            with torch.no_grad():
                _, gnn_embedding_tensor = self.gnn_model(data, return_embedding=True)

            semantic_embedding = semantic_embedding_tensor.cpu().numpy()
            gnn_embedding = gnn_embedding_tensor.cpu().numpy()

            context_examples = self.icl_selector.select_examples(code, semantic_embedding, gnn_embedding)
            prompt = self._create_prompt(code, transformer_probability, context_examples)
            llm_prediction, llm_response = self._get_llm_prediction_with_retry(prompt)

            if llm_prediction == 0 and transformer_probability > self.transformer_rescue_threshold:
                final_prediction = 1
                logging.info(f"  [Transformer RESCUE] Transformer prob {transformer_probability:.4f} > {self.transformer_rescue_threshold}. Overriding LLM SAFE verdict.")
            elif llm_prediction == -1:
                final_prediction = 1
            else:
                final_prediction = llm_prediction

        return {"transformer_probability": transformer_probability, "llm_prediction": llm_prediction,
                "final_prediction": final_prediction, "llm_full_response": llm_response, "prompt": prompt}

    def _preprocess_graph(self, graph_nx: nx.Graph) -> Data:
        node_labels = [data.get('labelV', 'UNKNOWN') for _, data in graph_nx.nodes(data=True)]
        node_type_indices = torch.tensor([self.node_type_map.get(label, 0) for label in node_labels], dtype=torch.long)
        node_type_one_hot = F.one_hot(node_type_indices, num_classes=len(self.node_type_map)).float()
        semantic_features = extract_node_semantic_features(graph_nx, self.tokenizer, self.codebert_for_nodes).cpu()
        if semantic_features.shape[0] != node_type_one_hot.shape[0]:
            raise ValueError(f"Node semantic features ({semantic_features.shape[0]}) and type features ({node_type_one_hot.shape[0]}) mismatch!")
        combined_features = torch.cat([semantic_features, node_type_one_hot], dim=1)

        node_mapping = {node_id: i for i, node_id in enumerate(sorted(graph_nx.nodes(), key=int))}
        edge_u = [node_mapping[str(u)] for u, v, _ in graph_nx.edges(data=True)]
        edge_v = [node_mapping[str(v)] for u, v, _ in graph_nx.edges(data=True)]
        edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long).contiguous()
        edge_types = torch.tensor(
            [self.edge_type_map.get(data.get('labelE', 'UNKNOWN'), 0) for _, _, data in graph_nx.edges(data=True)],
            dtype=torch.long)

        data = Data(x=combined_features, edge_index=edge_index, edge_type=edge_types)
        data.graph_level_features = torch.tensor(extract_graph_level_features(graph_nx), dtype=torch.float)
        return data

    def _create_prompt(self, code: str, transformer_probability: float, context_examples: List[Dict]) -> str:
        prob_percent = transformer_probability * 100
        ai_section = (f"### AI Preliminary Analysis (Transformer)\n- **Preliminary Conclusion**: Suspicious, needs expert review\n- **AI Model Vulnerability Probability**: {prob_percent:.2f}%")
        context_section = "### Similar Cases Retrieved via Multimodal Search (For Reference)\n"
        vuln_ex = next((ex for ex in context_examples if ex['label'] == 1), None)
        safe_ex = next((ex for ex in context_examples if ex['label'] == 0), None)
        if vuln_ex: context_section += f"#### [SIMILAR VULNERABLE EXAMPLE]\n```c\n{vuln_ex['code']}\n```\n\n"
        if safe_ex: context_section += f"#### [SIMILAR SAFE EXAMPLE]\n```c\n{safe_ex['code']}\n```\n\n"
        task_section = (f"### Your Task\n"
                        f"1. **Carefully compare** the 'Code to Audit' with the 'Similar Safe Example'. Pay special attention to the **key differences** between them, as this difference is likely the core of the vulnerability.\n"
                        f"2. Combine the attack patterns provided by the 'Similar Vulnerable Example' to make a final judgment on the 'Code to Audit'.\n"
                        f"3. Please respond strictly in the following format, without any additional explanation:\n\n"
                        f"[VERDICT] VULNERABLE or SAFE\n[REASON] A brief sentence explaining your core judgment basis.")
        return (f"You are a world-class C/C++ code security audit expert.\n\n"
                f"{ai_section}\n\n{context_section}### Code to Audit\n```c\n{code}\n```\n\n{task_section}")

    def _get_llm_prediction_with_retry(self, prompt: str, max_retries: int = 5, initial_delay: int = 5) -> Tuple[int, str]:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(model=config.llm_model,
                                                                   messages=[{"role": "user", "content": prompt}],
                                                                   temperature=0.0, max_tokens=200, timeout=90.0)
                response_text = response.choices[0].message.content.strip()
                if "[VERDICT] VULNERABLE" in response_text: return 1, response_text
                if "[VERDICT] SAFE" in response_text: return 0, response_text
                return -1, f"LLM response format unparseable: {response_text}"
            except (APITimeoutError, RateLimitError, APIStatusError) as e:
                logging.warning(f"LLM API call error (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds... Error: {type(e).__name__} - {e}")
            except Exception as e:
                logging.warning(f"LLM API call unknown error (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay + random.uniform(0, 3))
            delay = min(delay * 2, 60)
        return -1, f"LLM API call failed completely after {max_retries} retries."

def load_and_cache_context_data(project_root: Path, cache_dir: Path, gnn_model, codebert_for_nodes, tokenizer,
                                node_type_map, edge_type_map,
                                context_limit: int, force_recache: bool) -> List[Dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hybrid_icl_context_limit{context_limit}.json"

    if not force_recache and cache_file.exists():
        logging.info(f"Loading ICL context data from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f: return json.load(f)

    logging.warning(f"Cache not found or forced refresh. Preprocessing samples from train set...")

    train_json_path = config.data_paths['train']
    with open(train_json_path, 'r', encoding='utf-8') as f:
        all_train_data = json.load(f)

    unique_files = {item[1]: item for item in all_train_data}
    unique_train_items = list(unique_files.values())
    vuln_data = [item for item in unique_train_items if item[2] == 1]
    safe_data = [item for item in unique_train_items if item[2] == 0]

    if context_limit == 0:
        context_data_list = vuln_data + safe_data
        logging.info(f"context_limit=0, using all {len(context_data_list)} deduplicated training samples as ICL context.")
    else:
        sample_size = context_limit // 2
        context_data_list = (random.sample(vuln_data, min(sample_size, len(vuln_data))) +
                             random.sample(safe_data, min(sample_size, len(safe_data))))
        logging.info(f"Sampling up to {context_limit} samples from training set as ICL context.")

    context_data = []
    temp_detector = CodeRagDetector(gnn_model, None, codebert_for_nodes, tokenizer, [], node_type_map, edge_type_map, 0, 0)

    for item in tqdm(context_data_list, desc="Preprocessing ICL context (all modalities)"):
        try:
            _, c_fname, label, graph_path_rel = item
            code_path = config.data_paths['functions'] / c_fname
            graph_path = project_root / graph_path_rel

            if not (graph_path.exists() and code_path.exists()): continue
            graph = nx.read_graphml(graph_path)
            if graph.number_of_nodes() == 0: continue
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            data_obj = temp_detector._preprocess_graph(graph)
            with torch.no_grad():
                _, gnn_embedding_tensor = gnn_model(data_obj, return_embedding=True)
            if gnn_embedding_tensor is None: continue

            inputs = tokenizer(code, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
            with torch.no_grad():
                semantic_outputs = codebert_for_nodes(**inputs)
                semantic_embedding = semantic_outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()

            context_data.append({
                'c_file_name': c_fname, 'code': code, 'label': label,
                'gnn_embedding': gnn_embedding_tensor.cpu().numpy().flatten().tolist(),
                'semantic_embedding': semantic_embedding,
                'tokens': list(get_tokens(code)),
                'ast_sequence': get_ast_sequence(code)
            })
        except Exception as e:
            logging.error(f"Severe error processing ICL sample {c_fname}: {type(e).__name__}: {e}")
            logging.error(traceback.format_exc())

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f, indent=2, ensure_ascii=False)
    logging.info(f"ICL context data preprocessed and cached to: {cache_file}")
    return context_data

def process_single_item(args: tuple) -> Dict:
    item_data, detector, llm_client = args
    try:
        graph_path, code_path = Path(item_data['graph_path']), Path(item_data['code_path'])
        if not graph_path.exists() or not code_path.exists():
            return {**item_data, 'error': "File not found"}
        graph = nx.read_graphml(graph_path)
        if graph.number_of_nodes() == 0:
            return {**item_data, 'error': "Empty graph"}
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        result = detector.detect(graph, code, llm_client)
        result.update(item_data)
        return result
    except Exception:
        error_message = f"Severe error: {traceback.format_exc()}"
        logging.error(f"Severe error processing file {item_data['c_file_name']}:\n{error_message}")
        return {**item_data, 'error': error_message}

def calculate_metrics_summary(results: List[Dict]) -> Dict:
    filtered_results = [res for res in results if 'error' not in res and res.get('final_prediction') is not None]
    if not filtered_results: return {}
    y_true = [res['true_label'] for res in filtered_results]
    y_pred = [res['final_prediction'] for res in filtered_results]
    y_prob_transformer = [res['transformer_probability'] for res in filtered_results]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_transformer_based": roc_auc_score(y_true, y_prob_transformer) if len(set(y_true)) > 1 else 0.0
    }
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    except ValueError:
        pass
    return metrics

def main():
    limit_test_samples = 0
    limit_context_samples = 0

    parser = argparse.ArgumentParser(description="Hybrid ICL-CodeRag Vulnerability Detection Framework")
    parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    parser.add_argument('--workers', type=int, default=4, help="Concurrent API requests")
    parser.add_argument('--transformer_decision_threshold', type=float, default=0.16, help="Transformer initial screening decision threshold")
    parser.add_argument('--transformer_rescue_threshold', type=float, default=0.82, help="Transformer high confidence/rescue threshold")
    parser.add_argument('--force_recache', action='store_true', help="Force regenerate ICL context cache")
    parser.add_argument('--api_keys', type=str, default="", help="Comma separated API keys")
    parser.add_argument('--project_root', type=str, default="", help="Project root directory")
    args = parser.parse_args()

    config.load_from_args(args)
    if not config.api_keys:
        logging.error("No API keys provided. Set LLM_API_KEYS env var or pass --api_keys")
        return

    current_script_dir = Path(__file__).resolve().parent
    project_root = config.project_root

    output_dir = current_script_dir / "outputs"
    cache_dir = config.data_paths['cache']

    gnn_model_path = project_root / "best_model_mil.pt"
    transformer_model_path = project_root / "best_transformer_model.pt"
    vocab_path = config.data_paths['cache'] / "processed_train_withGraphFeat_graphcodebert" / "_vocab.json"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_archive_dir = output_dir / f"{timestamp}_CodeRag_run_{config.dataset}"
    run_archive_dir.mkdir(parents=True, exist_ok=True)
    persistent_results_file = output_dir / f"coderag_persistent_results_{config.dataset}.json"

    log_file_handler = logging.FileHandler(run_archive_dir / 'run.log', encoding='utf-8')
    logging.getLogger().addHandler(log_file_handler)

    logging.info("--- Hybrid ICL-CodeRag Vulnerability Detection Framework Started ---")
    logging.info(f"Project Root: {project_root}")
    logging.info(f"Script Directory: {current_script_dir}")
    logging.info(f"Archive Output Directory: {output_dir}")
    logging.info(f"Cache Directory: {cache_dir}")

    if not gnn_model_path.exists(): raise FileNotFoundError(f"GNN model file not found: {gnn_model_path}")
    if not transformer_model_path.exists(): raise FileNotFoundError(f"Transformer model file not found: {transformer_model_path}")
    

    if not vocab_path.exists():
         found_vocab = list(config.data_paths['cache'].glob("**/_vocab.json"))
         if found_vocab:
             vocab_path = found_vocab[0]
             logging.info(f"Found vocab file at: {vocab_path}")
         else:
             raise FileNotFoundError(f"Vocab file not found at {vocab_path} or in cache subdirectories.")

    clients = [OpenAI(api_key=key, base_url=config.api_base_url) for key in config.api_keys if key]
    if not clients: raise ValueError("LLM_API_KEYS not set or all empty.")
    client_cycler = cycle(clients)

    logging.info("\n[1/5] Loading all models and data vocabularies...")
    tokenizer = RobertaTokenizer.from_pretrained(config.codebert_model)

    transformer_model = TransformerVulnerabilityModel().to(device)
    transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
    transformer_model.eval()

    codebert_for_nodes = RobertaModel.from_pretrained(config.codebert_model).to(device)
    codebert_for_nodes.eval()

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    node_type_map, edge_type_map = vocab['node_type_map'], vocab['edge_type_map']

    num_node_features = 768 + len(node_type_map)
    num_relations = len(edge_type_map)
    num_graph_features = 12
    gnn_model = RGCNModel(
        num_node_features=num_node_features,
        num_relations=num_relations,
        num_graph_features=num_graph_features
    ).to(device)

    gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device))
    gnn_model.eval()

    logging.info("\n[2/5] Preparing multimodal ICL context data...")
    context_data = load_and_cache_context_data(project_root, cache_dir, gnn_model, codebert_for_nodes, tokenizer,
                                               node_type_map,
                                               edge_type_map, limit_context_samples, args.force_recache)

    logging.info("\n[3/5] Preparing test data and checking breakpoints...")
    with open(config.data_paths['test'], 'r', encoding='utf-8') as f:
        test_items_info = json.load(f)

    unique_files = {item[1]: item for item in test_items_info}
    all_test_items = [{'c_file_name': c_fname, 'true_label': label,
                       'graph_path': str(project_root / gpath),
                       'code_path': str(config.data_paths['functions'] / c_fname)}
                      for _, c_fname, label, gpath in unique_files.values()]

    existing_results = []
    if persistent_results_file.exists():
        try:
            with open(persistent_results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, TypeError):
            logging.warning(f"Persistent results file {persistent_results_file} parse failed, restarting this time.")
            existing_results = []

    processed_files = {res['c_file_name'] for res in existing_results}
    items_to_process = [item for item in all_test_items if item['c_file_name'] not in processed_files]

    logging.info(f"Total deduplicated samples: {len(all_test_items)}.")
    logging.info(f"Loaded {len(existing_results)} historical results, skipping {len(processed_files)} processed files.")

    if limit_test_samples > 0:
        items_to_process = items_to_process[:limit_test_samples]
        logging.info(f"Sample limit applied, processing max {len(items_to_process)} new samples this time.")
    else:
        logging.info(f"New samples to process this time: {len(items_to_process)}.")

    if not items_to_process:
        logging.info("All samples processed, no need to run again.")
        all_results = existing_results
    else:
        logging.info(f"\n[4/5] Starting concurrent detection with {args.workers} worker threads...")
        detector = CodeRagDetector(gnn_model, transformer_model, codebert_for_nodes, tokenizer, context_data,
                                node_type_map, edge_type_map,
                                args.transformer_decision_threshold, args.transformer_rescue_threshold)
        new_results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            tasks = [(item, detector, next(client_cycler)) for item in items_to_process]
            future_to_item = {executor.submit(process_single_item, task): task[0] for task in tasks}

            for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc="CodeRag Detecting"):
                new_results.append(future.result())

        all_results = existing_results + new_results

        with open(persistent_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Detection complete. Total {len(all_results)} results updated to '{persistent_results_file}'.")

    logging.info("\n[5/5] Calculating final evaluation metrics based on all results...")
    metrics = calculate_metrics_summary(all_results)

    run_args = {
        "limit_test_samples": limit_test_samples,
        "limit_context_samples": limit_context_samples,
        **vars(args)
    }

    print("\n" + "=" * 20 + " 📊 Final Evaluation Summary " + "=" * 20)
    if metrics:
        total_evaluated = len([r for r in all_results if 'error' not in r])
        print(f"Total Evaluated Samples: {total_evaluated}")
        for key, val in metrics.items(): print(f"{key.replace('_', ' ').capitalize():<25}: {val:.4f}")

        summary_filename = run_archive_dir / 'run_summary.json'
        run_args_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in run_args.items()}
        summary_data = {"args": run_args_serializable, "metrics": metrics}

        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Evaluation summary for this run archived to '{summary_filename}'.")
    else:
        print("Insufficient valid results to generate evaluation.")
    print("=" * 58)

if __name__ == "__main__":
    main()