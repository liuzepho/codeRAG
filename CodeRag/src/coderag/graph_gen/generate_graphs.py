import os
import json
import subprocess
import shutil
import multiprocessing
import argparse
from tqdm import tqdm
from pathlib import Path
from functools import partial

def process_single_file(file_info, project_root_path, temp_cpg_base_dir, final_graph_dir, joern_cli_path):
    index, relative_path, label = file_info
    joern_parse_path = os.path.join(joern_cli_path, "joern-parse")
    joern_export_path = os.path.join(joern_cli_path, "joern-export")
    
    temp_cpg_filename = f"{os.getpid()}_{index}.cpg.bin"
    temp_cpg_path = os.path.join(temp_cpg_base_dir, temp_cpg_filename)
    temp_export_dir = os.path.join(temp_cpg_base_dir, f"{os.getpid()}_{index}_export_out")
    
    try:
        absolute_source_path = os.path.join(project_root_path, "functions", relative_path)
        if not os.path.exists(absolute_source_path): return None

        parse_cmd = [joern_parse_path, absolute_source_path, f"--output={temp_cpg_path}"]
        result_parse = subprocess.run(parse_cmd, capture_output=True, text=True, check=False, timeout=180)
        if result_parse.returncode != 0: return None

        if os.path.exists(temp_export_dir): shutil.rmtree(temp_export_dir)
        export_cmd = [
            joern_export_path, "--repr", "cpg", "--format", "graphml",
            f"--out={temp_export_dir}", temp_cpg_path
        ]
        result_export = subprocess.run(export_cmd, capture_output=True, text=True, check=False, timeout=180)
        if result_export.returncode != 0 or "error" in result_export.stderr.lower(): return None

        destination_dir = os.path.join(final_graph_dir, os.path.basename(absolute_source_path))
        if os.path.exists(destination_dir): shutil.rmtree(destination_dir)

        shutil.move(temp_export_dir, destination_dir)
        return True
    except (Exception, subprocess.TimeoutExpired):
        return None
    finally:
        if os.path.exists(temp_cpg_path): os.remove(temp_cpg_path)
        if os.path.exists(temp_export_dir): shutil.rmtree(temp_export_dir)

def main():
    parser = argparse.ArgumentParser(description="Generate GraphML files using Joern")
    parser.add_argument('--project_root', type=str, required=True, help="Root directory of the project containing source code")
    parser.add_argument('--joern_path', type=str, required=True, help="Path to joern-cli directory")
    parser.add_argument('--labels_file', type=str, required=True, help="Path to the labels JSON file")
    parser.add_argument('--workers', type=int, default=0, help="Number of workers (0 for cpu count)")
    args = parser.parse_args()

    project_root = args.project_root
    joern_cli_path = args.joern_path
    labels_json_path = args.labels_file
    
    graph_output_dir = os.path.join(project_root, "graph_data")
    temp_cpg_dir = os.path.join(project_root, "temp_cpgs")

    print("--- Starting Graph Generation ---")
    Path(graph_output_dir).mkdir(exist_ok=True)
    Path(temp_cpg_dir).mkdir(exist_ok=True)

    if not os.path.exists(labels_json_path):
        print(f"Error: Manifest file {labels_json_path} not found")
        return

    with open(labels_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        label_data = data
        all_tasks = [[i, item["file_name"], item["label"]] for i, item in enumerate(label_data)]
    elif isinstance(data, dict) and "files" in data and "labels" in data:
        files = data["files"]
        labels = data["labels"]
        all_tasks = [[i, f, l] for i, (f, l) in enumerate(zip(files, labels))]
    else:
        print("Error: Unknown labels JSON format")
        return

    processed_files = set()
    for task in all_tasks:
        task_path = task[1]
        expected_dir = os.path.join(graph_output_dir, os.path.basename(task_path))
        if os.path.exists(expected_dir) and len(os.listdir(expected_dir)) > 0:
            processed_files.add(task_path)
            
    tasks_to_run = [task for task in all_tasks if task[1] not in processed_files]

    print(f"Total tasks: {len(all_tasks)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"New tasks: {len(tasks_to_run)}")

    if tasks_to_run:
        worker_count = multiprocessing.cpu_count() if args.workers == 0 else args.workers
        print(f"Using {worker_count} workers...")
        worker_func = partial(process_single_file,
                              project_root_path=project_root,
                              temp_cpg_base_dir=temp_cpg_dir,
                              final_graph_dir=graph_output_dir,
                              joern_cli_path=joern_cli_path)
        with multiprocessing.Pool(processes=worker_count) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks_to_run), total=len(tasks_to_run), desc="Generating Graphs"))

    print("\n--- Generating final manifest ---")
    final_graph_list = []

    
    for i, task in tqdm(enumerate(all_tasks), total=len(all_tasks), desc="Building manifest"):
        task_path = task[1]
        label = task[2]
        expected_output_dir = os.path.join(graph_output_dir, os.path.basename(task_path))

        if os.path.exists(expected_output_dir):
            for root, _, files in os.walk(expected_output_dir):
                for file in files:
                    if file.endswith(('.graphml', '.xml')):
                        found_graph_file = os.path.join(root, file)
                        relative_graph_path = os.path.relpath(found_graph_file, project_root)
                        final_graph_list.append([len(final_graph_list), task_path, label, relative_graph_path])

    output_manifest = os.path.join(project_root, "graph_data_list.json")
    with open(output_manifest, "w") as f:
        json.dump(final_graph_list, f, indent=2)

    print(f"Manifest saved to: {output_manifest}")
    print("--- Done ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()