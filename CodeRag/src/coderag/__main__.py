import argparse
import sys
from .coderag import main as detector_main
from .train_rgcn import main as rgcn_main
from .train_transformer import main as transformer_main

def main():
    parser = argparse.ArgumentParser(description="CodeRag Unified Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    detect_parser = subparsers.add_parser("detect", help="Run the vulnerability detector")
    detect_parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    detect_parser.add_argument('--workers', type=int, default=4, help="Workers")
    detect_parser.add_argument('--api_keys', type=str, default="", help="API keys")
    detect_parser.add_argument('--project_root', type=str, default="", help="Project root directory")
    
    rgcn_parser = subparsers.add_parser("train_rgcn", help="Train the RGCN model")
    rgcn_parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    rgcn_parser.add_argument('--project_root', type=str, default="", help="Project root directory")
    
    trans_parser = subparsers.add_parser("train_transformer", help="Train the Transformer model")
    trans_parser.add_argument('--dataset', type=str, default="devign", help="Dataset name")
    trans_parser.add_argument('--project_root', type=str, default="", help="Project root directory")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args, remaining_args = parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining_args
    
    if args.command == "detect":
        if args.dataset: sys.argv.extend(['--dataset', args.dataset])
        if args.api_keys: sys.argv.extend(['--api_keys', args.api_keys])
        if args.workers: sys.argv.extend(['--workers', str(args.workers)])
        if args.project_root: sys.argv.extend(['--project_root', args.project_root])
        detector_main()
    elif args.command == "train_rgcn":
        if args.dataset: sys.argv.extend(['--dataset', args.dataset])
        if args.project_root: sys.argv.extend(['--project_root', args.project_root])
        rgcn_main()
    elif args.command == "train_transformer":
        if args.dataset: sys.argv.extend(['--dataset', args.dataset])
        if args.project_root: sys.argv.extend(['--project_root', args.project_root])
        transformer_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()