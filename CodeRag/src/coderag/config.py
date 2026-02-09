import os
from pathlib import Path
import argparse

class Config:
    def __init__(self):
        self.project_root = Path(os.getcwd())
        self.dataset = "devign" 
        self.api_keys = []
        self.api_base_url = "https://api.siliconflow.cn/v1"
        self.llm_model = "deepseek-ai/DeepSeek-V3.1-Terminus"
        self.codebert_model = "microsoft/graphcodebert-base"
        
    def load_from_args(self, args):
        if hasattr(args, 'project_root') and args.project_root:
            self.project_root = Path(args.project_root).resolve()
            
        if hasattr(args, 'dataset'):
            self.dataset = args.dataset
            
        if hasattr(args, 'api_keys') and args.api_keys:
             self.api_keys = args.api_keys.split(',')
        elif "LLM_API_KEYS" in os.environ:
             self.api_keys = os.environ["LLM_API_KEYS"].split(',')

    @property
    def data_paths(self):
        base_dir = self.project_root
        return {
            "train": base_dir / f"{self.dataset}_train_set.json" if self.dataset != "default" else base_dir / "train_set.json",
            "val": base_dir / f"{self.dataset}_val_set.json" if self.dataset != "default" else base_dir / "val_set.json",
            "test": base_dir / f"{self.dataset}_test_set.json" if self.dataset != "default" else base_dir / "test_set.json",
            "functions": base_dir / "functions",
            "cache": base_dir / "cache" / self.dataset
        }

config = Config()