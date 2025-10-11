from dataclasses import dataclass

@dataclass
class Config:
    working_dir: str = "./latex_kg_output"
    parser: str = "mineru"
    enable_math_analysis: bool = True