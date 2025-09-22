from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainConfig:
    target: str
    cat_features: Optional[List[str]] = None
    num_features: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42


DEFAULT_TARGETS = ["Churn", "churn"]
