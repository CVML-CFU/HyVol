
from .evaluation_mm import evaluate_mm
from .evaluation_classification import evaluate_mm as evaluate_classification

evaluation_registry={
                     'evaluation_mm':evaluate_mm,
                     'evaluation_classification':evaluate_classification
                     }
