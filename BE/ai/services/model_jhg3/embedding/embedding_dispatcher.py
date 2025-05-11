# services/model_jhg3/embedding/embedding_dispatcher.py
from services.model_jhg3.config import EMBEDDING_MODE

if EMBEDDING_MODE == "handcrafted":
    from services.model_jhg3.extractor.feature_extractors_v2 import (
        extract_batch_handcrafted as extract_embedding,
    )
else:
    from services.model_jhg3.extractor.cnn_feature_extractor import (
        extract_batch as extract_embedding,
    )
