# services/lgbm_seg/embedding/embedding_dispatcher.py
from services.lgbm_seg.config import EMBEDDING_MODE

if EMBEDDING_MODE == "handcrafted":
    from services.lgbm_seg.extractor.feature_extractors import (
        extract_batch_handcrafted as extract_embedding,
    )
else:
    from services.lgbm_seg.extractor.cnn_feature_extractor import (
        extract_batch as extract_embedding,
    )
