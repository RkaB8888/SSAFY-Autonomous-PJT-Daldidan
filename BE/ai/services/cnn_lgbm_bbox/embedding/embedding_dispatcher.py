# services/cnn_lgbm_bbox/embedding/embedding_dispatcher.py
from services.cnn_lgbm_bbox.config import EMBEDDING_MODE

if EMBEDDING_MODE == "handcrafted":
    from services.cnn_lgbm_bbox.extractor.feature_extractors import (
        extract_batch_handcrafted as extract_embedding,
    )
else:
    from services.cnn_lgbm_bbox.extractor.cnn_feature_extractor import (
        extract_batch as extract_embedding,
    )
