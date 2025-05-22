# ai/services/predict_service.py

def predict(model_name: str, image_bytes: bytes):


    if model_name == "cnn_feature_enhanced_seg":
        from services.cnn_feature_enhanced_seg.predictor import (
            predict_bytes as cnn_feature_enhanced_seg,
        )
        
        print("cnn_feature_enhanced_seg 사용")
        return cnn_feature_enhanced_seg(image_bytes)


    elif model_name == "cnn_feature_finetuning_seg":
        from services.cnn_feature_finetuning_seg.predictor import (
            predict_bytes as cnn_feature_finetuning_seg,
        )
        
        print("cnn_feature_finetuning_seg 사용")
        return cnn_feature_finetuning_seg(image_bytes)

    elif model_name == "cnn_feature_maskcrop_seg":
        from services.cnn_feature_maskcrop_seg.predictor import (
            predict_bytes as cnn_feature_maskcrop_seg,
        )
        
        print("cnn_feature_maskcrop_seg 사용")
        return cnn_feature_maskcrop_seg(image_bytes)
    
    elif model_name == "cnn_feature_seg":
        from services.cnn_feature_seg.predictor import (
            predict_bytes as cnn_feature_seg,
        )
        
        print("cnn_feature_seg 사용")
        return cnn_feature_seg(image_bytes)

    elif model_name == "cnn_feature_seg_v2":
        from services.cnn_feature_seg_v2.predictor import (
            predict_bytes as cnn_feature_seg_v2,
        )
        
        print("cnn_feature_seg_v2 사용")
        return cnn_feature_seg_v2(image_bytes)

    

    
    elif model_name == "model_a":
        from services.model_a.predictor import (
            predict as predict_a,
        )

        print("model_a 모델 사용")
        return predict_a(image_bytes)

    # elif model_name == "model_b":
    #     from services.model_b.predictor import predict as predict_b

    #     return predict_b(image_bytes)




    elif model_name == "cnn_lgbm_bbox":
        from services.cnn_lgbm_bbox.predict.predictor import (
            predict_bytes as predict_cnn_lgbm_bbox,
        )

        print("cnn_lgbm_bbox 모델 사용")
        return predict_cnn_lgbm_bbox(image_bytes)

    elif model_name == "cnn_lgbm_seg":
        from services.cnn_lgbm_seg.predict.predictor import (
            predict_bytes as predict_cnn_lgbm_seg,
        )

        print("cnn_lgbm_seg 모델 사용")
        return predict_cnn_lgbm_seg(image_bytes)

    elif model_name == "lgbm_bbox":
        from services.lgbm_bbox.predict.predictor import (
            predict_bytes as predict_lgbm_bbox,
        )

        print("lgbm_bbox 모델 사용")
        return predict_lgbm_bbox(image_bytes)

    elif model_name == "lgbm_seg":
        from services.lgbm_bbox.predict.predictor import (
            predict_bytes as predict_lgbm_seg,
        )

        print("lgbm_seg 모델 사용")
        return predict_lgbm_seg(image_bytes)
    
    

    elif model_name == "xgb_bbox":
        from services.xgb_bbox.predict.predictor import (
            predict_bytes as predict_xgb_bbox,
        )

        print("xgb_bbox 모델 사용")
        return predict_xgb_bbox(image_bytes)

    elif model_name == "xgb_seg":
        from services.xgb_seg.predict.predictor import (
            predict_bytes as predict_xgb_seg,
        )

        print("xgb_seg 모델 사용")
        return predict_xgb_seg(image_bytes)


    else:
        raise ValueError(f"Unknown model: {model_name!r}")
