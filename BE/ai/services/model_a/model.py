def calculate_sweetness(gloss, color, texture):
    """광택, 색상, 질감을 조합해서 당도 점수 계산 (강화 버전)"""
    sweetness_score = (gloss * 0.6) + (color * 0.6) - (texture * 0.1)
    return sweetness_score


def score_to_brix(sweetness_score):
    """Sweetness Score를 보정된 Brix로 변환"""
    a = 2.860
    b = 12.824
    brix = a * sweetness_score + b
    return round(brix, 2)  # 소수점 2자리까지 출력


# def score_to_brix_refined(sweetness_score):
#     """Sweetness Score를 더 세밀하게 Brix로 변환"""
#     if sweetness_score >= 0.640:
#         return 15.0
#     elif sweetness_score >= 0.630:
#         return 14.8
#     elif sweetness_score >= 0.620:
#         return 14.6
#     elif sweetness_score >= 0.610:
#         return 14.4
#     elif sweetness_score >= 0.600:
#         return 14.2
#     else:
#         return 14.0
