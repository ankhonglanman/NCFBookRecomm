
import numpy as np


def evaluate_ranking(
    scores,
    candidate_items,
    positive_items,
    top_ks=(5, 10, 20)
):
    """
    Đánh giá hệ thống gợi ý bằng HR@K và NDCG@K.

    scores: (N, C)
        Điểm dự đoán của N users cho C candidate books.

    candidate_items: (N, C)
        ID sách của từng candidate.

    positive_items: (N,)
        ID sách positive của mỗi user.

    Mô hình dự đoán điểm càng cao càng tốt.
    """

    scores = np.asarray(scores, dtype=float)
    candidate_items = np.asarray(
        candidate_items, dtype=np.int64
    )
    positive_items = np.asarray(
        positive_items, dtype=np.int64
    )

    # Kiểm tra dữ liệu đầu vào
    if scores.ndim != 2:
        raise ValueError("scores phải là ma trận 2 chiều")

    if scores.shape != candidate_items.shape:
        raise ValueError("scores và candidates khác kích thước")

    if len(positive_items) != len(scores):
        raise ValueError("Số positive items không khớp")

    if len(scores) == 0:
        raise ValueError("Không có dữ liệu đánh giá")

    if not np.isfinite(scores).all():
        raise ValueError("scores chứa NaN hoặc Infinity")

    top_ks = sorted(set(top_ks))

    if any(k <= 0 for k in top_ks):
        raise ValueError("K phải lớn hơn 0")

    hits = {k: 0 for k in top_ks}
    ndcgs = {k: 0.0 for k in top_ks}

    # Đánh giá từng user
    for user_index in range(len(scores)):

        user_scores = scores[user_index]
        user_items = candidate_items[user_index]
        positive_id = positive_items[user_index]

        # Kiểm tra candidate không trùng ID
        if len(np.unique(user_items)) != len(user_items):
            raise ValueError("Candidate IDs bị trùng")

        positive_positions = np.flatnonzero(
            user_items == positive_id
        )

        if len(positive_positions) != 1:
            raise ValueError("Positive item không hợp lệ")

        positive_position = positive_positions[0]

        # Xếp theo score giảm dần.
        # Nếu bằng điểm, dùng item ID để phá hòa,
        # không ưu tiên positive ở cột đầu tiên.
        sorted_indices = np.lexsort(
            (user_items, -user_scores)
        )

        rank = int(
            np.flatnonzero(
                sorted_indices == positive_position
            )[0]
        ) + 1

        for k in top_ks:
            if rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / np.log2(rank + 1)

    n_users = len(scores)

    results = {"num_users": n_users}

    for k in top_ks:
        results[f"hr@{k}"] = hits[k] / n_users
        results[f"ndcg@{k}"] = ndcgs[k] / n_users

    return results
