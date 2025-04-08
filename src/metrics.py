def precision(golden_text: str, retrieved_text: str) -> float:
    golden_set = set(golden_text.split())
    retrieved_set = set(retrieved_text.split())
    if not retrieved_set:
        return 0.0
    return len(retrieved_set.intersection(golden_set)) / len(retrieved_set)

def recall(golden_text: str, retrieved_text: str) -> float:
    golden_set = set(golden_text.split())
    retrieved_set = set(retrieved_text.split())
    if not golden_set:
        return 0.0
    return len(retrieved_set.intersection(golden_set)) / len(golden_set)
