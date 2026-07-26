from datasets import load_dataset

def load_hotpotqa(n_samples=200):
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n_samples}]")
    examples = []
    for item in dataset:
        docs = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            docs.append({"title": title, "text": " ".join(sentences)})
        examples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "documents": docs,
        })
    return examples