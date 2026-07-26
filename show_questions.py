from data_loader import load_hotpotqa

examples = load_hotpotqa(5)
for i, e in enumerate(examples):
    print(f"Q{i+1}: {e['question']}")
    print(f"   Reponse attendue: {e['answer']}")
    print()
