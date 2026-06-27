import json

with open(r'D:\pakistan_constitution_llm\data\splits\constitution_train.jsonl') as f:
    data = [json.loads(line) for line in f]

print(f'Train examples: {len(data)}')
print('\nSample questions:')
for d in data[:5]:
    print(f'  - {d["question"]}')