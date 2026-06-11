import json

path = 'RAG_reorganized.ipynb'  # cambia por el nombre real de tu notebook

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

nb.get('metadata', {}).pop('widgets', None)
for cell in nb.get('cells', []):
    cell.get('metadata', {}).pop('widgets', None)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done')