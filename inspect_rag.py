with open('Rag_Char.py','rb') as f:
    lines = f.read().splitlines()
print('total', len(lines))
for i, l in enumerate(lines, 1):
    print(i, repr(l))
