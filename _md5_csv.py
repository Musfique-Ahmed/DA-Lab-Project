import hashlib, os
p = r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'
size = os.path.getsize(p)
print('size:', size, 'expected:', 166132757)
h = hashlib.md5()
with open(p, 'rb') as f:
    for chunk in iter(lambda: f.read(1<<20), b''):
        h.update(chunk)
print('md5:', h.hexdigest(), 'expected:', '66f17781415d099dfb9ab65e218cee44')