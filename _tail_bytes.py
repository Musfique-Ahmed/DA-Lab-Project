p = r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'
with open(p, 'rb') as f:
    f.seek(-5, 2)
    tail = f.read()
    print('LAST 5 BYTES:', tail)
    f.seek(0)
    head = f.read(200)
    print('FIRST 200 BYTES:', head)