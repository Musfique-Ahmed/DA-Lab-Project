f = open(r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv', 'rb')
first = f.readline()
print('FIRST_LEN:', len(first))
print('FIRST_200:', first[:200])
f.seek(-300, 2)
print('TAIL_300:', f.read())
# count lines
import subprocess
f.close()
import os
print('FILE_SIZE:', os.path.getsize(r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'))