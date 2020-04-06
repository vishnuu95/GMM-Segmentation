import os
import os.path

path = './Yellow_png/'
files = os.listdir(path)
i = 0
for src in files:
	print(i)
	dst = src.replace('gb', 'yb')
	os.rename(os.path.join(path, src), os.path.join(path, dst))
	i+=1