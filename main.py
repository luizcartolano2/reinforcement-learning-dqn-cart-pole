import os

# first we need to download the libs
try:
	os.system('pip3 install -r requirements.txt')
except:
	print("Check your Python3 and Pip installations.")