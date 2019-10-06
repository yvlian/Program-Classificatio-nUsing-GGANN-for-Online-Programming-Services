import os

# cmd = 'python create_valid_ids.py'
# os.system(cmd)

cmd = 'python select_data.py'
os.system(cmd)

cmd = 'python create_data.py'
os.system(cmd)

cmd = 'python ../GGANN/Model.py'
os.system(cmd)