import os
import shutil

participants = ['Test%d' % i for i in range(10)]
contents = ['Introduction to Python in Jupyter Notebooks.ipynb']

for participant in participants:
    dest_folder = '/home/user/Workshop/%s' % participant
    os.makedirs(dest_folder, exist_ok=True)
    print('Copying Files for %s' % participant)
    for content in contents:
        source = '/home/user/Workshop/AlexanderBeck/%s' % content
        print(source)
        shutil.copyfile(src=source, dst='%s/%s' % (dest_folder, content))
