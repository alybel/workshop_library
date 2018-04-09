import os
import shutil

participants = ['Test Teilnehmer 1']
contents = ['Introduction to Python in Jupyter Notebooks.ipynb']

for participant in participants:
    dest_folder = '/home/user/Workshop/%s' % participant.replace(' ', '\\ ')
    os.makedirs(dest_folder, exist_ok=True)
    print('Copying Files for %s' % participant)
    for content in contents:
        shutil.copyfile(src='/home/user/Workshop/AlexanderBeck/%s' % content.replace(' ', '\\ '),
                        dst='%s/%s' % (dest_folder, content.replace(' ', '\\ ')))
