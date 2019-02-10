import os
import shutil

participants = ['Guest_%d' % i for i in range(5)]
participants.extend([
    'Bernhard Breloer',
    'Alexandar Cherkezov',
    'Carsten Rother',
    'Tilman Krempel',
    'Egon Gemmel',
    'Herbert Nachbagauer',
    'Kai Kauffmann',
    'Albert Reicherzer',
    'Patrick Seidel',
    'Philipp Cremer',
    'Knut Kuehnhausen'
])

contents = [
    'Introduction to Python in Jupyter Notebooks.ipynb',
    'Building a trading strategy.ipynb',
    'A first and very simple machine learning model.ipynb',
    'Data processing and feature engineering.ipynb',
    'Forecast Success Simulator.ipynb',
]

for participant in participants:
    dest_folder = '/home/user/Workshop/%s' % participant
    os.makedirs(dest_folder, exist_ok=True)
    print('Copying Files for %s' % participant)
    for content in contents:
        source = '/home/user/Workshop/AlexanderBeck/%s' % content
        print(source)
        shutil.copyfile(src=source, dst='%s/%s' % (dest_folder, content))
