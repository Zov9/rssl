import os
import shutil

# Create the 'txt' subdirectory if it doesn't exist
if not os.path.exists('txt'):
    os.makedirs('txt')

# Get a list of all .txt files in the current directory
txt_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.txt')]

# Move each .txt file to the 'txt' subdirectory
for txt_file in txt_files:
    shutil.move(txt_file, os.path.join('txt', txt_file))