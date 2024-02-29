import os

folder_path = "E:/Nikon/240121_tibet"
files = os.listdir(folder_path)

for file in files:
    if file.endswith(("-2.jpg", "-2.CR2")) and os.path.isfile(os.path.join(folder_path, file)):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
