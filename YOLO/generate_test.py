import os

image_files = []
os.chdir(os.path.join("data", "data"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".JPG"):
        image_files.append("data/data/" + filename)
os.chdir("..")
with open("test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")