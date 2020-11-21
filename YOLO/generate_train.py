import os

image_files = []
os.chdir(os.path.join("data", "data3"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".JPG"):
        image_files.append("data/data3/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")