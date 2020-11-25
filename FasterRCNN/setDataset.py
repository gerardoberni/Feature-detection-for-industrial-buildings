import os, glob, shutil, sys, argparse

def check(txt_file,ROOT):
    c45 = 45 # Número de clase no válida
    path = os.path.join(ROOT,txt_file)
    a_file = open(path, "r")
    listas = [(line.strip()).split() for line in a_file]
    a_file.close()
    clases = list(int(l[0]) for l in listas)

    if (c45 in clases) or (len(clases) == 0):
        print("La anotación tiene la categoría 45 o está vacía")
        print("Nombre de archivo: ", path)
        print("clases: ", clases)
        return False
    else:
        return True
        
def createFolders(ROOT):
    ANOTATIONS = os.path.join(ROOT,"Anotations")
    IMAGES = os.path.join(ROOT,"Images")
    
    if not os.path.exists(ANOTATIONS):
        os.makedirs(ANOTATIONS)
        print("Directorio creado: ", ANOTATIONS)
    
    
    if not os.path.exists(IMAGES):
        os.makedirs(IMAGES)
        print("Directorio creado: ", IMAGES)

def checkFiles(ROOT):
    ANOTATIONS = os.path.join(ROOT,"Anotations")
    IMAGES = os.path.join(ROOT,"Images")
    anotations = [f for f in os.listdir(ROOT) if f.endswith('.txt')]
    anotations.sort()
    existe = False
    c_img = 0
    c_txt = 0
    for filename in anotations:
        img_file = filename[:-4] + '.JPG'
        txt_file = filename
        if check(txt_file,ROOT):
            try:
                sourceIMG = os.path.join(ROOT, img_file)
                destinationIMG = os.path.join(IMAGES, img_file)
                shutil.move(sourceIMG, destinationIMG)
                existe = True
                c_img += 1
            except:
                print("Imagen no existe, skipping...", img_file)
                existe = False

            if existe:
                sourceTXT = os.path.join(ROOT, txt_file)
                destinationTXT = os.path.join(ANOTATIONS, txt_file)
                shutil.move(sourceTXT, destinationTXT)
                c_txt += 1
    print("Img count: ", c_img, ". Txt count: ", c_txt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help='Ruta en donde están las imágenes "crudas"')
    args = parser.parse_args()
    ROOT = args.data_path
    createFolders(ROOT)
    checkFiles(ROOT)