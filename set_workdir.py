import shutil, os
def create_folders():
	#Folders
	data_dir = [	
		"./Data/Raw/DetectionPatches_256x256/Potsdam_ISPRS/",
		"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/valid_img/",
		"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/",
		"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/",
		]
		
	for path in data_dir:
		if not os.path.exists(path):
			os.makedirs(path)
			print("Directorio creado: ", path)

	saved_dir = [
		"./saved/my_models/",
		"./saved/FRCNN_model_LR_LR_cowc/",
		"./saved/val_images/",
		"./saved/output_images/",
		"./saved/logs/",
		"./saved/Final_SR_images_test/",
		"./saved/SR_images_test/",
		"./saved/combined_SR_images_216000/",
		"./saved/enhanced_SR_images_1/",
		"./saved/enhanced_SR_images_2/",
		"./saved/enhanced_SR_images_3/",
		"./saved/Test_Result_LR_LR_COWC/",
		"./saved/Test_Result_SR/"
		]

	for path in saved_dir:
		if not os.path.exists(path):
			os.makedirs(path)
			print("Directorio creado: ", path)

	weights_dir = "./weights/"
	if not os.path.exists(weights_dir):
		os.makedirs(weights_dir)

	print("\nCopiar imágenes con anotaciones a esta dirección: ", data_dir[0])
	print("Copiar pesos (.pth) de Google Drive a esta dirección: ", weights_dir)
	print("")

	next = input("Ya que hayas copiado lo anterior presiona (y), para interrumpir presiona (n): ")
	
	#Anotaciones
	if next == 'y' or next == 'Y':
		source_dir = "./Data/Raw/DetectionPatches_256x256/Potsdam_ISPRS/"
		dest_dir = [	
			"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/",
			"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/",
			"./Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/",
			]
		annotations = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
		annotations.sort()
		for annotation in annotations:
			shutil.copy(os.path.join(source_dir,annotation), dest_dir[0])
			shutil.copy(os.path.join(source_dir,annotation), dest_dir[1])
			shutil.copy(os.path.join(source_dir,annotation), dest_dir[2])
			print("Anotación copiada: ",annotation)
	else:
		print("Abortando...")

	#Validation dataset
	source_dir = "./Data/Raw/DetectionPatches_256x256/Potsdam_ISPRS/"
	dest_dir = "./saved/val_images/"
	img_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg') and not f.endswith('check.jpg')]
	img_files.sort()

	val_len = int(input("Ingresa tamaño del validation dataset (INT): "))
	for i in range (val_len):
		shutil.copy(os.path.join(source_dir,img_files[i]), dest_dir)
		print("Imagen copiada: ", img_files[i])

create_folders()
print("Estructura lista, procede a modificar los siguientes archivos:")
print("config_GAN.json")
print("scripts_GAB_HR-LR.py")
print("test.py")

