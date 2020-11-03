import shutil, os, json
def create_config_JSON(base_dir,data_dir,saved_dir,weights_dir,processed_x4_dir):
	config_file = open("config_GAN.json","r")
	config_object = json.load(config_file)
	config_file.close()
	
	#Data loader dir
	config_object["data_loader"]["args"]["data_dir_GT"] = os.path.join(base_dir,processed_x4_dir[1])
	config_object["data_loader"]["args"]["data_dir_LQ"] = os.path.join(base_dir,processed_x4_dir[2])

	#Train save dir
	config_object["train"]["save_dir"] = os.path.join(base_dir,"saved")

	#Weights dir
	config_object["path"]["pretrain_model_G"] = os.path.join(base_dir,weights_dir,"170000_G.pth")
	config_object["path"]["pretrain_model_D"] = os.path.join(base_dir,weights_dir,"170000_D.pth")
	config_object["path"]["pretrain_model_FRCNN"] = os.path.join(base_dir,weights_dir,"170000_FRCNN.pth")

	#Data dir
	config_object["path"]["data_dir_Bic_x4"] = os.path.join(base_dir,processed_x4_dir[0])
	config_object["path"]["data_dir_LR_train"] = os.path.join(base_dir,processed_x4_dir[2])
	config_object["path"]["data_raw"] = os.path.join(base_dir, data_dir[0])
	config_object["path"]["data_dir_Bic_valid"] = os.path.join(base_dir, data_dir[1])
	config_object["path"]["data_dir_Bic"] = os.path.join(base_dir, data_dir[2])
	config_object["path"]["data_dir_Valid"] = os.path.join(base_dir, data_dir[3])

	#config_object["path"]["data_processed"] = os.path.join(base_dir,)
	
	#Saved dir
	config_object["path"]["models"] = os.path.join(base_dir,saved_dir[0])
	config_object["path"]["FRCNN_model"] = os.path.join(base_dir,saved_dir[1])
	config_object["path"]["training_state"] = os.path.join(base_dir,saved_dir[2])
	config_object["path"]["val_images"] = os.path.join(base_dir,saved_dir[3])
	config_object["path"]["output_images"] = os.path.join(base_dir,saved_dir[4])
	config_object["path"]["log"] = os.path.join(base_dir,saved_dir[5])
	config_object["path"]["data_dir_F_SR"] = os.path.join(base_dir,saved_dir[6])
	config_object["path"]["data_dir_SR"] = os.path.join(base_dir,saved_dir[7])
	config_object["path"]["data_dir_SR_combined"] = os.path.join(base_dir,saved_dir[8])
	config_object["path"]["data_dir_E_SR_1"] = os.path.join(base_dir,saved_dir[9])
	config_object["path"]["data_dir_E_SR_2"] = os.path.join(base_dir,saved_dir[10])
	config_object["path"]["data_dir_E_SR_3"] = os.path.join(base_dir,saved_dir[11])
	config_object["path"]["Test_Result_LR_LR_COWC"] = os.path.join(base_dir,saved_dir[12])
	config_object["path"]["Test_Result_SR"] = os.path.join(base_dir,saved_dir[13])
	
	config_file = open("config_GAN.json","w")
	json.dump(config_object, config_file, indent=4)
	config_file.close()
	
def create_folders(base_dir):

	#Folders
	data_dir = [	
		"Data/Raw/DetectionPatches_256x256/Potsdam_ISPRS/",
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/valid_img/",
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/",
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/"
		]
	
	for p in data_dir:
		path = os.path.join(base_dir,p)
		if not os.path.exists(path):
			os.makedirs(path)
			print("Directorio creado: ", path)

	saved_dir = [
		"saved/my_models/",
		"saved/FRCNN_model_LR_LR_cowc/",
		"saved/training_state",
		"saved/val_images/",
		"saved/output_images/",
		"saved/logs/",
		"saved/Final_SR_images_test/",
		"saved/SR_images_test/",
		"saved/combined_SR_images_216000/",
		"saved/enhanced_SR_images_1/",
		"saved/enhanced_SR_images_2/",
		"saved/enhanced_SR_images_3/",
		"saved/Test_Result_LR_LR_COWC/",
		"saved/Test_Result_SR/"
	]

	for p in saved_dir:
		path = os.path.join(base_dir,p)
		if not os.path.exists(path):
			os.makedirs(path)
			print("Directorio creado: ", path)

	weights_dir = "weights/"
	path = os.path.join(base_dir,weights_dir)
	if not os.path.exists(path):
		os.makedirs(weights_dir)

	# Transferir dataset de entrenamiento a carpeta de la aplicación
	origin_data_dir = input("Ingresa el path absoluto en donde se encuentran tus imagenes con anotaciones de entrenamiento: ")
	origin_data_files = os.listdir(origin_data_dir)
	for f in origin_data_files:
		shutil.copy(os.path.join(origin_data_dir,f), data_dir[0])

	# Transferir dataset de entrenamiento a carpeta de la aplicación
	origin_weights_dir = input("Ingresa el path absoluto en donde se encuentran tus weights (.pth): ")
	origin_weights_files = os.listdir(origin_weights_dir)
	for weight in origin_weights_files:
		shutil.copy(os.path.join(origin_weights_dir,weight), weights_dir)

	# Anotaciones
	#print("Copiando anotaciones...")
	
	raw_data_dir = "Data/Raw/DetectionPatches_256x256/Potsdam_ISPRS/"
	processed_x4_dir = [
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/",
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/",
		"Data/Processed/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/",
		]
	annotations = [f for f in os.listdir(raw_data_dir) if f.endswith('.txt')]
	annotations.sort()
	for annotation in annotations:
		shutil.copy(os.path.join(raw_data_dir,annotation), processed_x4_dir[0])
		shutil.copy(os.path.join(raw_data_dir,annotation), processed_x4_dir[1])
		shutil.copy(os.path.join(raw_data_dir,annotation), processed_x4_dir[2])
		print("Anotación copiada: ",annotation)
	

	#Validation dataset
	val_data_dir = ["saved/val_images/","saved/output_images/"]
	img_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.jpg') and not f.endswith('check.jpg')]
	img_files.sort()

	val_len = int(input("Ingresa tamaño del validation dataset (INT): "))
	for i in range (val_len):
		shutil.copy(os.path.join(raw_data_dir,img_files[i]), val_data_dir[0])
		shutil.copy(os.path.join(raw_data_dir,img_files[i]), val_data_dir[1])
		print("Imagen copiada: ", img_files[i])

	return data_dir,saved_dir,weights_dir,processed_x4_dir

base_dir = os.getcwd()
data_dir,saved_dir,weights_dir,processed_x4_dir = create_folders(base_dir)
create_config_JSON(base_dir,data_dir,saved_dir,weights_dir,processed_x4_dir)
print("¡Estructura y archivo config listos!")
print("Procede a correr python ./scripts_for_datasets/scripts_GAN_HR-R.py")
