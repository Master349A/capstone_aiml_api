# capstone_aiml_api
AIML API repo

Guide:
1. The 'main' folder contains the final api code
2. Each api has a class file inside the respective name 
3. All the pre-trained models are saved inside the 'apis' folder in zipped format
3. Inside the 'main' folder, there is a program that can push the pretrained models to Database. This file is clou_api_sync.py

Steps:
1. First we need to upload the models to database or we can simply use them from the local filesystem. For either approach, first step is to unzip the models in the apis folder.

2. (OPTIONAL) To upload the models to cloud:
	A. First open the file /main/cloud_api_sync.py and make the following changes - 
	
	if __name__ == '__main__':
    sync = cloudsynchroniser(client='mongodb://localhost:15485', db='mydatabase', connection='saved_models')
	
	B. Here change the field values for client, db and connection to respective values. 
	C. Then run the file.

3. Then to use the api, create an object of respective api and call the get_prediction() function. 
   Example- 
   To call the model for Pneumonia 
   
	   import PneumoniaPredictor 
	   import cloud_api_sync as cas 
	   
	   sync = cas.cloudsynchroniser(client=<client url>, db=<db name>, connection=<saved_models>)
	   pnpred = PneumoniaPredictor( cloudsync=sync )
	   
	   res = pnpred.get_prediction(bytearray)
	   print(res)
	   
	OR 
	
		import PneumoniaPredictor 

		pnpred = PneumoniaPredictor( PATH='apis/api1' )
		
		res = pnpred.get_prediction(bytearray)
		print(res)

Note: Pneumonia model is named api1 and Tuberculosis model is named api2 


Libraries used:
1. keras
2. skimage
3. PIL.image (Pillow)
4. io
5. numpy
6. pymongo
7. pickle
8. time