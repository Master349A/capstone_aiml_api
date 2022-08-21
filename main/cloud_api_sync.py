import keras
import pymongo
import pickle
import time

class cloudsynchroniser:
    def __init__(self, client, db, connection):
        self.client = client 
        self.db = db
        self.connection = connection 
        
    
    def upload_model(self, model, model_name):
        pickled_model = pickle.dumps(model)
        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]
        mycon = mydb[self.connection]
        info = mycon.insert_one( { model_name : pickled_model, 'name' : model_name, 'created_time' : time.time() } )
        print(info.inserted_id, ' saved with this id successfully!')
        details = { 'inserted_id':info.inserted_id, 'model_name':model_name, 'created_time':time.time() }
        return details 
    
    
    def download_model(self, model_name):
        json_data = {}
        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]
        mycon = mydb[self.connection]
        data = mycon.find( { 'name': model_name } )
        for i in data:
            json_data = i 
        pickled_model = json_data[model_name]
        return pickle.loads(pickled_model)
        

if __name__ == '__main__':
    sync = cloudsynchroniser(client='mongodb://localhost:15485', db='mydatabase', connection='saved_models')
    
    path = 'api1/api1'
    model = keras.models.load_model(path)
    details = sync.upload_model(model, 'api1')
    
    path = 'api2/api2'
    model = keras.models.load_model(path)
    details = sync.upload_model(model, 'api2')


