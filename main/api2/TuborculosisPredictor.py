import keras
from skimage.transform import resize
import PIL.Image as Image
import io
import numpy as np 


class TuborculosisPredictor:

  def __init__(self, PATH = '/content/drive/MyDrive/GL_Capstone/api2'):
    self.PATH = PATH
    self.cnn = keras.models.load_model(PATH)


  def get_predictions(self, img_byte):
    img = Image.open(io.BytesIO(img_byte))
    img2 = np.array(img)
    #print("Dim: ", img2.shape)
    img_rs = resize(img2, (320, 320, 3), anti_aliasing=True)
    test = np.expand_dims(img_rs, axis=0)
    res = self.cnn.predict(test)
    return res[0,0]
	
if __name__ == '__main__':
    tbpred = TuborculosisPredictor() 

    ipath = '/content/drive/MyDrive/data/chest_xray/test/NORMAL/IM-0001-0001.jpeg'
    img_byte = b''
    with open(ipath, 'rb') as f:
        img_byte = bytearray(f.read())
    
    res = tbpred.get_predictions(img_byte)
    print(res)