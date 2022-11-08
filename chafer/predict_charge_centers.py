
'''
Copyright 2022 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''

#####
# Unet predictions of charge centers
# Make sure slices in images are not too big to feed the Unet with
import numpy as np

try:
    from fastai.vision import *
    from fastai.callbacks.hooks import *
    from fastai.utils.mem import *
    from skimage import img_as_ubyte, img_as_float
    import os, shutil

    class cls_predict_charge_centers_with_unet():
        #Class to help doing predictions using fastai unet

        #def __init__(self, model_fn = 'trainedLipid50-50_Unet256_2.pkl' ):
        def __init__(self, model_fn ):
            # Parameter: model_fn - filename of fastai model in pkl format, that can be opened using fastai load_learner
            # Make sure filename is ok

            #If this code is in a package, this line will help locate the file
            #model_fn_full = os.path.join(os.path.dirname(__file__),model_fn)
            #self.model = load_learner('', model_fn_full)

            self.model = load_learner('', model_fn)

            # Remove the restriction on the model prediction size
            self.model.data.single_ds.tfmargs['size'] = None

        def create_model_from_zip(self, weights_fn):
            #Loads model from zip file.
            #This code is untested

            from zipfile import ZipFile
            from pathlib import PosixPath

            """Creates a deep learning model linked to the dataset and stores it as
            an instance attribute.
            """
            #weights_fn = weights_fn.resolve()

            print(f"Unzipping the model weights and label classes from {weights_fn}")
            
            output_dir = "extracted_model_files"

            #make sure the folder is empty
            try:
                shutil.rmtree(output_dir)
            except:
                print(f"Error deleting dir {output_dir}")

            os.makedirs(output_dir, exist_ok=True)
            with ZipFile(weights_fn, mode='r') as zf:
                zf.extractall(output_dir)
            out_path = output_dir

            # Load in the label classes from the json file
            # Can cause error if trained zip file has changed.
            # Better just load the json file that is inside the zip file
            #with open(out_path/f"{weights_fn.stem}_codes.json") as jf:
            
            pthfilematch = []
            listzipfiles=os.listdir(out_path)
            for f in listzipfiles:
                if '.pth' in f:
                    pthfilematch.append(f)
            #print(f"pthfilematch:{pthfilematch}")

            if len(pthfilematch)>0:
                modelfn = PosixPath(pthfilematch[0])
                #print(f"modelfn:{modelfn}")

                with open(out_path/f"{modelfn.stem}_codes.json") as jf:
                    self.codes = json.load(jf)
                # Have to create dummy files and datset before loading in model weights 
                self.create_dummy_files()
                self.create_dummy_dataset()
                print("Creating 2d U-net model for prediction.")
                self.model = unet_learner(
                    self.data, models.resnet34, model_dir=out_path)
                
                print("Loading in the saved weights.")
                #self.model.load(weights_fn.stem) #only needs the stem of the filename
                self.model.load(modelfn.stem) #only needs the stem of the filename

                # Remove the restriction on the model prediction size
                self.model.data.single_ds.tfmargs['size'] = None
            else:
                print("Failed to load model file.")

        def predict_charge_centers_with_unet_2d(self,data2d):
            # unet was trained for images, so convert to image
            data = img_as_float(data2d)
            img = Image(pil2tensor(data, dtype=np.float32))
            
            #Prediction here
            prediction = self.model.predict(img)
            pred_ubyte = img_as_ubyte(prediction[1][0]) #RGB, but gets only one channel

            return pred_ubyte

        def predict_charge_centers_with_unet_3d(self, data_to_pred_3d):
            data_predicted = np.zeros(data_to_pred_3d.shape ,dtype=np.uint8)


            for iz in range(data_to_pred_3d.shape[0]):
                print(f'iz: {iz}')

                dslice = data_to_pred_3d[iz,:,:]

                dpred =  self.predict_charge_centers_with_unet_2d(dslice)

                data_predicted[iz,:,:]= dpred
            
            return data_predicted

except ImportError:
    print("Failed to import python modules needed for predicting charge centres. This functionality will be unavailable.")



