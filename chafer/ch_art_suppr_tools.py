'''
ch_art_suppr_tools.py

Tools for Charge Artifact Suppression pipeline

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

import numpy as np

from skimage.transform import downscale_local_mean, rescale
# Downscale 3D data along X and Y axis, but not Z, preserving number of slices along Z
def downscale_along_XY(data3D, factor=2):
    data_downscaled_xy = downscale_local_mean(data3D,factors=(1,factor,factor)).astype(data3D.dtype)
    return data_downscaled_xy

def upscale_along_XY(data3D, factor=2 , bool_intermediate=True):
    #Upscale data along X and Y axis but not along Z, preserving number of slices along Z

    # Rescale does not work well with uint8 on binary label data
    # To ensure proper rescale make sure data in in bool format, not uint8
    data_to_scale = None
    if bool_intermediate:
        data_to_scale = data3D.astype(bool)
    else:
        data_to_scale = data3D

    data_upscaled= rescale(data_to_scale,(1,factor,factor) ).astype(data3D.dtype)
    return data_upscaled

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

except ImportError:
    print("Failed to import python modules needed for predicting charge centres. This functionality will be unavailable.")


def predict_charge_centers_with_unet_3d(data_to_pred_3d):
    data_predicted = np.zeros(data_to_pred_3d.shape ,dtype=np.uint8)

    my_pred_cls = cls_predict_charge_centers_with_unet()

    for iz in range(data_to_pred_3d.shape[0]):
        print(f'iz: {iz}')

        dslice = data_to_pred_3d[iz,:,:]

        dpred =  my_pred_cls.predict_charge_centers_with_unet_2d(dslice)

        data_predicted[iz,:,:]= dpred
    
    return data_predicted


#####################################
# Remove charging artifacts filters

from scipy.ndimage import binary_dilation
from scipy.optimize import curve_fit

class cls_charge_artifact_suppression_filter():
    # Class to deal with filtering of charge artifacts
    # It requires charging centres to be segmented

    PASS_DIR_DOWN = False
    PASS_DIR_UP = True

    def __init__(self, nlinesaverage=20 ,data_fit_max_length_edge_px = 700, data_fit_min_length_px = 50 ):

        self.nlinesaverage = nlinesaverage
        self.data_fit_max_length_edge_px = data_fit_max_length_edge_px
        self.data_fit_min_length_px = data_fit_min_length_px

    # Note to undersand these functions: charging artifacts typically appear with lower Z than background.
    def fermidirac_left(self, x, x0, a0, sigma0): # ‾‾\__
        func = a0 *( 1.0/ (np.exp((x-x0)/sigma0) + 1) -1.0)
        return func
    
    def fermidirac_right(self, x, x0, a0, sigma0): # __/‾‾
        func = a0 *( - 1.0/ (np.exp((x-x0)/sigma0) + 1) )
        return func

    def fermidirac_right_left(self, x, x0, a0, sigma0, x1, a1,sigma1): # __/‾‾ ... ‾‾\__
        func = self.fermidirac_right(x,x0,a0,sigma0) + self.fermidirac_left(x,x1,a1,sigma1) 
        return func

    def row_get_intervals(self, labels_row):
        row_edges = np.abs(labels_row[1:] - labels_row[:-1]) > 0.5
        edge_indices = np.nonzero(row_edges)[0]

        out_intervals=[]
        in_intervals=[]

        e_prev=0
        #print(f"edge indices: {edge_indices}")
        for e0 in edge_indices:
            interval = [e_prev,e0]
            if labels_row[e0]==0:
                out_intervals.append(interval)
            else:
                in_intervals.append(interval)
            e_prev=e0
        #last one still is an interval
        interval = [e_prev, len(labels_row)]
        if labels_row[-1]==0:
            out_intervals.append(interval)
        else:
            in_intervals.append(interval)

        return edge_indices, out_intervals, in_intervals
    

    def charge_artifact_FD_filter_downup_av_prevlines3_irow_pass( self, data0, data_labels0, irow0, pass_dir0 = PASS_DIR_DOWN):
    
        errdata_xvalues_yvalues = []

        irow_av_min =irow0 - self.nlinesaverage
        irow_av_max = irow0
        if pass_dir0 == self.PASS_DIR_UP:
            irow_av_min =irow0
            irow_av_max = irow0 + self.nlinesaverage

        #Check there are any charge centres in this row
        labels_row0 = data_labels0[irow0,:]
        edge_indices0 , out_intervals0, in_intervals0 = self.row_get_intervals(labels_row0)
        
        curline = (data0[irow0,:]).astype(np.float32)

        opts = []

        if len(edge_indices0)>0:

            #Average previous lines
            wherearr = data_labels0[irow_av_min:irow_av_max,:]<1
            prevlines = np.mean(data0[irow_av_min:irow_av_max,:], axis=0, where = wherearr) #Average along the y-axis
            
            np.nan_to_num(prevlines, copy=False)

            if(np.isnan(prevlines).any()):
                print("The prevlines array contain NaN values")

            line = curline - prevlines

            for i0,i1 in out_intervals0:
                data_length = i1-i0
                #print(f"interval  {i0}:{i1} , data_length:{data_length}")

                if data_length >= self.data_fit_min_length_px:

                    #Check what type of function to fit
                    guessvalues=[]
                    bounds0 = ([0,0,0], [np.inf, 255, 255])

                    if i0==0 and i1< len(labels_row0):
                        #left-tail only
                        if data_length > self.data_fit_max_length_edge_px:
                            i0 = i1-self.data_fit_max_length_edge_px
                        func0 = self.fermidirac_left
                        #guessvalues=[i1,140,50,0]
                        guessvalues=[i1,140,20]

                    elif i0!=0 and i1!=len(labels_row0) : #Throwing exceptions
                        #mid interval
                        func0= self.fermidirac_right_left
                        guessvalues=[i0,140,20, i1,140,20]
                        bounds0 = ([0,0,0, 0,0,0], [np.inf, 255, 255, np.inf, 255, 255])

                    else:
                        #right tail only
                        if data_length > self.data_fit_max_length_edge_px:
                            i1 =  i0+self.data_fit_max_length_edge_px
                        func0= self.fermidirac_right
                        guessvalues=[i0,140,20]

                    x_values = np.arange(i0,i1)
                    y_values = line[i0:i1]
                    try:
                        popt , _ = curve_fit ( func0, x_values, y_values, guessvalues, bounds=bounds0)
                    except Exception as e:
                        #print(f"Exception.")
                        #print(str(e))
                        errdata_xvalues_yvalues.append([irow0, x_values,y_values])
                    else: #if there is no error in the optimization execute this
                        y_values_fit = func0( x_values, *popt )
                        curline[ i0:i1 ] -= y_values_fit
                        opts.append([irow0, func0.__name__, popt])

        return curline.astype(data0.dtype), errdata_xvalues_yvalues, opts

    #Filter whole slice (2D) in two passes
    def charge_artifact_FD_filter_downup_av_prevlines3_2d(self, data_2d, data_labels_2d):
        #down and then up filtering
        data_filtered = np.array(data_2d).astype(np.float32)

        pass_dirs = [self.PASS_DIR_DOWN, self.PASS_DIR_UP]

        opts = []
        for pass_dir in pass_dirs:
            #line-by-line
            for j in range(self.nlinesaverage, data_2d.shape[0]):
                irow = j #Default for down pass
                #print(f"pass: {pass_dir}, irow={irow}")

                if pass_dir == self.PASS_DIR_UP:
                    irow = data_2d.shape[0]-j-1

                data_new_row,err, opts0 = self.charge_artifact_FD_filter_downup_av_prevlines3_irow_pass(data_filtered, data_labels_2d, irow, pass_dir0=pass_dir)

                #Replace scan line with filtered
                data_filtered[irow,:] = data_new_row

                #opts0.append( pass_dir) #append info of the passdir to the optimisation values
                if len(opts0)>0:
                    opts.append([pass_dir, opts0]) # Collect optimisation values to a large array

        #print("slice complete")
        return data_filtered.astype(data_2d.dtype), opts

    def charge_artifact_suppression_filter_3d(self, data3D, charge_center_labels_3d, dilate_iter=1):
        #Filter whole 3D volume

        data_all_filt = np.zeros_like(data3D)
        optsz = []
        for iz in range(data3D.shape[0]):
            print(f"iz:{iz} / {data3D.shape[0]-1} ")
            data_slice = data3D[iz,:,:]
            labels_slice = charge_center_labels_3d[iz,:,:]

            # dilate labels
            labels_slice_dilated = binary_dilation(labels_slice,iterations=dilate_iter).astype(labels_slice.dtype)

            res0, opts = self.charge_artifact_FD_filter_downup_av_prevlines3_2d(data_slice,labels_slice_dilated)

            data_all_filt[iz,:,:] = res0.astype(data3D.dtype)

            #opts.append(iz)
            if len(opts)>0:
                optsz.append([iz,opts])

        return data_all_filt, optsz


