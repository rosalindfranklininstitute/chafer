import numpy as np
import tifffile

import chafer

def test_chafer_filter(ndarrays_regression):

    #open file and label for charge artifact suppression
    data_filename="../samples/SEM_yeast.tif"
    label_filename="../samples/SEM_yeast_label.tif"
    ref_filename="../samples/SEM_yeast_chafer_filtered.tif"

    data=tifffile.imread(data_filename)
    label = tifffile.imread(label_filename)
    res_ref_data = tifffile.imread(ref_filename)

    c_chafer = chafer.cls_charge_artifact_suppression_filter()

    res = c_chafer.charge_artifact_FD_filter_downup_av_prevlines3_2d(data, label)
    
    res_data = res[0]

    ndarrays_regression.check(
        {
            'points': res_data,
            'values': res_ref_data,
        },
        default_tolerance=dict(atol=1e-8,rtol=1e-8)
    )


