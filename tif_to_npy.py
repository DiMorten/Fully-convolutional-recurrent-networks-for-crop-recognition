import gdal
import numpy as np
import glob

def read_tiff(tiff_file):
    data = gdal.Open(tiff_file).ReadAsArray()
    return data

def load_sentinel_data(path):
    '''
    loads the VV and VH bands of a SAR image and concatenates both channels into a numpy array
    
        Parameters:
            path (str): A folder containing VH and VV bands for a SAR image
        Returns:
            image (np.array): numpy array of shape (h,w,channels) with channels=2 
            
    Folder path example:
    path = '1_October'
    
    Folder files:
    |1_October
    |--- 29Oct_2015_VH.tif
    |--- 29Oct_2015_VV.tif
    
    '''

    img_paths = sorted(glob.glob(path + '*.tif'))
    image = [np.expand_dims(read_tiff(img).astype('float32'), -1) for img in img_paths]
    image = np.concatenate(image, axis=-1)
    print("Image shape: ", image.shape, " Min value: ", image.min(), " Max value: ", image.max())
    return image

def db2intensities(img):
    img = 10**(img/10.0)
    return img

# replace im_folders with the desired folders to convert from TIF to NPY. Exammple: ['1_October', '2_November']
im_folders = ['1_October', '2_November_1', '2_November_2', '4_December_1', '5_December_2',
        '6_January', '7_January', '8_March_1', '9_March_2', '10_May_1', '11_May_2', '12_June',
        '13_July_1', '14_July_2']

#im_folders = ['20190720/','20190614/','20190521/','20190415/','20190322/','20190214/','20190121/',
#    '20181216/','20181110/']

results_path = '../preproc_npy/
for im_folder in im_folders:
    S1_img = load_sentinel_data(im_folder)

    # to intensity
    S1_img = db2intensities(S1_img)

    # mask nan values
    #S1_img[S1_img==np.nan]=-1
    S1_img=np.nan_to_num(S1_img, nan=-1)
    print(S1_img.min(), S1_img.max())

    print("Per band stats")
    print(S1_img[:,:,0].min(), S1_img[:,:,0].max())
    print(S1_img[:,:,1].min(), S1_img[:,:,1].max())
    S1_img = S1_img.astype(np.float16)
    print(S1_img.dtype)

    result_filename = im_folder+'.npy'
    np.save(results_path + result_filename, S1_img)