import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import tensorflow_addons as tfa
from numpy.random import randint
import os

PATCH_SIZE           = 40
GAUSSIAN_NOISE       = 0.25
SCALING_FACTOR       = 2
interpolation_method = 'bicubic'
BOUNDARY_VOXELS_1    = 50
BOUNDARY_VOXELS_2    = BOUNDARY_VOXELS_1-PATCH_SIZE


@tf.function
def NormalizeImage(image):
    return (image - tf.math.reduce_min(image)) / (tf.math.reduce_max(image) - tf.math.reduce_min(image))

@tf.function
def get_random_patch_dims(image):    
    r_x = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[0]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_y = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[1]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_z = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[2]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    return r_x, r_y, r_z

def flip_model_x(model):
    return model[::-1, :, :]

def flip_model_y(model):
    return model[:, ::-1, :]

def flip_model_z(model):
    return model[:, :, ::-1]

@tf.function
def data_augmentation(lr_image, hr_image):
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_x(lr_image), flip_model_x(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_y(lr_image), flip_model_y(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_z(lr_image), flip_model_z(hr_image)
    return lr_image, hr_image

def get_nii_file(nii_file_path):
    img_sitk = sitk.ReadImage(nii_file_path.decode('UTF-8'), sitk.sitkInt32)
    hr_image = sitk.GetArrayFromImage(img_sitk)
    return hr_image

@tf.function
def add_noise(hr_image):
    blurred_image = tfa.image.gaussian_filter2d(hr_image, sigma=GAUSSIAN_NOISE)
    return blurred_image, hr_image

def get_low_res(blurred_image, hr_image):
    x, y, z = blurred_image.shape
    lr_image   = tf.image.resize(blurred_image, [x//SCALING_FACTOR, y//SCALING_FACTOR], method=interpolation_method).numpy()
    lr_image = np.rot90(lr_image, axes=(1,2))
    lr_image = tf.image.resize(lr_image, [x//SCALING_FACTOR, z//SCALING_FACTOR], method=interpolation_method).numpy()
    ups_lr_image = tf.image.resize(lr_image, [x//SCALING_FACTOR, z], method=interpolation_method).numpy()
    ups_lr_image = np.rot90(ups_lr_image, axes=(1,2))
    ups_lr_image = tf.image.resize(ups_lr_image, [x, y], method=interpolation_method).numpy()
    ups_lr_image = np.array(np.rot90(ups_lr_image, k=2, axes=(1,2)), dtype='int32')
    return ups_lr_image, hr_image

@tf.function
def normalize(lr_image, hr_image):
    hr_image = NormalizeImage(hr_image)
    lr_image = NormalizeImage(lr_image)
    return lr_image, hr_image

@tf.function
def extract_patch(lr_image, hr_image):
    r_x, r_y, r_z = get_random_patch_dims(hr_image)
    hr_random_patch = hr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    lr_random_patch = lr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    return tf.expand_dims(lr_random_patch, axis=3), tf.expand_dims(hr_random_patch, axis=3)

def get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE):

    # Directories containing High Resolution (HR) and Low Resolution (LR) MRI volumes
    HR_DIR = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/High-Res"
    LR_DIR = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/Low-Res"

    hr_nii_files = glob.glob(os.path.join(HR_DIR, "**/*.nii"), recursive=True)
    lr_nii_files = []
    matched_hr_nii_files = []
    for hr_path in hr_nii_files:
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(LR_DIR, filename)
        if os.path.exists(lr_path):
            matched_hr_nii_files.append(hr_path)
            lr_nii_files.append(lr_path)

    hr_nii_files = np.array(matched_hr_nii_files)
    lr_nii_files = np.array(lr_nii_files)

    AUTOTUNE = tf.data.AUTOTUNE

    # Create paired Dataset of (lr_path, hr_path)
    file_names_lr = tf.data.Dataset.from_tensor_slices(lr_nii_files)
    file_names_hr = tf.data.Dataset.from_tensor_slices(hr_nii_files)
    file_pairs    = tf.data.Dataset.zip((file_names_lr, file_names_hr))

    # Load LR and HR volumes
    image_pairs = file_pairs.map(lambda lr_p, hr_p: (
                                    tf.numpy_function(func=get_nii_file, inp=[lr_p], Tout=tf.int32),
                                    tf.numpy_function(func=get_nii_file, inp=[hr_p], Tout=tf.int32)),
                                 num_parallel_calls=AUTOTUNE, deterministic=False)

    # Normalization & Patch Extraction
    norm_image_pairs  = image_pairs.map(normalize, num_parallel_calls=AUTOTUNE, deterministic=False)
    norm_image_pairs  = norm_image_pairs.map(extract_patch, num_parallel_calls=AUTOTUNE, deterministic=False)

    dataset_size          = norm_image_pairs.cardinality().numpy()
    train_data_threshold  = int(0.7*dataset_size) # 70% of the dataset

    # Training Data Pipeline
    train_dataset = norm_image_pairs.take(train_data_threshold)
    train_dataset = train_dataset.map(data_augmentation, num_parallel_calls=AUTOTUNE, deterministic=False)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    train_dataset_size = train_dataset.cardinality().numpy()*BATCH_SIZE

    remain_dataset        = norm_image_pairs.skip(train_data_threshold)
    remain_dataset_size   = remain_dataset.cardinality().numpy()
    valid_data_threshold  = int(0.5*remain_dataset_size) # 15% of the dataset

    # Validation Data Pipeline
    valid_dataset      = remain_dataset.take(valid_data_threshold)
    valid_dataset_size = valid_dataset.cardinality().numpy()
    valid_dataset = valid_dataset.batch(VALIDATION_BATCH_SIZE, drop_remainder=True)
    valid_dataset_size = valid_dataset.cardinality().numpy()*VALIDATION_BATCH_SIZE

    # Test Data Pipeline
    test_dataset      = remain_dataset.skip(valid_data_threshold)
    test_dataset_size = test_dataset.cardinality().numpy()
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    test_dataset_size = test_dataset.cardinality().numpy()*BATCH_SIZE

    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size, test_dataset, test_dataset_size
