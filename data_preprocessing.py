import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import os
import torchio as tio
from sklearn.model_selection import train_test_split

PATCH_SIZE = 40
SCALING_FACTOR = 2
interpolation_method = 'bicubic'


@tf.function
def NormalizeImage(image):
    image = tf.cast(image, tf.float32)
    min_val = tf.math.reduce_min(image)
    max_val = tf.math.reduce_max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


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
    img_sitk = sitk.ReadImage(nii_file_path, sitk.sitkInt32)
    image_array = sitk.GetArrayFromImage(img_sitk)
    return image_array


@tf.function
def normalize_patches(lr_patch, hr_patch):
    hr_patch = NormalizeImage(hr_patch)
    lr_patch = NormalizeImage(lr_patch)
    return lr_patch, hr_patch


def non_overlapping_patch_generator(subjects):
    patch_size = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

    def generator():
        # Iterate over each full MRI scan (subject)
        for subject in subjects:
            sampler = tio.GridSampler(
                subject=subject,
                patch_size=patch_size,
                patch_overlap=(0, 0, 0)
            )

            # Iterate directly over the sampler.
            for patch in sampler:
                lr_patch_tensor = patch['lr_image'][tio.DATA]
                hr_patch_tensor = patch['hr_image'][tio.DATA]

                # Convert from (C, D, H, W) torch tensor to (D, H, W, C) numpy array for TF
                lr_patch_np = np.expand_dims(lr_patch_tensor.numpy().squeeze(), axis=-1)
                hr_patch_np = np.expand_dims(hr_patch_tensor.numpy().squeeze(), axis=-1)

                yield (lr_patch_np, hr_patch_np)

    return generator


def get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE):
    HR_DIR = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/High-Res"
    LR_DIR = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/Low-Res"

    hr_nii_files = glob.glob(os.path.join(HR_DIR, '**', '*.nii'), recursive=True)

    subjects_list = []
    for hr_path in hr_nii_files:
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(LR_DIR, 'lowres_' + filename)
        if os.path.exists(lr_path):
            subject = tio.Subject(
                lr_image=tio.ScalarImage(lr_path),
                hr_image=tio.ScalarImage(hr_path),
            )
            subjects_list.append(subject)

    # This prevents data leakage between sets.
    train_subjects, test_val_subjects = train_test_split(subjects_list, test_size=0.3, random_state=42)
    valid_subjects, test_subjects = train_test_split(test_val_subjects, test_size=0.5, random_state=42)

    print(f"Total Subjects: {len(subjects_list)}")
    print(f"Training Subjects: {len(train_subjects)}")
    print(f"Validation Subjects: {len(valid_subjects)}")
    print(f"Test Subjects: {len(test_subjects)}")

    AUTOTUNE = tf.data.AUTOTUNE

    output_signature = (
        tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), dtype=tf.float32)
    )

    # Training Data Pipeline
    train_dataset = tf.data.Dataset.from_generator(
        non_overlapping_patch_generator(train_subjects),
        output_signature=output_signature
    )
    train_dataset = train_dataset.shuffle(buffer_size=1000)  # Shuffle the patches
    train_dataset = train_dataset.map(normalize_patches, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(data_augmentation, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    # Validation Data Pipeline
    valid_dataset = tf.data.Dataset.from_generator(
        non_overlapping_patch_generator(valid_subjects),
        output_signature=output_signature
    )
    valid_dataset = valid_dataset.map(normalize_patches, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.batch(VALIDATION_BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    # Test Data Pipeline
    test_dataset = tf.data.Dataset.from_generator(
        non_overlapping_patch_generator(test_subjects),
        output_signature=output_signature
    )
    test_dataset = test_dataset.map(normalize_patches, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset