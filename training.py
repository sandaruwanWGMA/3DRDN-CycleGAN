# -*- coding: utf-8 -*-
import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf

from logger import log
from model import Model3DRDN
from plotting import generate_images
from data_preprocessing import get_preprocessed_data, PATCH_SIZE


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at {}".format(time.ctime())
    log(stop_log)
    if 'm' in globals():
        m.save_models("stopping_epoch")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def evaluation_loop(dataset):
    all_psnrs, all_ssims, all_abs_errors = [], [], []

    for lr_batch, hr_batch in dataset:
        pred_batch = m.generator_g(lr_batch, training=False)

        # Ensure both tensors are float32 for comparison
        pred_batch = tf.cast(pred_batch, tf.float32)
        hr_batch = tf.cast(hr_batch, tf.float32)

        all_psnrs.append(tf.image.psnr(pred_batch, hr_batch, max_val=1.0))
        all_ssims.append(tf.image.ssim(pred_batch, hr_batch, max_val=1.0))
        all_abs_errors.append(tf.math.abs(hr_batch - pred_batch))

    psnr_tensor = tf.concat(all_psnrs, axis=0)
    ssim_tensor = tf.concat(all_ssims, axis=0)
    abs_err_tensor = tf.concat(all_abs_errors, axis=0)

    psnr_tensor = psnr_tensor[tf.math.is_finite(psnr_tensor)]

    mean_errors, std_errors = [], []
    for v in [psnr_tensor, ssim_tensor, abs_err_tensor]:
        mean_errors.append(round(tf.reduce_mean(v).numpy(), 4))
        std_errors.append(round(tf.math.reduce_std(v).numpy(), 4))

    return mean_errors, std_errors


def generate_random_image_slice(sample_image, str1, str2=""):
    comparison_image_lr, comparison_image_hr = sample_image
    # The input sample is already a float32 tensor, no need to convert
    comparison_image_lr = tf.expand_dims(comparison_image_lr, axis=0)
    prediction_image = tf.squeeze(m.generator_g(comparison_image_lr, training=False)).numpy()
    comparison_image_lr = tf.squeeze(comparison_image_lr).numpy()
    comparison_image_hr = tf.squeeze(comparison_image_hr).numpy()
    generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, str1, str2)


def main_loop(LR, DB, DU, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN,
              LAMBDA_CYC, LAMBDA_IDT, CRIT_ITER, TRAIN_ONLY, MODEL):
    begin_log = '\n### Began training {} at {} with parameters: ...\n'.format(MODEL,
                                                                              time.ctime())
    log(begin_log)

    training_start = time.time()
    log("Setting up Data Pipeline")
    VALIDATION_BATCH_SIZE = 4  # Can be larger now

    train_dataset, N_TRAINING_DATA, valid_dataset, N_VALIDATION_DATA, test_dataset, N_TESTING_DATA = get_preprocessed_data(
        BATCH_SIZE, VALIDATION_BATCH_SIZE)

    pipeline_seconds = time.time() - training_start
    log("Pipeline took {} to set up".format(datetime.timedelta(seconds=pipeline_seconds)))
    log("Number of Training Patches: {}, Validation Patches: {}, Test Patches: {}".format(N_TRAINING_DATA,
                                                                                          N_VALIDATION_DATA,
                                                                                          N_TESTING_DATA))

    global m
    m = Model3DRDN(PATCH_SIZE=PATCH_SIZE, DB=DB, DU=DU, BATCH_SIZE=BATCH_SIZE, LR_G=LR, LR_D=LR, LAMBDA_ADV=LAMBDA_ADV,
                   LAMBDA_GRD_PEN=LAMBDA_GRD_PEN, LAMBDA_CYC=LAMBDA_CYC, LAMBDA_IDT=LAMBDA_IDT, MODEL=MODEL,
                   CRIT_ITER=CRIT_ITER,
                   TRAIN_ONLY=TRAIN_ONLY)

    # Get one sample image for plotting throughout training.
    log("Getting a sample image for plotting...")
    sample_lr_for_plotting, sample_hr_for_plotting = next(iter(valid_dataset.take(1)))
    sample_image_for_plotting = (sample_lr_for_plotting[0], sample_hr_for_plotting[0])

    # Initial Evaluation on the full validation set
    log("Performing initial evaluation on the full validation set...")
    (va_psnr, va_ssim, va_error), (va_psnr_std, va_ssim_std, va_error_std) = evaluation_loop(valid_dataset)
    generate_random_image_slice(sample_image_for_plotting, 'a_first_plot_{}'.format(EPOCH_START))
    log("Before training: MAE = {} ± {}, PSNR = {} ± {}, SSIM = {} ± {}".format(va_error, va_error_std, va_psnr,
                                                                                va_psnr_std, va_ssim, va_ssim_std))

    for epoch in range(EPOCH_START, EPOCH_START + EPOCHS):
        log("Began epoch {} at {}".format(epoch, time.ctime()))
        epoch_start = time.time()

        for lr, hr in train_dataset:
            m.training(lr, hr, epoch)

        log("Performing validation for epoch {}...".format(epoch))
        (va_psnr, va_ssim, va_error), (va_psnr_std, va_ssim_std, va_error_std) = evaluation_loop(valid_dataset)
        generate_random_image_slice(sample_image_for_plotting, "epoch_{}".format(epoch),
                                    str2=" Epoch: {}".format(epoch))

        with m.summary_writer_valid.as_default():
            tf.summary.scalar('Mean Absolute Error', va_error, step=epoch)
            tf.summary.scalar('PSNR', va_psnr, step=epoch)
            tf.summary.scalar('SSIM', va_ssim, step=epoch)

        # Epoch Logging
        log("Finished epoch {} at {}.".format(epoch, time.ctime()))
        epoch_seconds = time.time() - epoch_start
        log("Epoch took {}".format(datetime.timedelta(seconds=epoch_seconds)))
        log("After epoch: MAE = {} ± {}, PSNR = {} ± {}, SSIM = {} ± {}".format(va_error, va_error_std, va_psnr,
                                                                                va_psnr_std, va_ssim, va_ssim_std))

        if (epoch + 1) % 30 == 0:
            m.save_models(epoch)

    m.save_models("last_epoch")

    log("Performing final evaluation on the full test set...")
    (test_psnr, test_ssim, test_error), (test_psnr_std, test_ssim_std, test_error_std) = evaluation_loop(test_dataset)

    log("After training: MAE = {} ± {}, PSNR = {} ± {}, SSIM = {} ± {}".format(test_error, test_error_std, test_psnr,
                                                                               test_psnr_std, test_ssim, test_ssim_std))
    log("Finished training at {}".format(time.ctime()))
    training_seconds = time.time() - training_start
    log("Training took {}".format(datetime.timedelta(seconds=training_seconds)))