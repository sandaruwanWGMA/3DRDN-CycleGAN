# -*- coding: utf-8 -*-
import sys
import time
import signal
import datetime
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


def evaluation_loop(dataset, dataset_size, batch_size):
    all_psnrs, all_ssims, all_abs_errors = [], [], []
    num_batches = -(-dataset_size // batch_size)  # Ceiling division
    log(f"Starting evaluation on {dataset_size} patches in {num_batches} batches...")

    for i, (lr_batch, hr_batch) in enumerate(dataset):
        if (i + 1) % 100 == 0:
            log(f"  > Evaluating batch {i + 1} / {num_batches}")
        pred_batch = m.generator_g(lr_batch, training=False)
        all_psnrs.append(tf.image.psnr(pred_batch, hr_batch, max_val=1.0))
        all_ssims.append(tf.image.ssim(pred_batch, hr_batch, max_val=1.0))
        all_abs_errors.append(tf.math.abs(hr_batch - pred_batch))

    log("Evaluation finished.")
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
    comparison_image_lr = tf.expand_dims(comparison_image_lr, axis=0)
    prediction_image = tf.squeeze(m.generator_g(comparison_image_lr, training=False)).numpy()
    comparison_image_lr = tf.squeeze(comparison_image_lr).numpy()
    comparison_image_hr = tf.squeeze(comparison_image_hr).numpy()
    generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, str1, str2)


def main_loop(LR, DB, DU, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN,
              LAMBDA_CYC, LAMBDA_IDT, CRIT_ITER, TRAIN_ONLY, MODEL):
    begin_log = '\n### Began training {} ...\n'.format(MODEL)
    log(begin_log)
    training_start = time.time()
    log("Setting up Data Pipeline")
    VALIDATION_BATCH_SIZE = 4

    train_dataset, N_TRAINING_DATA, valid_dataset, N_VALIDATION_DATA, test_dataset, N_TESTING_DATA = get_preprocessed_data(
        BATCH_SIZE, VALIDATION_BATCH_SIZE)

    log("Pipeline took {} to set up".format(datetime.timedelta(seconds=time.time() - training_start)))
    log(f"Training Patches: {N_TRAINING_DATA}, Validation Patches: {N_VALIDATION_DATA}, Test Patches: {N_TESTING_DATA}")

    global m
    m = Model3DRDN(PATCH_SIZE=PATCH_SIZE, DB=DB, DU=DU, BATCH_SIZE=BATCH_SIZE, LR_G=LR, LR_D=LR, LAMBDA_ADV=LAMBDA_ADV,
                   LAMBDA_GRD_PEN=LAMBDA_GRD_PEN, LAMBDA_CYC=LAMBDA_CYC, LAMBDA_IDT=LAMBDA_IDT, MODEL=MODEL,
                   CRIT_ITER=CRIT_ITER,
                   TRAIN_ONLY=TRAIN_ONLY)

    sample_lr, sample_hr = next(iter(valid_dataset.take(1)))
    sample_image_for_plotting = (sample_lr[0], sample_hr[0])

    log("Performing initial evaluation...")
    (va_psnr, va_ssim, va_error), _ = evaluation_loop(valid_dataset, N_VALIDATION_DATA, VALIDATION_BATCH_SIZE)
    generate_random_image_slice(sample_image_for_plotting, 'a_first_plot_{}'.format(EPOCH_START))
    log(f"Before training: MAE={va_error}, PSNR={va_psnr}, SSIM={va_ssim}")

    gen_loss_metric = tf.keras.metrics.Mean(name='train_gen_loss')
    disc_loss_metric = tf.keras.metrics.Mean(name='train_disc_loss')

    for epoch in range(EPOCH_START, EPOCH_START + EPOCHS):
        log(f"Began epoch {epoch} at {time.ctime()}")
        epoch_start = time.time()

        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()

        num_train_batches = N_TRAINING_DATA // BATCH_SIZE
        for step, (lr, hr) in enumerate(train_dataset):
            gen_loss, disc_loss = m.training(lr, hr, epoch)
            gen_loss_metric.update_state(gen_loss)
            disc_loss_metric.update_state(disc_loss)

            if (step + 1) % 50 == 0:
                log(
                    f"  Epoch {epoch}, Batch {step + 1}/{num_train_batches}, "
                    f"Gen Loss: {gen_loss_metric.result():.4f}, "
                    f"Disc Loss: {disc_loss_metric.result():.4f}"
                )

        log(f"Performing validation for epoch {epoch}...")
        (va_psnr, va_ssim, va_error), (va_psnr_std, va_ssim_std, va_error_std) = evaluation_loop(valid_dataset,
                                                                                                 N_VALIDATION_DATA,
                                                                                                 VALIDATION_BATCH_SIZE)
        generate_random_image_slice(sample_image_for_plotting, f"epoch_{epoch}", str2=f" Epoch: {epoch}")

        with m.summary_writer_valid.as_default():
            tf.summary.scalar('Mean Absolute Error', va_error, step=epoch)
            tf.summary.scalar('PSNR', va_psnr, step=epoch)
            tf.summary.scalar('SSIM', va_ssim, step=epoch)
            tf.summary.scalar('Generator Training Loss', gen_loss_metric.result(), step=epoch)
            tf.summary.scalar('Discriminator Training Loss', disc_loss_metric.result(), step=epoch)

        log(f"Finished epoch {epoch} at {time.ctime()}. Took {datetime.timedelta(seconds=time.time() - epoch_start)}")
        log(f"  Training -> Gen Loss: {gen_loss_metric.result():.4f}, Disc Loss: {disc_loss_metric.result():.4f}")
        log(f"  Validation -> MAE: {va_error} ± {va_error_std}, PSNR: {va_psnr} ± {va_psnr_std}, SSIM: {va_ssim} ± {va_ssim_std}")

        if (epoch + 1) % 30 == 0:
            m.save_models(epoch)

    m.save_models("last_epoch")

    # FINAL TESTING
    log("Performing final evaluation on the test set...")
    (test_psnr, test_ssim, test_error), (test_psnr_std, test_ssim_std, test_error_std) = evaluation_loop(test_dataset,
                                                                                                         N_TESTING_DATA,
                                                                                                         BATCH_SIZE)
    log(f"After training: MAE = {test_error} ± {test_error_std}, PSNR = {test_psnr} ± {test_psnr_std}, SSIM = {test_ssim} ± {test_ssim_std}")

    log(f"Total training took {datetime.timedelta(seconds=time.time() - training_start)}")