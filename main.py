from cgi import print_directory

import filepaths
import labels
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
import matplotlib.cm as cm
from keras.src.backend.jax.nn import threshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from time import perf_counter
import seaborn as sns
#opencv
import cv2





def crop_square(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours_tuple = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]

    if not contours:
        return img

    cnt = max(contours, key=cv2.contourArea)
    x,y,w_rect,h_rect = cv2.boundingRect(cnt)

    side = max(w_rect, h_rect)
    cx, cy = x + w_rect // 2, y + h_rect // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = x1 + side
    y2 = y1 + side

    return img[y1:y2, x1:x2]


def resize_img(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img):
    return img.astype("float32") / 255.0

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def shade_correction(img, sigma = 30):
    bg = cv2.GaussianBlur(img, (0,0), sigma)
    corrected = cv2.addWeighted(img, 4, bg, -4, 128)
    return corrected

def denoise(img):
    return cv2.medianBlur(img, 3)

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_zoom(img(0.9,1.1))
    return img


def get_retina_center_and_radius(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)


    contours_tuple = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]

    if not contours:
        return None, None, None, None


    cnt = max(contours, key=cv2.contourArea)


    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    return int(cx), int(cy), int(radius), gray.shape


import math


def apply_circular_mask(img):
    cx, cy, radius, shape = get_retina_center_and_radius(img)

    if cx is None:
        print("Avertisment: Nu s-a putut detecta globul ocular. Se returnează imaginea originală.")
        return img


    mask = np.zeros(img.shape, dtype=np.uint8)


    radius_clipped = int(radius * 0.95)

    cv2.circle(mask, (cx, cy), radius_clipped, (255, 255, 255), -1)


    masked_img = cv2.bitwise_and(img, mask)


    side = radius_clipped * 2
    h_img, w_img, _ = img.shape

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w_img, cx + side // 2)
    y2 = min(h_img, cy + side // 2)


    cropped = masked_img[y1:y2, x1:x2]

    return cropped

def gamma_correction(img, gamma=1.2):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def unsharp_mask(img, k=0.6):
    blur = cv2.GaussianBlur(img, (0,0), 3)
    sharp = cv2.addWeighted(img, 1 + k, blur, -k, 0)
    return sharp


def limit_size(img, max_dim=2000):

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def show_pipeline_steps(steps, titles, cols=4):
    rows = (len(steps) + cols - 1) // cols
    plt.figure(figsize=(16, 10))

    for i, (img, title) in enumerate(zip(steps, titles)):
        plt.subplot(rows, cols, i+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def preprocess_pipeline(path, size=224, debug_visual=False):
    steps = []
    titles = []

    # 1. Load
    img = cv2.imread(path)
    if img is None:
        print("Error loading:", path)
        return None
    steps.append(img.copy())
    titles.append("1. Original")

    # 2. Limit size
    img = limit_size(img, max_dim=2000)
    steps.append(img.copy())
    titles.append("2. Limit Size")

    # 3. Circular mask
    try:
        im2 = apply_circular_mask(img)
        if im2 is None or im2.size == 0:
            im2 = img
    except:
        im2 = img
    img = limit_size(im2, max_dim=2000)
    steps.append(img.copy())
    titles.append("3. Circular Mask")

    # 4. Crop square
    try:
        im2 = crop_square(img)
        if im2 is None or im2.size == 0:
            im2 = img
    except:
        im2 = img
    img = limit_size(im2, max_dim=1500)
    steps.append(img.copy())
    titles.append("4. Crop Square")

    # 5. Shade correction
    try:
        im2 = shade_correction(img)
    except:
        im2 = img
    img = limit_size(im2, max_dim=1024)
    steps.append(img.copy())
    titles.append("5. Shade Correction")

    # 6. CLAHE
    try:
        im2 = apply_clahe(img)
    except:
        im2 = img
    img = limit_size(im2, max_dim=1024)
    steps.append(img.copy())
    titles.append("6. CLAHE")

    # 7. Gamma correction
    try:
        im2 = gamma_correction(img, gamma=1.2)
    except:
        im2 = img
    img = im2
    steps.append(img.copy())
    titles.append("7. Gamma Correction")

    # 8. Unsharp mask
    try:
        im2 = unsharp_mask(img, k=0.6)
    except:
        im2 = img
    img = im2
    steps.append(img.copy())
    titles.append("8. Sharpening")

    # 9. Denoise
    try:
        im2 = denoise(img)
    except:
        im2 = img
    img = limit_size(im2, max_dim=1024)
    steps.append(img.copy())
    titles.append("9. Denoise")

    # 10. Resize
    try:
        resized = resize_img(img, size)
    except:
        resized = cv2.resize(img, (size, size))
    img = resized
    steps.append(img.copy())
    titles.append("10. Resize")

    # 11. Normalize
    try:
        final = normalize(img)
    except:
        final = img.astype("float32") / 255.0
    steps.append((final*255).astype(np.uint8))
    titles.append("11. Final Normalized")

    # Show visual steps
    if debug_visual:
        show_pipeline_steps(steps, titles)

    return final





if __name__ == "__main__":
    trainLabels = pd.read_csv("trainLabels.csv")
    print(trainLabels.head())

    listing = os.listdir("sample/")
    print(np.size(listing))

    filepaths = []
    labels = []

    for file in listing:
        print(file)
        base = os.path.basename("sample/" + file)
        fileName = os.path.splitext(base)[0]
        filepaths.append("sample/" + file)
        labels.append(trainLabels.loc[trainLabels.image == fileName, 'level'].values[0])

    print(filepaths)
    print(labels)

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')


    image_df = pd.concat([filepaths, labels], axis=1)


    image_df = image_df.sample(frac=1).reset_index(drop=True)

    print(image_df.head(3))


    # ---------------------------------------
    # DISPLAY ORIGINAL IMAGES
    # ---------------------------------------
    #fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7),
    #                         subplot_kw={'xticks': [], 'yticks': []})

    #for i, ax in enumerate(axes.flat):
    #    if i >= len(image_df):
    #        break
    #    ax.imshow(plt.imread(image_df.Filepath[i]))
    #    ax.set_title(image_df.Label[i])

    #plt.tight_layout()
    #plt.show()


    # ---------------------------------------
    # DISPLAY DISTRIBUTION OF LABELS
    # ---------------------------------------
    #vc = image_df['Label'].value_counts()
    #plt.figure(figsize=(9, 5))
    #sns.barplot(x=vc.index, y=vc)
    #plt.title("Number of pictures of each category", fontsize=15)
    #plt.show()


    # ======================================================
    #  APPLY PREPROCESS PIPELINE TO IMAGES
    # ======================================================
    processed_images = []

    for i in range(len(image_df)):
        img_path = image_df.Filepath[i]
        processed = preprocess_pipeline(img_path, debug_visual=True)
        processed_images.append(processed)

    print("Număr imagini preprocesate:", len(processed_images))


    # ======================================================
    #  DISPLAY PREPROCESSED IMAGES
    # ======================================================
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        if i >= len(processed_images):
            break
        ax.imshow(processed_images[i])
        ax.set_title(f"Preprocessed: {image_df.Label[i]}")

    plt.tight_layout()
    plt.show()


