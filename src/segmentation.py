import cv2
import numpy as np


def kmeans_segment(img, k=3):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)

    return segmented, labels.reshape(img.shape[:2])

def get_initial_mask(labels):
    unique, counts = np.unique(labels, return_counts=True)

    # assume biggest cluster = background
    bg_label = unique[np.argmax(counts)]

    mask = np.uint8(labels != bg_label) * 255
    return mask

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def grabcut_refine(img, mask):
    gc_mask = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    final_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return final_mask

def segment_object(img):
    # 1. K-means
    segmented, labels = kmeans_segment(img, k=3)

    # 2. initial mask
    mask = get_initial_mask(labels)

    # 3. cleanup
    mask = clean_mask(mask)

    # 4. refine with GrabCut
    final_mask = grabcut_refine(img, mask)

    return final_mask

def isolate_turmeric(img, object_mask):

    # keep only foreground
    masked = cv2.bitwise_and(
        img,
        img,
        mask=object_mask
    )

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    cv2.imshow("Masked Input", masked)
    # turmeric range
    lower = np.array([0, 20, 20])
    upper = np.array([60, 255, 255])

    turmeric = cv2.inRange(
        hsv,
        lower,
        upper
    )
    cv2.imshow("Before Invert", turmeric)

    turmeric = cv2.bitwise_not(turmeric)

    cv2.imshow("After Invert", turmeric)
    kernel = np.ones((5,5), np.uint8)

    turmeric = cv2.morphologyEx(
        turmeric,
        cv2.MORPH_OPEN,
        kernel
    )

    turmeric = cv2.morphologyEx(
        turmeric,
        cv2.MORPH_CLOSE,
        kernel
    )
    cv2.imshow("Turmeric Mask", turmeric)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return turmeric

