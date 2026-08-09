import cv2
from src.segmentation import segment_object

img = cv2.imread("data/raw/adulterated/turmeric1.jpg")  # change filename

if img is None:
    print("Image not found")
    exit()

mask = segment_object(img)

masked = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Segmented", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()