import numpy as np
import cv2


def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def blur_except_faces(original_image):
    gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # mask
    # detect faces and create mask
    mask = np.full(
        (original_image.shape[0], original_image.shape[1], 1), 0, dtype=np.uint8)
    faces = face_cascade.detectMultiScale(gray_original_image, 1.3, 5)
    face_areas = []
    for (x, y, w, h) in faces:
        face_areas.append(([x, y, w, h], original_image[y:y+h, x:x+w]))
        center = (x + w // 2, y + h // 2)
        radius = max(h, w) // 2
        cv2.circle(mask, center, radius, (255), -1)

    blur_amount = 50
    # blur original image
    kernel = np.ones((blur_amount, blur_amount), np.float32) / \
        blur_amount ** 2
    blurred_image = cv2.filter2D(original_image, -1, kernel)
    # blur mask to get tapered edge
    mask = cv2.filter2D(mask, -1, kernel)

    # composite blurred and unblurred faces
    mask_inverted = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(
        blurred_image, blurred_image, mask=mask_inverted)
    foreground = cv2.bitwise_and(original_image, original_image, mask=mask)

    # create composite
    composite = alphaBlend(background, foreground, mask)
    return composite


cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    composite = blur_except_faces(frame)
    cv2.imshow('composite', composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
