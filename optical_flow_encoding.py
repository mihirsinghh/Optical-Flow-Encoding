import numpy as np
import cv2 as cv
cap = cap = cv.VideoCapture(r"C:\Users\mihir\Research\Gallant Lab\Optical-Flow-Encoding\test2_redcross.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

allHorizontalFlowValues = []
allVerticalFlowValues = []

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    allHorizontalFlowValues.append(flow[...,0])
    allVerticalFlowValues.append(flow[..., 1])

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()

hflow = np.array(allHorizontalFlowValues)
vflow = np.array(allVerticalFlowValues)

print("Horizontal flow array shape:", np.array(allHorizontalFlowValues).shape)
print("Vertical flow array shape:", np.array(allVerticalFlowValues).shape)

np.save("horizontal_flow.npy", hflow)
np.save("vertical_flow.npy", vflow)
print("Saved horizontal_flow.npy and vertical_flow.npy successfully.")
