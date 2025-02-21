import cv2
import numpy as np

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

image = cv2.imread("person.jpg")
height, width, _ = image.shape

inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward
nPoints = 15
threshold = 0.1
points = []

for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (width * point[0]) / output.shape[3]
    y = (height * point[1]) / output.shape[2]
    if prob > threshold:
        points.append((int(x), int(y)))
    else:
        points.append(None)

for point in points:
    if point:
        cv2.circle(image, point, 5, (0, 255, 255), -1)

cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
