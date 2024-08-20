import cv2
import numpy as np


# Correct the paths if necessary
age_net = cv2.dnn.readNetFromCaffe(r"C:\Users\DELL\AI & ML Projects\ML Project 02\models\deploy_age.prototxt",
                                   r"C:\Users\DELL\AI & ML Projects\ML Project 02\models\age_net.caffemodel")

# Load the pre-trained models
age_net = cv2.dnn.readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel")

# Define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Load an image from disk
image = cv2.imread("path/to/your/image.jpg")
(h, w) = image.shape[:2]

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

# Pass the blob through the network to obtain the age predictions
age_net.setInput(blob)
age_preds = age_net.forward()

# Find the age bucket with the highest probability
age = AGE_BUCKETS[age_preds[0].argmax()]
print(f"Predicted age: {age}")

# Draw the predicted age on the image
cv2.putText(image, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
