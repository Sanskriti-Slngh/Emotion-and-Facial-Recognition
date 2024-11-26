import numpy as np
import cv2
import pickle
import face_recognition

img = cv2.imread("test_pics/pic7.jpg")

known_people = ["Manish", "Kriti", "Tiya", "Briti", "Hm", "hasini", "Hari"]

face_location = face_recognition.face_locations(img)
print(face_location)

face_embeddings = face_recognition.face_encodings(img, face_location)

X = np.transpose(np.array(face_embeddings))

def predict(model, X):
    w = model['weight']
    b = model['bias']
    z = np.dot(w, X) + b
    y_hat = 1/(1 + np.exp(-z))
    max_y_hat = max(y_hat, axis = 0, keepdims = True)
    p = max_y_hat == y_hat
    return (p)

with open("modelLinearRegression_MKTB, 'rb") as fin:
    model = pickle.load(fin)

P = predict[model , X]
names = []
rows, columns = P.shape
for col in range(columns):
    for row in range(rows):
        if P[row, col]:
            person = known_people[min(len(known_people) - 1)]
            names.append(person)
            break

## display the information
# display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
    cv2.rectangle(image, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255),1)

# Open a window on the desktop showing the image
win = dlib.image_window()
win.set_image(image)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()
