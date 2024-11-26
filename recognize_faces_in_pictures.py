import face_recognition
import dlib
import pickle


# Load the jpg files into numpy arrays
pic = face_recognition.load_image_file('pic12.jpg')
face_detector = dlib.get_frontal_face_detector()
all_file_names = []
names = ['Manish', 'Tiya', 'Briti', 'Kirti', 'Hasini', 'Hari', 'Hm']
for name in names:
    for i in range(15):
        if name == 'Tiya':
            file_names_1 = []
            a = 'Tiya'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_1.append(file_name)
            all_file_names.append(file_names_1)
        if name == 'Manish':
            file_names_2 = []
            a = 'Manish'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_2.append(file_name)
            all_file_names.append(file_names_2)
        if name == 'Briti':
            file_names_3 = []
            a = 'Briti'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_3.append(file_name)
            all_file_names.append(file_names_3)
        if name == 'Kirti':
            file_names_4 = []
            a = 'Kirti'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_4.append(file_name)
            all_file_names.append(file_names_4)
        if name == 'face_recognition.load_image_file(fn)':
            file_names_5 = []
            a = 'Hasini'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_5.append(file_name)
            all_file_names.append(file_names_5)
        if name == 'Hari':
            file_names_6 = []
            a = 'Hari'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_6.append(file_name)
            all_file_names.append(file_names_6)
        if name == 'Hm':
            file_names_7 = []
            a = 'Hm'
            file_name = "data/faces/" + a.lower() + "/" + a + "/ (" + str(i+1) + ").jpg"
            file_names_7.append(file_name)
            all_file_names.append(file_names_7)

for fn in all_file_names:
    if name == 'Tiya':
        Tiya_image = face_recognition.load_image_file(fn[0])
        Tiya_face_encoding = face_recognition.face_encodings(Tiya_image)[0]
    if name == 'Manish':
        Manish_image = face_recognition.load_image_file(fn[0])
        Manish_face_encoding = face_recognition.face_encodings(Manish_image)[0]
    if name == 'Briti':
        Briti_image = face_recognition.load_image_file(fn[0])
        Briti_face_encoding = face_recognition.face_encodings(Briti_image)[0]
    if name == 'Kirti':
        Kirti_image = face_recognition.load_image_file(fn[0])
        Kirti_face_encoding = face_recognition.face_encodings(Kirti_image)[0]
    if name == 'Hasini':
        Hasini_image = face_recognition.load_image_file(fn[0])
        Hasini_face_encoding = face_recognition.face_encodings(Hasini_image)[0]
    if name == 'Hari':
        Hari_image = face_recognition.load_image_file(fn[0])
        Hari_face_encoding = face_recognition.face_encodings(Hari_image)[0]
    if name == 'Hm':
        Hm_image = face_recognition.load_image_file(fn[0])
        Hm_face_encoding = face_recognition.face_encodings(Hm_image)[0]

faces = face_detector(pic)

for face in faces:
    file = open('data/unknown', 'wb')
    unknown_image = face_recognition.load_image_file(pickle.dump(face, file))
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

known_faces = [
    Tiya_image,
    Manish_image,
    Briti_image,
    Kirti_image,
    Hasini_image,
    Hari_image,
    Hm_image
]

results =  face_recognition.compare_faces(known_faces, unknown_face_encoding)

for kf in known_faces:
    if unknown_image == kf:
        print (kf)
    else:
        print ("Stranger")



# # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
# results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
#
# print("Is the unknown face a picture of Tiya? {}".format(results[0]))
# print("Is the unknown face a picture of Manish? {}".format(results[1]))
# print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))