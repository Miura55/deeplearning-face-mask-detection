import cv2
import numpy as np

sample_image = "examples/example_01.png"
onnx_model_path = 'detector/face_mask_detector.onnx'
prototxt_path = 'detector/deploy.prototxt'
faceDetect_path = 'detector/res10_300x300_ssd_iter_140000.caffemodel'
classes = {0: 'Mask', 1: 'No Mask'}

faceMaskNet = cv2.dnn.readNetFromONNX(onnx_model_path)
faceDetectModel = cv2.dnn.readNetFromCaffe(prototxt_path, faceDetect_path)


def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceDetectModel.setInput(blob)
    detections = faceDetectModel.forward()
    faces = []
    positions = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX-15), max(0, startY-15))
        (endX, endY) = (min(w-1, endX+15), min(h-1, endY+15))
        confidence = detections[0, 0, i, 2]
        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            face = image[startY:endY, startX:endX]
            faces.append(face)
            positions.append((startX, startY, endX, endY))
    return faces, positions


def preprocess(img_data):
    ''' 画像データのスケーリング/正規化 '''
    mean_vec = np.array([0.485, 0.456, 0.406])[::-1]
    stddev_vec = np.array([0.229, 0.224, 0.225])[::-1]
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[2]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[:, :, i] = (
            img_data[:, :, i]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def detect_mask(faces):
    predictions = []
    for face in faces:
        face = cv2.resize(face, (224, 224))
        preprocessed = preprocess(face)
        blob = cv2.dnn.blobFromImage(preprocessed)
        faceMaskNet.setInput(blob)
        pred = np.squeeze(faceMaskNet.forward())
        predictions.append(pred)
    return predictions


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        (faces, postions) = detect_face(img)
        predictions = detect_mask(faces)

        for(box, prediction) in zip(postions, predictions):
            (startX, startY, endX, endY) = box
            max_index = np.argsort(-prediction)[0]
            label = classes[max_index]
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
            text = "{}: {:.2f}%".format(
                label, (1 + prediction[max_index]) * 100)
            cv2.putText(img, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        # Display the resulting frame
        cv2.imshow('Result', img)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
