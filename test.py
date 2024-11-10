from processor import *
import cv2 as cv


with Verifier() as verifier:

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    identified = False


    while True:
        ret, frame = cap.read()

        if not ret:
            print("cannot read image")
            break


        ebd = verifier.get_embedding(image=frame, from_array=True)

        if not identified and len(ebd) > 0:
            for e in ebd:
                idt, dist = verifier.identify(e)
                print(idt, dist)
