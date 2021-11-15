import cv2
import time


def contour_detect(frame1, frame2):
        """
        frame1 matrix type
        frame2 matrix type
        :rtype: array
        """
        diff = cv2.absdiff(frame1,
                       frame2)  # seeking the distinction between two frames which appears only if one has changed i.e since that moment programme reacts to the movement

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # converting frames to the black&white gradation

        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # filtering out faux contours

        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # method for higlighting the edge of an object with white colour

        dilated = cv2.dilate(thresh, None, iterations=3)  # expands the area

        сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)  # finding array of contour points
        return сontours


def main():
    cap = cv2.VideoCapture(0)  # video stream from a web camera

    cap.set(3, 1280)  # setting the window size
    cap.set(4, 700)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    mode = True  # True = red, False = green
    timing = time.time()
    while cap.isOpened():  # method isOpened() returns status of a videostream
        сontours = contour_detect(frame1, frame2)
        for contour in сontours:
            # method contourArea() by the given contour points, calculates an area of a fixated object in every moment of time
            print(cv2.contourArea(contour))

            if cv2.contourArea(contour) < 700:  # condition under which the area of highlited object is less than 700 px
                continue

            if time.time() - timing > 5.0:  # if 5 sec passed:
                timing = time.time()  # update time
                mode = not mode
            if mode:
                cv2.putText(frame1, "Status: {}".format("Movement"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.drawContours(frame1, сontours, -1, (0, 255, 0), 2)

            else:
                cv2.drawContours(frame1, сontours, -1, (0, 0, 255), 2)
                cv2.putText(frame1, "Status: {}".format("Movement"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                            cv2.LINE_AA)

        cv2.imshow("frame1", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()