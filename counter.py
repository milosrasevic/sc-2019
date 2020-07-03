import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error


def detect_borders():
    video = cv2.VideoCapture("data/video1.mp4")
    has_next, first_frame = video.read()

    if not has_next:
        print("Error loading video.")
        return []

    gray_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 50, 140, apertureSize=3)

    lines = cv2.HoughLinesP(image=edges_img, rho=0.5, theta=np.pi / 360, threshold=10, lines=np.array([]),
                            minLineLength=200, maxLineGap=20)

    borders = []

    for i in range(0, len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        if y1 > 300 and y2 > 300:
            continue
        if 140 < x1 < 500 and x2 < 510 and not (120 < y2 < 430):
            if y1 < 120 and y2 < 120:
                y1 += 50
                y2 += 50
            borders.append([x1, y1, x2, y2])

    return borders[1:4]


def borders_cross(borders, current_pedestrian_position):
    top_border = borders[0]
    left_border = borders[2]
    right_border = borders[1]

    if border_cross(top_border, current_pedestrian_position):
        return True

    if border_cross(left_border, current_pedestrian_position):
        return True

    if border_cross(right_border, current_pedestrian_position):
        return True

    return False


def border_cross(border, current_pedestrian_position):
    current_pedestrian_x, current_pedestrian_y = get_pedestrian_positions(current_pedestrian_position)

    return (border[3] < current_pedestrian_y < border[1]) and (
            border[0] < current_pedestrian_x < border[2])


def get_pedestrian_positions(current_pedestrian_position):
    return current_pedestrian_position[0], current_pedestrian_position[1]


def count_pedestrians(borders, video_number):
    pedestrian_count = 0

    video = cv2.VideoCapture("data/video" + str(video_number) + ".mp4")
    has_next, previous_frame = video.read()
    frame_number = 0
    while True:
        frame_number += 1
        has_next, frame = video.read()

        if not has_next:
            break

        if frame_number % 6 != 0:
            continue

        differences = cv2.absdiff(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY),
                                  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        binary_differences = cv2.dilate(cv2.threshold(differences, 13, 255, cv2.THRESH_BINARY)[1], np.ones((3, 3)), 3)

        found_contours = cv2.findContours(binary_differences, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        for contour in found_contours:
            if cv2.contourArea(contour) < 200 or cv2.contourArea(contour) > 1800:
                continue
            m = cv2.moments(contour)
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])

            if borders_cross(borders, [c_x, c_y]):
                pedestrian_count += 1

        previous_frame = frame

    video.release()
    return pedestrian_count


if __name__ == '__main__':
    borders = detect_borders()
    if len(borders) > 0:
        test_y = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
        y_predicted = []

        for i in range(1, 11):
            count = count_pedestrians(borders, i)
            y_predicted.append(count)
            print("Video " + str(i) + ": " + str(count) + " pedestrians")

        print("MAE: ", mean_absolute_error(test_y, y_predicted))
    else:
        "Error! Could not detect borders."
