import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def detect_borders():
    video = cv2.VideoCapture("data/video1.mp4")
    has_next, first_frame = video.read()

    if not has_next:
        print("Error loading video.")
        return []

    gray_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 50, 140, apertureSize=3)

    plt.imshow(edges_img, "gray")

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

    # for i in range(1, len(borders)):
    #     plt.plot([borders[i][0], borders[i][2]], [borders[i][1], borders[i][3]], color='red')
    # plt.show()

    return borders[1:4]


def borders_cross(borders, current_pedestrian_position):
    top_border = borders[0]
    left_border = borders[2]
    right_border = borders[1]

    if top_border_cross(top_border, current_pedestrian_position):
        return True

    if left_border_cross(left_border, current_pedestrian_position):
        return True

    if right_border_cross(right_border, current_pedestrian_position):
        return True

    return False


def top_border_cross(top_border , current_pedestrian_position):
    current_pedestrian_x, current_pedestrian_y = get_pedestrian_positions(current_pedestrian_position)

    return (top_border[0] < current_pedestrian_x < top_border[2]) and (
            top_border[3] < current_pedestrian_y < top_border[1])


def left_border_cross(left_border, current_pedestrian_position):
    current_pedestrian_x, current_pedestrian_y = get_pedestrian_positions(current_pedestrian_position)

    return (left_border[3] < current_pedestrian_y < left_border[1]) and (
            left_border[0] < current_pedestrian_x < left_border[2])


def right_border_cross(right_border, current_pedestrian_position):
    current_pedestrian_x, current_pedestrian_y = get_pedestrian_positions(current_pedestrian_position)

    return (right_border[3] < current_pedestrian_y < right_border[1]) and (
            right_border[0] < current_pedestrian_x < right_border[2])


def get_pedestrian_positions(current_pedestrian_position):
    return current_pedestrian_position[0], current_pedestrian_position[1]


def count_pedestrians(borders, path):
    pedestrian_count = 0
    previous_frame = None
    video = cv2.VideoCapture(path)

    step = -1

    while True:
        has_next, current_frame = video.read()

        if not has_next:
            break

        step += 1

        # step = 6 frames at a time
        if step != 6:
            continue

        step = 0

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is None:
            previous_frame = current_frame_gray
            continue

        differences = cv2.absdiff(previous_frame, current_frame_gray)
        binary_differences = cv2.threshold(differences, 17, 255, cv2.THRESH_BINARY)[1]
        binary_differences = cv2.dilate(binary_differences, np.ones((3, 3)), 3)

        previous_frame = current_frame_gray

        contours = cv2.findContours(binary_differences, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        for contour in contours:
            if cv2.contourArea(contour) < 200 or cv2.contourArea(contour) > 2000:
                continue
            m = cv2.moments(contour)
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])

            plt.scatter(c_x, c_y)

            if borders_cross(borders, [c_x, c_y]):
                pedestrian_count += 1

    video.release()
    return pedestrian_count


if __name__ == '__main__':
    borders = detect_borders()
    if len(borders) > 0:
        videos = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4", "video5.mp4", "video6.mp4",
                  "video7.mp4", "video8.mp4", "video9.mp4", "video10.mp4"]

        y_test = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
        y_pred = []

        for video in videos:
            y_pred.append(count_pedestrians(borders, "data/" + video))

        print("MAE: ", mean_absolute_error(y_test, y_pred))
