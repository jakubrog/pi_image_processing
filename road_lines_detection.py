import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None:
        return


    line_dict = {'left': [], 'right': []}
    img_center = img.shape[1]/2

    img = np.copy(img)
    line_img = np.zeros_like(img)



    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < img_center and x2 < img_center:
                position = 'left'
            elif x1 > img_center and x2 > img_center:
                position = 'left'
            #line_dict[position].append(x1, y1)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process(img):
    image = img

    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
    print(region_of_interest_vertices)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,
        theta=np.pi / 60,
        threshold=100,
        lines=np.array([]),
        minLineLength=100,
        maxLineGap=50
    )

    #print(lines)
    line_img = draw_lines(image, lines)
    return line_img


def main():
    #cap = VideoStream(src=0).start()
    cap = cv2.VideoCapture(r'C:\Users\Mateusz\pi_image_processing\video\solidWhiteRight.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        frame = process(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()