import cv2

# Global variables
roi_points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(roi_points) > 1:
            # Close the polygon by connecting last to first point
            cv2.line(param, roi_points[-1], roi_points[0], (0, 255, 0), 2)
            cv2.imshow("Draw ROI - Press ESC to finish", param)
        drawing = False

def draw_roi(image_path):
    global roi_points
    image = cv2.imread(image_path)
    clone = image.copy()

    cv2.namedWindow("Draw ROI - Press ESC to finish")
    cv2.setMouseCallback("Draw ROI - Press ESC to finish", mouse_callback, param=clone)

    while True:
        temp = clone.copy()
        # Draw lines between points
        for i in range(1, len(roi_points)):
            cv2.line(temp, roi_points[i - 1], roi_points[i], (0, 0, 255), 2)
        for point in roi_points:
            cv2.circle(temp, point, 3, (255, 0, 0), -1)

        cv2.imshow("Draw ROI - Press ESC to finish", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
    return roi_points

# Example usage:
if __name__ == "__main__":
    image_path = "/Users/adarshgowda/pro/PPE_4/BML2_Video3-avi_frame_38_jpg.rf.a7a2308bbbf8bef785f09e672e616e7e.jpg"  # Replace with your image path
    points = draw_roi(image_path)
    print("ROI Points:", points)
