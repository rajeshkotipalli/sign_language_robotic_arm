import cv2

print("Testing with DirectShow backend...")

for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"SUCCESS — Camera {i} works with DSHOW! Frame size: {frame.shape}")
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print(f"Camera {i} opened but still cannot read frame")
        cap.release()
    else:
        print(f"No camera at index {i}")