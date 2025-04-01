import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Execute YOLO on the frame
    results = model(frame)

    alert = False
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":  # Check if there is a person in the frame
                alert = True

    # Annotate the frame to be able to write to image
    annotated_frame = results[0].plot()

    # Add alert message if a person is detected
    if alert:
        cv2.putText(
            annotated_frame,
            "ALERT: Person detected!",
            (frame.shape[0] // 2, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            5,
        )
    
    cv2.imshow("Watching", annotated_frame)
    
    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Closing webcam...")
        break

cap.release()
cv2.destroyAllWindows()
