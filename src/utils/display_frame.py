import numpy as np
import cv2


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Load the .npz file
npz_file = np.load("data/processed/FakeAVCeleb_v1.2/batch_0.npz", allow_pickle=True)

# Loop through all the arrays stored under arr_0, arr_1, etc.
for key in npz_file.files:
    item = npz_file[key].item()

    # If item is a dictionary or structured object
    frames = item['data']  # Extract the actual frames

    frames_unnormalized = (frames * std + mean) * 255.0
    frames_uint8 = np.clip(frames_unnormalized, 0, 255).astype(np.uint8)

    print(f"Displaying frames from key: {key} | shape: {frames.shape}")

    for frame in frames_uint8:
        frame = frame.astype(np.uint8)
        cv2.imshow("Video Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
