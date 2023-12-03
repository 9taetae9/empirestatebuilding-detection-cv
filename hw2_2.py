import cv2
import numpy as np
import sys

# Load the main image and the template
main_image = cv2.imread('empire_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image file
template_image = cv2.imread('empire_template2.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your template file

if main_image is None or template_image is None:
    print('Image load failed!')
    sys.exit()

# Initialize AKAZE detector
feature = cv2.KAZE_create()

# Detect keypoints and compute descriptors
kp1, desc1 = feature.detectAndCompute(main_image, None)
kp2, desc2 = feature.detectAndCompute(template_image, None)

# Feature matching
matcher = cv2.BFMatcher_create()
matches = matcher.match(desc1, desc2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:80]  # Select top 80 matches

# Proceed only if enough good matches are found
if len(good_matches) > 50:  # Adjust the threshold as needed
    # Extract location of good matches
    points_main = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_template = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(points_template, points_main, cv2.RANSAC, 5.0)

    # Check if a valid homography was found
    if M is not None:
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        main_image = cv2.polylines(main_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

# Draw matching results
matched_image = cv2.drawMatches(main_image, kp1, template_image, kp2, good_matches, None)

# Display the matched image
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', matched_image)
cv2.waitKey()
cv2.destroyAllWindows()
