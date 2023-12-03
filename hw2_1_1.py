import cv2
import numpy as np

# Load the main image and the template
#main_image = cv2.imread('shanghai_image.jpg')  # Replace with your image file
main_image = cv2.imread('empire_image5.jpg')  # Replace with your image file

template_image = cv2.imread('empire_template2.jpg')  # Replace with your template file

# Convert images to grayscale
gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
keypoints_main, descriptors_main = sift.detectAndCompute(gray_main, None)
keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)

# Create a FLANN Matcher object
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict()

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors_main, descriptors_template, k=2)

# Apply Lowe's ratio test to find good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print('len(good_matches) :',len(good_matches))

# Proceed only if enough good matches are found
if len(good_matches) > 300:
    src_pts = np.float32([keypoints_main[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Check if a valid homography was found
    if M is not None:
        h, w = gray_template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        main_image = cv2.polylines(main_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        print("True - Empire State Building is present")
    else:
        print("False - Empire State Building is not present")
else:
    print("False - Empire State Building is not present")

# Draw matches
matched_image = cv2.drawMatches(main_image, keypoints_main, template_image, keypoints_template, good_matches, None, flags=2)

# Display the matched image
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
