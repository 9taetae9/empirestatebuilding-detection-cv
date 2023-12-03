import sys
import numpy as np
import cv2


# 영상 불러오기

src1 = cv2.imread('empire_image.jpg',cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('empire_template2.jpg',cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
feature = cv2.KAZE_create()
#feature = cv2.AKAZE_create()
#feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(src1, None)
kp2, desc2 = feature.detectAndCompute(src2, None)

# 특징점 매칭
matcher=cv2.BFMatcher_create()
# matcher=cv2.BFMatcher_create(cv2.NORM_HAMMING)
matches=matcher.match(desc1,desc2)

# 좋은 매칭 결과 선별
matches=sorted(matches,key=lambda x: x.distance)
good_matches=matches[:80]

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))

# 호모그래피 계산  queryIdx=1번 이미지에서의 index, trainIdx=2번 이미지에서의 index
#reshaoe(-1,1,2) --> (N,1,2)로 만든다  여기서는 (80,1,2)가 될것임
pts1=np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float32)    #(80,2)
pts2=np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float32)

H,_=cv2.findHomography(pts1,pts2,cv2.RANSAC)


# 호모그래피를 이용하여 기준 영상 영역 표시
dst=cv2.drawMatches(src1,kp1,src2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Calculate corners for the bounding box on src1
(h, w) = src1.shape[:2]
corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)

print('corners1.shape:', corners1.shape, "H.shape:", H.shape)
corners2 = cv2.perspectiveTransform(corners1, H)

# Draw the bounding box on src1
cv2.polylines(src1, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

# Now create the combined image with matches and the bounding box
dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()