import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

os.chdir("/app/")

def remove_noise(inverted_skeleton):
    temp = np.array(inverted_skeleton[:])
    temp = np.array(temp)
    binary_image = temp / 255
    filtered_image = np.array(binary_image)
    
    enhanced_img = np.array(temp)
    window = np.zeros((10, 10))
    W, H = temp.shape[:2]
    filter_size = 6

    for i in range(W - filter_size):
        for j in range(H - filter_size):
            window = binary_image[i:i + filter_size, j:j + filter_size]

            flag = 0
            if sum(window[:, 0]) == 0:
                flag += 1
            if sum(window[:, filter_size - 1]) == 0:
                flag += 1
            if sum(window[0, :]) == 0:
                flag += 1
            if sum(window[filter_size - 1, :]) == 0:
                flag += 1
            if flag > 3:
                filtered_image[i:i + filter_size, j:j + filter_size] = np.zeros((filter_size, filter_size))

    return filtered_image

def extract_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = np.array(img, dtype=np.uint8)
    
    # Threshold
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = remove_noise(skeleton)
    
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    
    # Extract keypoints
    keypoints = []
    for x in range(harris_normalized.shape[0]):
        for y in range(harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
    
    # Define descriptor
    orb = cv2.ORB_create()
    _, des = orb.compute(img, keypoints)
    return keypoints, des

def main():
    image1_name = sys.argv[1]
    img1 = cv2.imread("database/" + image1_name, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = extract_descriptors(img1)
    
    image2_name = sys.argv[2]
    img2 = cv2.imread("database/" + image2_name, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = extract_descriptors(img2)
    
    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
    
    # Plot keypoints
    img_keypoints1 = cv2.drawKeypoints(img1, kp1, outImage=None)
    img_keypoints2 = cv2.drawKeypoints(img2, kp2, outImage=None)
    
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_keypoints1)
    axarr[1].imshow(img_keypoints2)
    plt.show()
    
    # Plot matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    plt.imshow(img_matches)
    plt.show()
    
    # Calculate score
    score = sum(match.distance for match in matches)
    score_threshold = 33
    
    if score / len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise
