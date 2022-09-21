# Arshdeep Singh
# 2020CSB1074

# importing required modules
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage.color import *
from skimage import feature,metrics,filters


#--------------------------utility functions--------------------------#

# utility functions to display image
def displayImg(image,r=1,c=1,msg='Image'):
  fig, axes = plt.subplots(r, ncols=c, figsize=(16, 8))
  axes.imshow(image,cmap='gray')
  axes.axis('off')
  axes.set_title(msg)
  plt.show()

# Input image is an array of images
def displayALL(image,msg=['Image']):
  if len(image)==1:
    displayImg(image[0],msg[0])
    return
  fig, axes = plt.subplots(1, ncols=len(image), figsize=(6, 6))
  for img in range(len(image)):
    axes[img].imshow(image[img],cmap='gray')
    axes[img].axis('off')
    text = 'Image'
    if img < len(image): 
      text = msg[img]
    axes[img].set_title(text)
  plt.show()


#--------------------------Gaussian Filter--------------------------#

#fetching desired gaussian kernel
def get_Gaussiankernel(sigma=1):
    size = 2*sigma+1
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*sigma**2))*(np.exp(-((i - (sigma)) ** 2 + (j - (sigma)) ** 2) / (2 * sigma ** 2)))
    ans = kernel / np.sum(kernel)
    return ans

# performing convolution
def convolution(image, kernel):
  size = int(len(kernel))
  sigma = 1
  sigma = int((size-1)/2)
  img = np.pad(image, (sigma, sigma), 'edge')
  H,W = img.shape
  new_image = np.full(img.shape,0.0)
  for i in range(sigma, H-sigma):
        for j in range(sigma, W-sigma):
          small_img = img[i-sigma:i+sigma+1,j-sigma:j+sigma+1]
          new_image[i,j] = np.sum(small_img*kernel)
  new_image = new_image[sigma:H-sigma, sigma:W-sigma]
  return new_image


# gausian filter main function
def Gaussian_Filter(image,sigma=1):
  kernel = get_Gaussiankernel(sigma)
  convolved_image = convolution(image,kernel)
  return convolved_image



#---------------------------------sobel filter-----------------------------------#

# for calculating direction of gradient
def sobel_direction(Gx,Gy):
  H,W = Gx.shape
  dir = np.zeros([H,W])
  for i in range(H):
    for j in range(W):
      angle = 90
      if Gx[i,j]!=0: 
        angle = np.rad2deg(np.arctan2(Gy[i,j],Gx[i,j]))
      if angle<0: angle = angle+180
      dir[i,j] = angle
  return dir

# for calculating magnitude of gradient
def sobel_magnitude(Gx,Gy):
  H,W = Gx.shape
  mag = np.zeros([H,W])
  for i in range(H):
    for j in range(W):
      mag[i,j] = np.sqrt((Gx[i,j]**2) + (Gy[i,j]**2))
  return mag


# sobel filter main function
def sobel_edge_detection(image):
  kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
  img = np.copy(image)

  Gy = convolution(img,kernel_y)
  Gx = convolution(img,kernel_x)

  G = sobel_magnitude(Gx,Gy)
  Theta = sobel_direction(Gx,Gy)
  
  return G,Theta


#------------------------------Non-Maximum Suppression--------------------------#

def non_maximal_supperssion(G,Theta):
  H,W = G.shape
  new_G = np.zeros(G.shape)
  for i in range(1,H-1):
    for j in range(1,W-1):
      nbr1 = 0
      nbr2 = 0
      Theta[i,j] = abs(Theta[i,j])
      #angle 0
      if (0 <= Theta[i,j] < 22.5) or (157.5 <= Theta[i,j] <= 180):
          nbr2 = G[i, j+1]
          nbr1 = G[i, j-1]
      #angle 45
      elif (22.5 <= Theta[i,j] < 67.5):
          nbr2 = G[i-1, j+1]
          nbr1 = G[i+1, j-1]
      #angle 90
      elif (67.5 <= Theta[i,j] < 112.5):
          nbr2 = G[i-1, j]
          nbr1 = G[i+1, j]
      #angle -45
      elif (112.5 <= Theta[i,j] < 157.5):
          nbr2 = G[i+1, j+1]
          nbr1 = G[i-1, j-1]

      if (G[i,j] >= nbr2) and (G[i,j] >= nbr1):
          new_G[i,j] = G[i,j]
      else:
          new_G[i,j] = 0
  return new_G


#------------------------------Double Thresholding--------------------------#
weak = 100
strong = 255

def Double_Thresholding(img, lowThreshold=0.3, highThreshold=0.4):
    
    H, W = img.shape
    threshold_img = np.full((H,W),0)
  
    # classifying pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img <= lowThreshold)
    weak_i, weak_j = np.where((img < highThreshold) & (img > lowThreshold))
    
    # assigning values
    threshold_img[strong_i, strong_j] = strong
    threshold_img[weak_i, weak_j] = weak
    threshold_img[zeros_i, zeros_j] = 0
    
    return threshold_img


#------------------------------Hysteresis--------------------------#

def hysteresis(G):
  H = len(G)
  W = len(G[0])
  new_G = np.zeros([H,W])
  G = np.pad(G,((1,1),(1,1)),'edge')
  dr = np.array([-1,-1,0,1,1,1,0,-1])
  dc = np.array([0,1,1,1,0,-1,-1,-1])
  for i in range(H):
    for j in range(W):
      if G[i,j] == weak:
        # check if any of the neighbours are strong
        flag = 0
        for k in range(8):
          if G[i+dr[k],j+dc[k]]>=strong:
            flag = 1
            break
        
        if flag == 1:
          # make it strong
          new_G[i,j] = strong
        else:
          # make it zero
          new_G[i,j] = 0

      else:
        new_G[i,j] = G[i,j]

  return new_G

#---------------------------------------Canny Edge Detection-----------------------------#
def myCannyEdgeDetector(image,Low_Threshold,High_Threshold):
  # convert to grayscale
  image = rgb2gray(image)

  #calling gaussian filter              
  sigma = 2
  gaussian_image = Gaussian_Filter(image,sigma)

  #calling sobel filter
  G, Theta = sobel_edge_detection(gaussian_image)

  #calling original sobel for comparison
  org_sobel = filters.sobel(image)

  #calling non maximal supperssion
  nms_G = non_maximal_supperssion(G,Theta)

  #calling double thresholding
  # print("Max: ",np.max(nms_G),np.min(nms_G))
  threshold_image = Double_Thresholding(nms_G,Low_Threshold,High_Threshold)

  #calling hysteresis
  final_image = hysteresis(threshold_image)

  #calling canny edge detection from skimage
  original_canny = feature.canny(image, sigma,low_threshold=Low_Threshold, high_threshold=High_Threshold)

  #plotting images
  displayALL([image,gaussian_image],['Original Input Image','After Gaussian Blur'])
  displayALL([org_sobel,G],['Inbuilt Sobel','My Sobel'])
  # displayALL([G,Theta],['Magnitude of G','Direction of G'])
  # displayALL([nms_G],['Non Maximal Supperssion'])
  # displayALL([threshold_image],[f'Thresholding with low threshold = {ll} and high threshold = {hh}'])
  displayALL([original_canny,final_image],['Canny from skimage','My Canny Egde Detection'])

  print(f'Thresholding with low threshold = {Low_Threshold} and high threshold = {High_Threshold}')
  #calculating psnr and ssim

  bool_img = final_image==255

  print("PNSR of Image is ",metrics.peak_signal_noise_ratio(original_canny,bool_img))
  print("SSIM of Image is ",metrics.structural_similarity(original_canny,bool_img)*100)



#---------------------------------------Main-----------------------------#
image = io.imread('./images/abd.jpg')
myCannyEdgeDetector(image,0.28,0.38)