import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
a function that reads the images and also 
coverts it to an array of intensity values.
otherwise known as gray values.
'''

left_img_path="view_l.png"
right_img_path="view_r.png"

left_img = cv2.imread(left_img_path,0)
right_img = cv2.imread(right_img_path,0)
left_img = np.array(left_img,dtype=np.float64)
right_img = np.array(right_img,dtype=np.float64)

pr_x , pr_y = right_img.shape
pl_x , pl_y = left_img.shape
img_disp = np.zeros((pr_x,pl_y))
idx = 3

def window(left_pixel, right_pixel, i,j):
  left_mean_pixel = 0
  right_mean_pixel = 0
  if i <=0 | j <=0 | j==pr_x|i==pr_y:
    left_mean_pixel = np.mean([left_img[i][j]*(idx-1),left_img[i+idx][j],left_img[i][j],left_img[i][j+idx],left_img[i][j]*(idx-1)],dtype=np.float32)
    right_mean_pixel = np.mean([right_img[i][j]*(idx-1),right_img[i+idx][j],right_img[i][j],right_img[i][j+idx],right_img[i][j]*(idx-1)],dtype=np.float32)
    return np.abs(left_mean_pixel - right_mean_pixel)
    #return np.abs(left_mean_pixel - right_mean_pixel)**2
    #return (left_mean_pixel*right_mean_pixel)/np.sqrt((left_mean_pixel**2)*(right_mean_pixel**2))
  else:
    for x in range(idx-1):
      left_mean_pixel += left_img[i+x][j]+left_img[i-x][j]+left_img[i][j+1]+left_img[i][j-x]
      right_mean_pixel += right_img[i+x][j]+right_img[i-x][j]+right_img[i][j+1]+right_img[i][j-x]
    left_mean_pixel += left_img[i][j]
    right_mean_pixel += left_img[i][j]
    left_mean_pixel /= idx**2
    right_mean_pixel /= idx**2
  return np.abs(left_mean_pixel - right_mean_pixel)
  #return np.abs(left_mean_pixel - right_mean_pixel)**2
  #return (left_mean_pixel*right_mean_pixel)/np.sqrt((left_mean_pixel**2)*(right_mean_pixel**2))


for i in range(pr_x-idx):
  for j in range(pl_y-idx):
    if left_img[i][j]!=right_img[i][j]:
      img_disp[i][j] = window(left_img,right_img,i,j)

      #sum of absolute difference
      '''
      else:
        img_disp[i][j] = abs(np.mean([left_img[i][j],left_img[i+1][j],left_img[i-1][j],left_img[i][j+1],left_img[i][j-1]],dtype=np.float32)-right_img[i][j])
      '''
      

      
      '''

      if i <=0 | j <=0:
        img_disp[i][j] = abs(np.max([left_img[i][j],left_img[i+1][j],left_img[i][j],left_img[i][j+1],left_img[i][j]])-right_img[i][j])

      #sum of absolute difference
      else:
        img_disp[i][j] = abs(np.max([left_img[i][j],left_img[i+1][j],left_img[i-1][j],left_img[i][j+1],left_img[i][j-1]])-right_img[i][j])
      '''
      ##sum of squared distance
      #img_disp[i][j] = (left_img[i][j]-right_img[i][j])**2
      ##zero mean SAD
      '''
      mean = (left_img[i][j]+right_img[i][j])/2
      left_img[i][j] = abs(left_img[i][j]-mean)
      right_img[i][j] = abs(right_img[i][j]-mean)
      img_disp[i][j] = left_img[i][j]-right_img[i][j]
      '''

    else:
      img_disp[i][j] = 0






plt.imshow(img_disp)
plt.show()


'''

def cal_disparity(left_img, right_img):
  pl_x,pl_y = left_img.shape


'''



  
