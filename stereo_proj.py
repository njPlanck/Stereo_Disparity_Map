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
WIN_SIZE = 3


def window(img_pixel,i,j):
  if i <=0 | j <=0 | j==pr_y|i==pr_x:
    img_mean_pixel = img_pixel[i][j]
  else:
    win_div = [img_pixel[i][j],img_pixel[i][j+1],img_pixel[i+1][j],img_pixel[i-1][j],img_pixel[i][j-1],img_pixel[i-1][j-1],img_pixel[i+1][j+1],img_pixel[i-1][j+1],img_pixel[i+1][j-1]]
    wind_std = np.std(win_div)
    if wind_std >= 8:
      img_mean_pixel = round(np.mean([img_pixel[i][j],img_pixel[i][j+1],img_pixel[i+1][j],img_pixel[i-1][j],img_pixel[i][j-1],img_pixel[i-1][j-1],img_pixel[i+1][j+1],img_pixel[i-1][j+1],img_pixel[i+1][j-1]]))
    else:
      img_mean_pixel = img_pixel[i,j]
      WIN_SIZE = 5
      for x in range(1,WIN_SIZE-2):
        img_mean_pixel += sum([img_pixel[i][j+x],img_pixel[i+x][j],img_pixel[i-x][j],img_pixel[i][j-x],img_pixel[i-x][j-x],img_pixel[i+x][j+x],img_pixel[i-x][j+x],img_pixel[i+x][j-x]])
      img_mean_pixel /= WIN_SIZE**2

  return img_mean_pixel
  
    #return np.abs(left_mean_pixel - right_mean_pixel)**2

    #return np.abs(left_mean_pixel - right_mean_pixel)**2
    #return (left_mean_pixel*right_mean_pixel)/np.sqrt((left_mean_pixel**2)*(right_mean_pixel**2))



disp_image = np.zeros((pl_x,pr_y),np.float32)
disp_image_filter = np.zeros((pl_x,pr_y),np.float32)
for i in range(pr_x-WIN_SIZE):
  for j in range(pl_y-WIN_SIZE):
    left_pixel = window(left_img,i,j)
    right_pixel = window(right_img,i,j)
    if left_pixel != right_pixel:
      disp_image[i][j] = round((10*1.45)/np.abs(left_pixel - right_pixel))
    else:
      disp_image[i][j] = -1
  


plt.imshow(disp_image)
plt.show()


     
     




