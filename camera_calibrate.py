<<<<<<< HEAD

=======
>>>>>>> 4efdfd727cc47ba7033ff5d85262cf6e7b395142
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt


def calibrate(Showpics=True):
  #read images  

  #initialize
  nRows = 8
  nCols = 6
  termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,25,0.001)
  worldptscor = np.zeros((nRows*nCols,3),np.float32)
  worldptscor[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
  worldptslist = []
  imgptslist = []



  for img_path in glob.glob("cali_imgs/*"):
    imgBGR = cv2.imread(img_path)
    imgGray = cv2.cvtColor(imgBGR,cv2.COLOR_BGRA2GRAY)    
    cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray,(nRows,nCols),None)
    if cornersFound:
      cornersRefined = cv2.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
      worldptslist.append(worldptscor)
      imgptslist.append(cornersRefined)
      if Showpics:
        cv2.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
        cv2.imshow('ChessBoard',imgBGR)
        cv2.waitKey(500)
  cv2.destroyAllWindows()

   ##calibrate
  repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldptslist,imgptslist,imgGray.shape[::-1],None,None)
  print("Camera Matix\n",camMatrix)
  print("ReProj Error (pixels) {:4f}".format(repError))

  #save calibration parameters
  np.savez('calibration.npz',camMatrix=camMatrix,distCoeff=distCoeff,rvecs=rvecs,tvecs=tvecs)

  return camMatrix,distCoeff


      


def removedistortion(camMatrix,distCoeff):
  root = os.getcwd()
  imgPath  = os.path.join(root,"images/distortions")
  img = cv2.imread(imgPath)

  height,width = img.shape[:2]
  camMatrixNew, roi = cv2.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height))
  imgUndist = cv2.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

  #draw a line to see the distortion change
  cv2.line(img,(1769,103),(1780,922),(255,255,255),2)
  cv2.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)
  plt.figure()
  plt.subplot(121)
  plt.imshow(img)
  plt.subplot(122)
  plt.imshow(imgUndist)
  plt.show()




def runCalibration():
  calibrate(Showpics=True)

runCalibration()



<<<<<<< HEAD

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt


def calibrate(Showpics=True):
  #read images  

  #initialize
  nRows = 8
  nCols = 6
  termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,25,0.001)
  worldptscor = np.zeros((nRows*nCols,3),np.float32)
  worldptscor[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
  worldptslist = []
  imgptslist = []



  for img_path in glob.glob("cali_imgs/*"):
    imgBGR = cv2.imread(img_path)
    imgGray = cv2.cvtColor(imgBGR,cv2.COLOR_BGRA2GRAY)    
    cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray,(nRows,nCols),None)
    if cornersFound:
      cornersRefined = cv2.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
      worldptslist.append(worldptscor)
      imgptslist.append(cornersRefined)
      if Showpics:
        cv2.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
        cv2.imshow('ChessBoard',imgBGR)
        cv2.waitKey(500)
  cv2.destroyAllWindows()

   ##calibrate
  repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldptslist,imgptslist,imgGray.shape[::-1],None,None)
  print("Camera Matix\n",camMatrix)
  print("ReProj Error (pixels) {:4f}".format(repError))

  #save calibration parameters
  np.savez('calibration.npz',camMatrix=camMatrix,distCoeff=distCoeff,rvecs=rvecs,tvecs=tvecs)

  return camMatrix,distCoeff


      


def removedistortion(camMatrix,distCoeff):
  imgPath  = ("cali_imgs\processed-0BF013EB-4837-4FBE-BDFF-769070C1F7D9.jpeg")
  img = cv2.imread(imgPath)

  height,width = img.shape[:2]
  camMatrixNew, roi = cv2.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height))
  imgUndist = cv2.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

  print("New Camera Matix\n",camMatrixNew)

  #draw a line to see the distortion change
  cv2.line(img,(1769,103),(1780,922),(255,255,255),2)
  cv2.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)
  plt.figure()
  plt.subplot(121)
  plt.imshow(img)
  plt.subplot(122)
  plt.imshow(imgUndist)
  plt.show()




def runCalibration():
  calibrate(Showpics=True)

def runRemoveDistortion():
  camMatrix, distCoeff = calibrate(Showpics=False)
  removedistortion(camMatrix,distCoeff)



runRemoveDistortion()

=======
>>>>>>> 4efdfd727cc47ba7033ff5d85262cf6e7b395142
