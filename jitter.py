import numpy as np
import random

# Jitter an image
# Returns several jittered versions of the input image
def jitterImage(frame,faces):
  # Define constants
  numShiftMax = 4;  # Number of shifted images to produce
  numColorMax = 6;  # Number of color-shifted images to produce
  maxShift = 0.1 # Maximum pixel displacement in x and y directions
  maxColorShift = 30; # Raw pixel shift

  # Frame width and height
  fw = frame.shape[1]
  fh = frame.shape[0]

  x,y,w,h = faces[0]

  frames = []; # Will hold output jittered images

  # Return original unjittered image
  # frames.append(frame[y:y+h,x:x+h])

  # Shift image by up to 10% of cropbox size in each direction
  shiftCount = 0
  while shiftCount < numShiftMax:
    # Generate shifts:    -0.1 < shift < .1
    xshift = np.random.uniform(0.0,maxShift*2) - maxShift
    yshift = np.random.uniform(0.0,maxShift*2) - maxShift

    # Apply shifts
    xt = x + int(xshift*w)
    yt = y + int(yshift*h)

    # Verify shifts are within limits
    if xt >= 0 and yt >= 0 and xt+w < fw and yt+h < fh:
      # New values are ok
      frames.append(frame[yt:yt+h,xt:xt+w])
      shiftCount += 1

  # Brighten or darken image uniformly
  # Raw pixel values are 0 to 255
  for i in range(numColorMax):
    shift = random.randint(0,2*maxColorShift) - maxColorShift/2
    ftmp = frame.astype(np.int) + shift
    
    # Make sure ftmp does not exceed 0 and 255
    ftmp[ftmp < 0] = 0
    ftmp[ftmp > 255] = 255

    # Add new image to output
    ftmp = ftmp.astype(np.uint8)
    frames.append(ftmp[yt:yt+h,xt:xt+w])

  return frames