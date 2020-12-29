import os

import transfer
import urllib
import util

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image
from subreddit import subreddit
from os.path import expanduser

home = expanduser("~")
jsonLimit = 50
min_width=1920
min_height=1080

if __name__== '__main__':


    main, style = util.getSubreddits()
    sub1 = subreddit(main)
    sub2 = subreddit(style)

    mainImageURL = sub1.getValidImage(jsonLimit, min_height, min_width)
    styleImageURL = sub2.getValidImage(jsonLimit, min_height, min_width)

    mainImage = Image.open(urllib.request.urlopen(mainImageURL))
    styleImage = Image.open(urllib.request.urlopen(styleImageURL))

    image = transfer.transfer(mainImageURL, styleImageURL)

    filename = home+"/styledImage.jpg"

    image.save(filename)
    util.setWallpaper(filename)
