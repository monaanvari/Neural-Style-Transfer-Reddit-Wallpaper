
import sys
import requests
import subprocess

# from gi.repository import Gio

def verifySubreddit(sub):
    sub.strip()
    URL = 'https://www.reddit.com/r/{}/'.format(sub)
    result = requests.get(URL, headers = {'User-agent': 'getWallpapers'})
    return result.status_code != 404

def getSubreddits():
    print('\nWELOCME!\n')
    print('This program performs neural style transfer to compose an image from one of your favorite '+
            'subreddits in the style of an image from a different subreddit to create unique wallpapers.'+ 
            '\nPress any key to quit.\n')
    sub1=''
    sub2 = ''
    while True:    
        sub1 = input('\nWhich subreddit do you want the program to take the main image from? \nSome '+ 
                     'recommendations are EarthPorn, spaceporn, wallpapers,...')
        sub2 = input('\nWhich subreddit do you want the program to take the style image from? \nSome '+
                     'recommendations are Art, comicbookart, conceptart, ...')
        if verifySubreddit(sub1) and verifySubreddit(sub2):
            return sub1, sub2
    print('The subreddit does not exist. Please try again...')

def getDesktopEnvironment():
    if sys.platform == "darwin":
        return "mac"
    else: 
        desktop_session = os.environ.get("DESKTOP_SESSION")
        if desktop_session is not None: 
            desktop_session = desktop_session.lower()
            if desktop_session in ["gnome","unity", "cinnamon", "mate", "xfce4", "lxde", "fluxbox", 
                                   "blackbox", "openbox", "icewm", "jwm", "afterstep","trinity", "kde"]:
                return desktop_session    

def setWallpaper(file_location):

    desktop_env = getDesktopEnvironment()
    try:
        if desktop_env in ['gnome', 'unity', 'cinnamon']:
            '''From https://github.com/eko5/Bing-Wallpaper/blob/36ee74edd1a6d51ae84c5df67e3fa3b4d05b8253/bingbackground.py'''
            uri = "'file://%s'" % file_location
            '''try:
                SCHEMA = 'org.gnome.desktop.background'
                KEY = 'picture-uri'
                gsettings = Gio.Settings.new(SCHEMA)
                gsettings.set_string(KEY, uri)
            except:
                args = ['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', uri]
                subprocess.Popen(args)'''

        elif desktop_env=="windows": 
           '''From https://stackoverflow.com/questions/1977694/change-desktop-background'''
           import ctypes
           SPI_SETDESKWALLPAPER = 20
           ctypes.windll.user32.SystemParametersInfoA(SPI_SETDESKWALLPAPER, 0, file_location , 0)
        elif desktop_env=="mac":

            script = 'tell application "System Events" to tell every desktop to set picture to "{}"'.format(file_location)

            proc = subprocess.Popen(['osascript', '-e'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
            stdout_output = proc.communicate(script)[0]

        else:
            sys.stderr.write("Your desktop environment is not supported.")
            sys.stderr.write("You can try manually to set Your wallpaper to %s" % file_loc)
            return False
        return True
    except:
        return False


