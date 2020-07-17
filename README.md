# Gamma Correction of Ego Lane ONLY

1. Download on clone this repository
2. Navigate to the repo on your hard disk on Anaconda Prompt using cd
3. Run **python classgamma.py**


The keras trained model is optimized to run on GPU

To modify screen capture parameters implemented using PyWin32
**Line 116**, which by default is:
screen = grab_screen(region=(0, 40, 1000, 600))