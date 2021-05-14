
def get_screenWH():
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

    except:
        import pyautogui
        WIDTH, HEIGHT = pyautogui.size()
    #return WIDTH, HEIGHT
#print(resolution)

#WIDTH, HEIGHT = get_screenWH()
print('Width, Height:')
print(WIDTH, HEIGHT)
