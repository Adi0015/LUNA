import tkinter as tk
import time
import os

class Spotlight(tk.Tk):
    def __init__(self):
        super().__init__()

        # Remove the default window border and title bar
        # self.overrideredirect(True)
      
        # Set the window size and position
        self.geometry("800x75")

        # Set window to always stay on top
        self.attributes('-topmost', True)

        # Set the window to be transparent
        self.attributes('-alpha', 0.5)

        # Create a text box with black background and white font
        self.textbox = tk.Text(self, bg='black', fg='white', font=('Helvetica', 14), width=150, height=3,state='normal')

        # Position the text box in the center of the screen
        self.textbox.pack(expand=True, fill='both')

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (800 // 2) # Set the x-coordinate to the center of the screen
        y = (screen_height // 2) - (75 // 2) # Set the y-coordinate to the center of the screen
        self.geometry(f"+{x}+{y}") # Set the window position to the center of the screen
        self.attributes('-type', 'splash')
        
        
    def show(self):
        # Minimize all windows
        os.system('xdotool key Super_L+d')
        time.sleep(0.2) # Wait a little bit to ensure all windows are minimized

        # Show the Spotlight window
        self.update_idletasks() # Force update the window
        self.attributes('-topmost', True) # Set the window to always stay on top
        self.attributes('-alpha', 0.5)
        self.lift() # Bring the window to the front of all other windows
        self.deiconify() # Display the window
        

if __name__ == '__main__':
    app = Spotlight()
    app.show()
    app.mainloop()
