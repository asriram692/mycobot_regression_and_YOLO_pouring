from pymycobot import MyCobot280

mc = MyCobot280("/dev/tty.usbserial-56E30046201", 115200) 


coords = mc.get_coords()

print(f"Target Location -> X: {coords[0]}, Y: {coords[1]}")