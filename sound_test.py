import pyttsx3
from time import sleep

engine = pyttsx3.init()

engine.say("Hello, "+"DADA" + " you can enter now.")
try:
	engine.endLoop()
except:
	pass
engine.runAndWait()
