import cv2
import time

def main():
	image = cv2.imread('0.jpg')
	cv2.imshow('a',image)
	cv2.waitKey(0)
	#time.sleep(5)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()