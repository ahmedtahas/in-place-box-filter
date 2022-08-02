import cv2
import numpy as np
import matplotlib.pyplot as plt


def in_place_box_filter_without_statement_and_padding():
	image = [255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255,
	         255, 255, 255, 255, 255, 255, 255, 255]
	row = 7
	col = 8

	buffer = [0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]

	def top_left():
		return (image[0] + image[1] + image[col] + image[col + 1]) / 9

	def top_right():
		return (buffer[2 * col - 2] + image[col - 1] + image[2 * col - 2] + image[2 * col - 1]) / 9

	def bottom_left():
		return (buffer[0] + buffer[1] + image[(row - 1) * col] + image[(row - 1) * col + 1]) / 9

	def bottom_right():
		return (buffer[col - 2] + buffer[col - 1] + buffer[2 * col - 2] + image[col * row - 1]) / 9

	def left():
		return (buffer[0] + buffer[1]) / 9

	def right():
		return (buffer[col - 2] + buffer[col - 1] + buffer[2 * col - 2]) / 9

	image[0] = top_left()

	for j in range(1, col - 1):
		image[j] = (buffer[col + j - 1] + image[j] + image[j + 1] + image[j - 1 + col] + image[j + col] + image[
			j + 1 + col]) / 9

	image[col - 1] = top_right()

	for i in range(1, row - 1):

		for j in range(int(len(buffer) / 2)):
			buffer[j] = buffer[col + j]
			buffer[col + j] = image[i * col + j]

		image[i * col] = left() + (
				image[i * col] + image[i * col + 1] + image[i * col + col] + image[i * col + col + 1]) / 9

		for j in range(1, col - 1):
			image[i * col + j] = (buffer[j - 1] + buffer[j] + buffer[j + 1] + buffer[col + j - 1] + image[i * col + j] +
			                      image[
				                      i * col + j + 1] + image[i * col + j - 1 + col] + image[i * col + j + col] + image[
				                      i * col + j + 1 + col]) / 9

		image[i * col + col - 1] = right() + (
				image[(i + 1) * col - 1] + image[(i + 2) * col - 2] + image[(i + 2) * col - 1]) / 9

	image[(row - 1) * col] = bottom_left()

	for j in range(1, col - 1):
		image[(row - 1) * col + j] = (buffer[j - 1] + buffer[j] + buffer[j + 1] + buffer[j + col - 1] + image[
			(row - 1) * col + j] + image[(row - 1) * col + j + 1]) / 9

	image[(row - 1) * col + col - 1] = bottom_right()

	for i in range(row):
		for j in range(col):
			print(str(image[i * col + j]) + " ", end="")
		print()


def in_place_box_filter_without_padding(image, kernel_size):
	buffer = np.zeros(image.shape[1] * 2, np.float32)
	line = image.shape[1]
	d = kernel_size ** 2
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			buffer[j] = buffer[j + line]
		for j in range(image.shape[1]):
			buffer[j + line] = image[i][j]
		for j in range(image.shape[1]):
			if i == 0 and j == 0:
				image[i][j] = round(image[i][j] / d + image[i][j + 1] / d + image[i + 1][j] / d + image[i + 1][j + 1] / d)

			elif i == 0 and j == image.shape[1] - 1:
				image[i][j] = round(buffer[j - 1 + line] / d + image[i][j] / d + image[i + 1][j - 1] / d + image[i + 1][j] / d)

			elif i == image.shape[0] - 1 and j == 0:
				image[i][j] = round(buffer[j] / d + buffer[j + 1] / d + image[i][j] / d + image[i][j + 1] / d)

			elif i == image.shape[0] - 1 and j == image.shape[1] - 1:
				image[i][j] = round(buffer[j - 1] / d + buffer[j] / d + buffer[j - 1 + line] / d + image[i][j] / d)

			elif i == 0:
				image[i][j] = round(
					buffer[j - 1 + line] / d + image[i][j] / d + image[i][j + 1] / d + image[i + 1][j - 1] / d
					+ image[i + 1][j] / d + image[i + 1][j + 1] / d)

			elif i == image.shape[0] - 1:
				image[i][j] = round(
					buffer[j - 1] / d + buffer[j] / d + buffer[j + 1] / d + buffer[j - 1 + line] / d
					+ image[i][j] / d + image[i][j + 1] / d)

			elif j == 0:
				image[i][j] = round(
					buffer[j] / d + buffer[j + 1] / d + image[i][j] / d + image[i][j + 1] / d + image[i + 1][j] / d
					+ image[i + 1][j + 1] / d)

			elif j == image.shape[1] - 1:
				image[i][j] = round(
					buffer[j - 1] / d + buffer[j] / d + buffer[j - 1 + line] / d + image[i][j] / d
					+ image[i + 1][j - 1] / d + image[i + 1][j] / d)
			else:
				image[i][j] = round(
					buffer[j - 1] / d + buffer[j] / d + buffer[j + 1] / d + buffer[j - 1 + line] / d + image[i][j] / d
					+ image[i][j + 1] / d + image[i + 1][j - 1] / d + image[i + 1][j] / d + image[i + 1][j + 1] / d)

	return image


def in_place_box_filter_with_padding(image, kernel_size):
	buffer = np.zeros(image.shape[1] * 2, np.float32)
	image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	image = np.float32(image)
	row = int(buffer.shape[0] / 2)
	divider = 1 / (kernel_size ** 2)
	for i in range(1, image.shape[0] - 1):
		for j in range(row):
			buffer[j] = buffer[j + row]
			buffer[j + row] = image[i][j + 1]
		for j in range(row):
			pixel = buffer[j] + image[i][j + 1] + image[i][j + 2] + image[i + 1][j] + image[i + 1][j + 1] + image[i + 1][
				j + 2]
			if 0 < j:
				pixel += buffer[j - 1] + buffer[j + row - 1]
			if j < row - 1:
				pixel += buffer[j + 1]
			image[i][j + 1] = round(pixel * divider)
	return image[1:image.shape[0] - 1, 1:image.shape[1] - 1]


lena_grayscale = cv2.imread("lena_grayscale_hq.jpg", 0)
plt.imshow(lena_grayscale, cmap="gray")
plt.show()
box_filtered = in_place_box_filter_without_padding(lena_grayscale, 3)
plt.imshow(box_filtered, cmap="gray")
plt.show()
lena_grayscale = cv2.imread("lena_grayscale_hq.jpg", 0)
zero_padding = cv2.copyMakeBorder(lena_grayscale, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                  value=0)
opencv_box_filtered = cv2.boxFilter(zero_padding, -1, (3, 3),
                                    borderType=cv2.BORDER_CONSTANT)
opencv_box_filtered = opencv_box_filtered[1:-1, 1:-1]
opencv_box_filtered = np.asarray(opencv_box_filtered, np.float32)
diff = np.abs(box_filtered - opencv_box_filtered, dtype=np.float32)
print(np.sum(diff) / diff.size)
plt.imshow(diff, cmap="gray")
plt.show()
