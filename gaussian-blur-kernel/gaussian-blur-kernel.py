import math

def gaussian_kernel(size: int, sigma: float) -> list:
    center = size // 2
    kernel = []
    total = 0.0

    for row in range(size):
        kernel_row = []
        for column in range(size):
            x_offset = column - center
            y_offset = row - center
            exponent = -(x_offset ** 2 + y_offset ** 2) / (2 * sigma ** 2)
            weight = math.exp(exponent)
            kernel_row.append(weight)
            total += weight
        kernel.append(kernel_row)

    for row in range(size):
        for column in range(size):
            kernel[row][column] = kernel[row][column] / total

    return kernel