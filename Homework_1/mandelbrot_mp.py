# Mike Lape
# CS6068
# This program allows you to calculate and visualize the mandelbrot
# set using both a serial method, and a parallel method that utilizes
# Python's multiprocessing package.

# I relied heavily on the following two examples:
# http://code.seas.harvard.edu/almondpy/almondpy/blobs/master/mandelbrot-multiprocessing.py
# https://www.raspberrypi.org/magpi/multiprocessing-with-python/

from numpy import linspace, reshape
from matplotlib import pyplot
from timeit import default_timer as timer
import multiprocessing as mp

# Resolution
HEIGHT = 2000
WIDTH  = 2000

MAX_ITER = 300

# X and Y range for Mandelbrot
X_MIN = -2.0
X_MAX =  0.5
Y_MIN = -1.25
Y_MAX =  1.25


def mandelbrot(z): # computation for one pixel
    c = z
    for n in range(MAX_ITER):
        if abs(z) > 2: 
            return n   # divergence test
        z = z*z + c
    return MAX_ITER


def serial(X, Y):
    N = []
    for y in Y:
        for x in X:
            z  = complex(x,y)
            N += [mandelbrot(z)]

    return N

def parallel(X, Y):
    #N = []
    p = mp.Pool(processes = 4)
    Z = [complex(x,y) for y in Y for x in X]
    N = p.map(mandelbrot,Z)
    return N
def main():
    X = linspace(X_MIN, X_MAX, WIDTH) 
    Y = linspace(Y_MIN, Y_MAX, HEIGHT)


    print("Starting Timer...")
    start = timer()

    # Run either in serial or parallel (defaults to use all cores)
    # Comment out the one you don't want to run.
    #image = serial(X, Y)
    image = parallel(X, Y)
    dt = timer() - start
    print("Mandelbrot created in %f s" % dt)

    # Reshape for display
    image = reshape(image, (WIDTH,HEIGHT))
    
    # Display image
    pyplot.imshow(image) 
    pyplot.show()

if __name__ == "__main__":
    main()