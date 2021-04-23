all:
	nvcc -Xcompiler -fPIC -shared -lgd -lm -ldl -o powtowfrac.so main.cu
mandelbrot:
	nvcc -Xcompiler -fPIC -shared -lgd -lm -ldl -o powtowfrac.so mandelbrot.cu
