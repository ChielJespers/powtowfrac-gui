all:
	nvcc -Xcompiler -fPIC -shared -lgd -lm -ldl -o powtowfrac.so main.cu