all:
	nvcc -Xcompiler -fPIC -shared -o powtowfrac.so main.cu