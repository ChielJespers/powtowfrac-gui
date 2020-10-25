all:
	nvcc -g -o powtowfrac main.cu
shared:
	nvcc -Xcompiler -fPIC -shared -o powtowfrac.so main.cu