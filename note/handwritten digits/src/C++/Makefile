all:handwritten_digits.cpp
	@if [ ! -e  t10k-labels-idx1-ubyte  ]; then gzip -dc ../../data/t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte;fi
	@if [ ! -e  t10k-images-idx3-ubyte  ]; then gzip -dc ../../data/t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte;fi
	@if [ ! -e  train-images-idx3-ubyte  ]; then gzip -dc ../../data/train-images-idx3-ubyte.gz > train-images-idx3-ubyte;fi
	@if [ ! -e  train-labels-idx1-ubyte  ]; then gzip -dc ../../data/train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte; fi
	g++ -I ../../../../component/eigen handwritten_digits.cpp -o bin
	./bin

phony: clean

clean:
	@rm bin;
	@rm t10k-labels-idx1-ubyte;
	@rm t10k-images-idx3-ubyte;
	@rm train-images-idx3-ubyte;
	@rm train-labels-idx1-ubyte;


