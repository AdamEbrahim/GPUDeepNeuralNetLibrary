CC = nvcc
SOURCE = src/lib/
BUILD = build/
DEPS = $(SOURCE)CostFunction.cuh $(SOURCE)GPUNeuralNetwork.cuh $(SOURCE)Image.cuh $(SOURCE)Layer.cuh \
	   $(SOURCE)Matrix.cuh $(SOURCE)SigmoidLayer.cuh

objects = $(BUILD)main.o $(BUILD)CostFunction.o $(BUILD)GPUNeuralNetwork.o $(BUILD)Image.o $(BUILD)Layer.o \
		  $(BUILD)Matrix.o $(BUILD)SigmoidLayer.o

main: $(objects)
	$(CC) -o main $^

$(BUILD)%.o : $(SOURCE)%.cu $(DEPS)
	@mkdir -p $(BUILD)
	$(CC) -c $> -o $@

clean : 
	rm main $(objects)
