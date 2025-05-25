gpu:
	nvcc -lboost_program_options -o task task.cu
	./task --size=256 --epsilon=0.000001 --max_iter=1000000
profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true -o profile ./task --size=512 --epsilon=0.000001 --max_iter=1000000