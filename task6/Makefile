cpu_sequential:
	pgc++ -o therm_multicore -lboost_program_options -acc=host -Minfo=all therm.cpp
	./therm_multicore --size 512 --epsilon 0.000001 --max_iter 1000000

cpu_multicore:
	pgc++ -o therm_multicore -lboost_program_options -acc=multicore -Minfo=all therm.cpp
	./therm_multicore --size 512 --epsilon 0.000001 --max_iter 1000000

gpu:
	pgc++ -o therm_gpu -lboost_program_options -acc=gpu -Minfo=all therm_optim.cpp
	./therm_gpu --size 1024 --epsilon 0.000001 --max_iter 1000000

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./therm_gpu --size 256 --epsilon 0.0001 --max_iter 50