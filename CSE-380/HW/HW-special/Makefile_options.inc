FC 		:= gfortran


ifneq:
	FFLAGS 	:= -O3 std=f2018
else:
	FFlags  := -g
