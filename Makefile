
# PETSC_DIR  = /path/to/your/petsc
# PETSC_ARCH = your-petsc-arch


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


SOURCEC     = mms.c
EXECUTABLE  = mms



all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCEC) 
	-${CLINKER} -o $@ $< ${PETSC_LIB}

clean::
	-${RM} ${EXECUTABLE} *.o

.PHONY: all clean
