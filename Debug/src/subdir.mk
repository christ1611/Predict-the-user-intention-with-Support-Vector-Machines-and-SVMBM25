################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SVMBM25.cpp 

OBJS += \
./src/SVMBM25.o 

CPP_DEPS += \
./src/SVMBM25.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -DFMT_HEADER_ONLY=1 -DFARMHASH_ASSUME_AESNI=1 -D_GLIBCXX_USE_C99=1 -D_GLIBCXX_HAS_GTHREADS=1 -D_GLIBCXX__PTHREADS=1 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1 -D_LARGEFILE64_SOURCE=1 -D__USE_LARGEFILE64=1 -I"D:\work\source\x86_64\workspace\SVMBM25\src\lib" -O3 -Wall -c -fmessage-length=0 -fopenmp -msse4.1 -maes -fexceptions -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


