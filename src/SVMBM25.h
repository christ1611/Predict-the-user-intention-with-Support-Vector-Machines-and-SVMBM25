/*
#####################################################################################
#Program name  : SVMBM25.h
#Description   : All the declaration that needed for SVMBM25.cpp
#Copyright     : Distribution, modification, reproduction, and copy are strictly prohibited to unauthorized persons.
#Author        : Hotama Christianus Frederick (christianusfrederick@gmail.com)
#Commit date   : December 2018
######################################################################################
*/
#include <unistd.h>
#ifndef SRC_SVMBM25_H_
#define SRC_SVMBM25_H_

#define _FILE_OFFSET_BITS 64
#define _LARGE_FILE_SUPPORT 1
#define _LARGEFILE64_SOURCE 1
#define _LARGEFILE_SOURCE 1
#define __USE_LARGEFILE64 1

#include <math.h>
#include <fcntl.h>
#include <omp.h>

#include <string>
#include <vector>
#include <dirent.h>
#include "lib/fmt/format.h"
#include "lib/fmt/format-inl.h"
#include "lib/fmt/printf.h"

#include "BM25.h"
#include "Types.h"

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
using namespace std;

void InitBM25Core(const vector<string>& src_file, BM25& bm25, TermVoca& voc, const int32_t& num_threads = 0);
void BuildSVMDataSet(const vector<string>& src_file, const string& dst_file, const BM25& bm25, const TermVoca& voc, const int32_t& num_threads = 0);

#endif /* SRC_SVMBM25_H_ */
