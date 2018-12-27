/*
#####################################################################################
#Program name  : Types.h
#Description   : Declaring the variable that needed for BM25 analysis
#Copyright     : Distribution, modification, reproduction, and copy are strictly prohibited to unauthorized persons.
#Author        : Hotama Christianus Frederick (christianusfrederick@gmail.com)
#Commit date   : December 2018
######################################################################################
*/

#ifndef SRC_TYPES_H_
#define SRC_TYPES_H_

#include <stdint.h>

#include <string>
#include <vector>

#include <unordered_map>
#include <unordered_set>

using namespace std;

typedef struct TermPair {
	uint32_t freq; //Term frequency.
	uint32_t docs; //The number of documents containing term.

	TermPair() : freq(0), docs(0) {}

	TermPair(uint32_t _freq, uint32_t _docs) {
		freq = _freq;
		docs = _docs;
	}
} TermPair;

typedef struct TermInfo : TermPair {
	uint32_t idx; //Term index.

	TermInfo() : idx(0) {}

	TermInfo(uint32_t _freq, uint32_t _docs, uint32_t _idx) {
		freq = _freq;
		docs = _docs;
		idx = _idx;
	}
} TermInfo;

typedef std::unordered_map<string, TermPair> hash_map_pair;
typedef std::unordered_map<string, TermInfo> hash_map_info;
typedef std::unordered_map<string, uint32_t> hash_map;
typedef std::unordered_set<string> hash_set;
typedef std::vector<hash_map_info::pointer> hash_vec_ptr;

typedef struct {
	hash_map_info map;
	hash_vec_ptr ptr;
} TermVoca;

#endif /* SRC_TYPES_H_ */
