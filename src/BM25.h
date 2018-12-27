/*
#####################################################################################
#Program name  : BM25.h
#Description   : This file contains the formula declaration to get the score of BM25 from each documents
#Copyright     : Distribution, modification, reproduction, and copy are strictly prohibited to unauthorized persons.
#Author        : Hotama Christianus Frederick (christianusfrederick@gmail.com)
#Commit date   : December 2018
######################################################################################
*/

#ifndef SRC_BM25_H_
#define SRC_BM25_H_

#include <math.h>

#include "Types.h"

/* Define BM25 class. */
class BM25 {
private:
	int64_t num_docs = 0;
	int64_t term_freq = 0;
	double b = 0.75f;
	double d = 1.0f;
	double k1 = 1.2f;
	double avdl = 0.0f;

public:
	void SetParameter(const double& _b, const double& _d, const double& _k1, const int64_t& _num_docs, const int64_t& _term_freq) {
		b = _b;
		d = _d;
		k1 = _k1;
		num_docs = _num_docs;
		term_freq = _term_freq;
		avdl = _term_freq / (double)_num_docs;
	}

	double GetScore(const uint32_t& _doc_term_freq, const uint32_t& _doc_length, const uint32_t& _num_term_docs) const {
		return ((((k1 + 1.0f) * _doc_term_freq) / (k1 * (1.0f - b + b * (_doc_length / avdl)) + _doc_term_freq)) + d) * log((num_docs + 1.0f) / _num_term_docs);
	}

	double GetScore(const uint32_t& _doc_term_freq, const uint32_t& _doc_length, const uint32_t& _num_term_docs, \
					const double& _b, const double& _d, const double& _k1, const int64_t& _num_docs, const int64_t& _term_freq) const {
		return ((((_k1 + 1.0f) * _doc_term_freq) / (_k1 * (1.0f - _b + _b * (_doc_length / (_term_freq / (double)_num_docs))) + _doc_term_freq)) + d) * log((_num_docs + 1.0f) / _num_term_docs);
	}
};

#endif /* SRC_BM25_H_ */
