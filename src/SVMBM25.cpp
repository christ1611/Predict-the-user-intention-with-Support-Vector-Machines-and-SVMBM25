/*
#####################################################################################
#Program name  : SVMBM25.cpp
#Description   : Building the LIBSVM format for test file and train file by using the BM25 measurement
#Copyright     : Distribution, modification, reproduction, and copy are strictly prohibited to unauthorized persons.
#Author        : Hotama Christianus Frederick (christianusfrederick@gmail.com)
#Commit date   : December 2018
######################################################################################
*/

#include "SVMBM25.h"

#ifdef __MINGW32__

#elif __GNUC__
#include <sys/mman.h>
#else
#endif
#include<iostream>

int main(void) {
    string directory;
    DIR *dir;
    std::string starray[10000];
    struct dirent *ent;
    vector<string> src_train_files,src_test_files;

    directory="./train_file";
    if ((dir = opendir (directory.c_str())) != NULL)
        {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL)
            {
                std::string output=ent->d_name;
                std::string file=directory+"/"+output;
                char first=output[0];
                if (first!='.' )
                {
                    src_train_files.push_back(file);
                    cout<<file<<endl;
                }

            }
            closedir (dir);
        }
        else
        {
  		/* could not open directory */
            perror ("Please copy the sample file to the folder");
        }

    directory="./test_file";
    if ((dir = opendir (directory.c_str())) != NULL)
        {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL)
            {
                std::string output=ent->d_name;
                std::string file=directory+"/"+output;
                char first=output[0];
                if (first!='.' )
                {
                    src_test_files.push_back(file);
                    cout<<file<<endl;
                }


            }
            closedir (dir);
        }
        else
        {
  		/* could not open directory */
            perror ("Please copy the sample file to the folder");
        }

	const string dst_train_file = "./output/train.svm";
	const string dst_test_file = "./output/test.svm";
	BM25 bm25;
	TermVoca voc;

	const int32_t kThreads = 0;
	double set_time = 0;

	set_time = omp_get_wtime();

	InitBM25Core(src_train_files, bm25, voc, kThreads);
	BuildSVMDataSet(src_train_files, dst_train_file, bm25, voc, kThreads);
	BuildSVMDataSet(src_test_files, dst_test_file, bm25, voc, kThreads);

	fprintf(stdout, "Runtime=%f\n", omp_get_wtime() - set_time);

	return 0;
}

/* Compute components and set parameters for BM25 scoring. */
void InitBM25Core(const vector<string>& src_file, BM25& bm25, TermVoca& voc, const int32_t& num_threads) {
	const int32_t kWordSize = 4096;
	const int32_t kNumThreads = num_threads ? num_threads : omp_get_num_procs(); //The number of threads can be adjusted due to total physical cores.

	int64_t sum_bytes = 0;
	int64_t num_docs = 0; //The total number of documents.
	int64_t num_lines = 0; //The total number of lines.

#ifndef O_BINARY
#define O_BINARY 0x0000
#endif
#ifndef O_LARGEFILE
#define O_LARGEFILE 0x0000
#endif

	/* Compute parameters for document scoring and generate an indexed vocabulary from input data. */

	for (size_t label = 0; label < src_file.size(); ++label) {
		int32_t src_fd = open(src_file[label].c_str(), O_RDONLY | O_BINARY | O_LARGEFILE);
		if (src_fd == -1) {
			fmt::fprintf(stdout, "Error: File open error. Program terminated.\n");
			exit(EXIT_FAILURE);
		}

		/* Split a file for multi-threading considering line feed. */
		int32_t thr_cnt = -1;
		int64_t thr_set_pos[kNumThreads] = {0};
		int64_t thr_end_pos[kNumThreads] = {0};
		int64_t thr_seg_len[kNumThreads] = {0};
		int64_t fix_pos = 0;
		int64_t file_size = lseek64(src_fd, 0, SEEK_END);
		char* cur_mmf_ptr = nullptr;
		char* set_mmf_ptr = static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, src_fd, 0));
		if (set_mmf_ptr == MAP_FAILED) {
			fmt::fprintf(stdout, "Error: Memory mapping failed. Program terminated.\n");
			exit(EXIT_FAILURE);
		}

		for (int32_t i = 0; i < kNumThreads; ++i) {
			int64_t tmp_set_pos = file_size * i / kNumThreads;
			int64_t tmp_end_pos = min(file_size * (i + 1) / kNumThreads, file_size);
			int64_t tmp_seg_len = tmp_end_pos - tmp_set_pos;


			if (tmp_seg_len <= 0) {
				thr_set_pos[i] = thr_end_pos[i] = thr_seg_len[i] = 0;
				continue;
			}

			tmp_set_pos = fix_pos;
			tmp_end_pos -= 1;
			cur_mmf_ptr = set_mmf_ptr + tmp_end_pos;
			fix_pos = tmp_end_pos;
			int64_t sav_pos = tmp_end_pos;

			while (sav_pos <= file_size - 1) {
				if (*cur_mmf_ptr == '\n' || sav_pos == file_size - 1) {
					tmp_end_pos = sav_pos;
					fix_pos = sav_pos;
					break;
				}
				cur_mmf_ptr++;
				sav_pos++;
			}

			fix_pos++;
			tmp_seg_len = tmp_end_pos - abs(tmp_set_pos) + 1;

			if (tmp_seg_len > 0) {
				thr_set_pos[i] = tmp_set_pos;
				thr_end_pos[i] = tmp_end_pos;
				thr_seg_len[i] = tmp_seg_len;
			}
		}

		/* 1-Pass. */
		#pragma omp parallel reduction(+:num_docs, num_lines) num_threads(kNumThreads)
		{
			int32_t thr_idx = 0;

			#pragma omp critical
			{
				thr_idx = ++thr_cnt;

			}

			fmt::fprintf(stdout, "1-Pass:%d:TID=%d\n", label, thr_idx);

			char* data_ptr = set_mmf_ptr + thr_set_pos[thr_idx];
			char str_buf[kWordSize + 1];
			int32_t str_length = 0;
			int32_t doc_length = 0;
			int64_t loc_bytes = 0;
			bool flg_eof = false;
			hash_map_pair loc_map_voc;
			hash_set loc_set_voc;

			while(loc_bytes <= thr_seg_len[thr_idx]) {
				flg_eof = (loc_bytes == thr_seg_len[thr_idx]);
				if (*data_ptr == ' ' ||
					*data_ptr == '\r' ||
					*data_ptr == '\n' ||
					*data_ptr == '\t' ||
					flg_eof) {
					if (str_length) {
						str_buf[str_length] = '\0';

						if (!(loc_map_voc.insert({str_buf, TermPair(1, 1)}).second)) { //Insertion statement for standard hash map.
							if (loc_set_voc.insert(str_buf).second)
								loc_map_voc[str_buf].docs++;
							loc_map_voc[str_buf].freq++;
						} else
							loc_set_voc.insert(str_buf);

						doc_length++;
						str_length = 0;
					}
					if (*data_ptr == '\n') {
						if (doc_length) {
							num_docs++;
							doc_length = 0;
						}
						num_lines++;
						hash_set().swap(loc_set_voc);
					}
					if (flg_eof) {
						if (file_size == loc_bytes + thr_set_pos[thr_idx]) {
							if (doc_length) {
								num_docs++;
								doc_length = 0;
							}
							num_lines++;
						}
						break;
					}
				} else {
					if (str_length < kWordSize)
						str_buf[str_length++] = *data_ptr;
					else {
						fmt::fprintf(stdout, "Error: Buffer overrun. Program terminated.\n");
						exit(EXIT_FAILURE);
					}
				}
				data_ptr++;
				loc_bytes++;
			}

			/* Iteration statement for standard hash map. */
			#pragma omp critical
			{
				for (const auto& i : loc_map_voc) {
					if (!(voc.map.insert({i.first, TermInfo(i.second.freq, i.second.docs, 0)}).second)) {
						auto& v = voc.map[i.first];
						v.freq += i.second.freq;
						v.docs += i.second.docs;
					}
				}
				sum_bytes += loc_bytes;
			}
		}

		munmap(set_mmf_ptr, file_size);
		close(src_fd);
	}

	int64_t term_freq = 0; //The total number of words. The sum of term frequencies.
	int64_t term_docs = 0; //The total number of documents containing each term.
	int32_t idx_cnt = 0;
	voc.ptr.reserve(voc.map.size());

	/* Summation statement for standard map. */
	for (auto& i : voc.map) {
		term_freq += i.second.freq;
		term_docs += i.second.docs;
		i.second.idx = idx_cnt++;
		voc.ptr.emplace_back(&i);
	}

	bm25.SetParameter(0.75f, 1.0f, 1.2f, num_docs, term_freq); //b, d, k1, num_docs, term_freq

	/* Report 1-pass results. */
	fmt::fprintf(stdout, "Term=%llu:%llu:%d\n", voc.map.size(), voc.ptr.size(), idx_cnt);
	fmt::fprintf(stdout, "Term Freq=%lld\n", term_freq);
	fmt::fprintf(stdout, "Term Docu=%lld\n", term_docs);
	fmt::fprintf(stdout, "Byte=%lld\n", sum_bytes);
	fmt::fprintf(stdout, "Docu=%lld\n", num_docs);
	fmt::fprintf(stdout, "Line=%lld\n", num_lines);
	fmt::fprintf(stdout, "AVDL=%f\n", term_freq / (double)num_docs);
	fmt::fprintf(stdout, "MMF based multi-threaded term counting process completed.\n");

	return;
}

/* Build SVM data set with BM25 feature from raw corpus. */
void BuildSVMDataSet(const vector<string>& src_file, const string& dst_file, const BM25& bm25, const TermVoca& voc, const int32_t& num_threads) {
	const int32_t kWordSize = 4096;
	const int32_t kNumThreads = num_threads ? num_threads : omp_get_num_procs(); //The number of threads can be adjusted due to total physical cores.

	int64_t sum_bytes = 0;
	int64_t num_docs = 0; //The total number of documents.
	int64_t num_lines = 0; //The total number of lines.

	int32_t dst_fd = open(dst_file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_BINARY | O_LARGEFILE, 0644);
	if (dst_fd == -1) {
		fmt::fprintf(stdout, "Error: File create error. Program terminated.\n");
		exit(EXIT_FAILURE);
	}

	/* Transform raw data into training data format with BM25 score feature. */
	for (size_t label = 0; label < src_file.size(); ++label) {
		int32_t src_fd = open(src_file[label].c_str(), O_RDONLY | O_BINARY | O_LARGEFILE);
		if (src_fd == -1) {
			fmt::fprintf(stdout, "Error: File open error. Program terminated.\n");
			exit(EXIT_FAILURE);
		}

		/* Split a file for multi-threading considering line feed. */
		int32_t thr_cnt = -1;
		int64_t thr_set_pos[kNumThreads] = {0};
		int64_t thr_end_pos[kNumThreads] = {0};
		int64_t thr_seg_len[kNumThreads] = {0};
		int64_t fix_pos = 0;
		int64_t file_size = lseek64(src_fd, 0, SEEK_END);
		char* cur_mmf_ptr = nullptr;
		char* set_mmf_ptr = static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, src_fd, 0));
		if (set_mmf_ptr == MAP_FAILED) {
			fmt::fprintf(stdout, "Error: Memory mapping failed. Program terminated.\n");
			exit(EXIT_FAILURE);
		}

		for (int32_t i = 0; i < kNumThreads; ++i) {
			int64_t tmp_set_pos = file_size * i / kNumThreads;
			int64_t tmp_end_pos = min(file_size * (i + 1) / kNumThreads, file_size);
			int64_t tmp_seg_len = tmp_end_pos - tmp_set_pos;

			if (tmp_seg_len <= 0) {
				thr_set_pos[i] = thr_end_pos[i] = thr_seg_len[i] = 0;
				continue;
			}

			tmp_set_pos = fix_pos;
			tmp_end_pos -= 1;
			cur_mmf_ptr = set_mmf_ptr + tmp_end_pos;
			fix_pos = tmp_end_pos;
			int64_t sav_pos = tmp_end_pos;

			while (sav_pos <= file_size - 1) {
				if (*cur_mmf_ptr == '\n' || sav_pos == file_size - 1) {
					tmp_end_pos = sav_pos;
					fix_pos = sav_pos;
					break;
				}
				cur_mmf_ptr++;
				sav_pos++;
			}

			fix_pos++;
			tmp_seg_len = tmp_end_pos - abs(tmp_set_pos) + 1;

			if (tmp_seg_len > 0) {
				thr_set_pos[i] = tmp_set_pos;
				thr_end_pos[i] = tmp_end_pos;
				thr_seg_len[i] = tmp_seg_len;
			}
		}

		/* 2-Pass. */
		#pragma omp parallel reduction(+:num_docs, num_lines) num_threads(kNumThreads)
		{
			int32_t thr_idx = 0;

			#pragma omp critical
			{
				thr_idx = ++thr_cnt;
			}

			fmt::fprintf(stdout, "2-Pass:%d:TID=%d\n", label, thr_idx);

			fmt::memory_buffer fmt_str_buf;
			char* data_ptr = set_mmf_ptr + thr_set_pos[thr_idx];
			char str_buf[kWordSize + 1];
			int32_t str_length = 0;
			int32_t doc_length = 0; //The number of words in documents. Document length.
			int64_t loc_bytes = 0;
			bool flg_eof = false;
			hash_map loc_map_voc;

			auto SaveIndexSortedDocumentScore = [&]() -> void {
				hash_map_info::const_iterator v;
				fmt::format_to(fmt_str_buf, "{0}", label);
				std::vector<uint32_t> idx_vec_voc;
				idx_vec_voc.reserve(loc_map_voc.size());
				for (const auto& i : loc_map_voc) {
					if ((v = voc.map.find(i.first)) != voc.map.end())
						idx_vec_voc.emplace_back(v->second.idx);
				}

				std::sort(idx_vec_voc.begin(), idx_vec_voc.end());
				hash_map::const_iterator i;
				for (const auto& h : idx_vec_voc) {

					i = loc_map_voc.find(voc.ptr[h]->first);

					v = voc.map.find(i->first);

					fmt::format_to(fmt_str_buf, " {0}:{1:.8g}", v->second.idx + 1, bm25.GetScore(i->second, doc_length, v->second.docs)); //Index starting from 1.
				}

				fmt::format_to(fmt_str_buf, "\n");
				#pragma omp critical
				{
					if (write(dst_fd, fmt::to_string(fmt_str_buf).c_str(), fmt::to_string(fmt_str_buf).length()) == -1) {
						fmt::fprintf(stdout, "Error: File write error. Program terminated.\n");
						exit(EXIT_FAILURE);
					}
				}
			};

			while(loc_bytes <= thr_seg_len[thr_idx]) {
				flg_eof = (loc_bytes == thr_seg_len[thr_idx]);
				if (*data_ptr == ' ' ||
					*data_ptr == '\r' ||
					*data_ptr == '\n' ||
					*data_ptr == '\t' ||
					flg_eof) {
					if (str_length) {
						str_buf[str_length] = '\0';

						if (!(loc_map_voc.insert({str_buf, 1}).second)) //Insertion statement for standard hash map.
							loc_map_voc[str_buf]++;

						doc_length++;
						str_length = 0;
					}
					if (*data_ptr == '\n') {
						if (doc_length) {
							SaveIndexSortedDocumentScore();
							fmt_str_buf.clear();
							num_docs++;
							doc_length = 0;
						}
						num_lines++;
						hash_map().swap(loc_map_voc);
					}
					if (flg_eof) {
						if (file_size == loc_bytes + thr_set_pos[thr_idx]) {
							if (doc_length) {
								SaveIndexSortedDocumentScore();
								num_docs++;
								doc_length = 0;
							}
							num_lines++;
						}
						break;
					}
				} else {
					if (str_length < kWordSize)
						str_buf[str_length++] = *data_ptr;
					else {
						fmt::fprintf(stdout, "Error: Buffer overrun. Program terminated.\n");
						exit(EXIT_FAILURE);
					}
				}
				data_ptr++;
				loc_bytes++;
			}

			/* Summation statement for standard hash map. */
#pragma omp atomic
			sum_bytes += loc_bytes;
		}

		munmap(set_mmf_ptr, file_size);
		close(src_fd);
	}

	/* Report 2-pass results. */
	fmt::fprintf(stdout, "Byte=%lld\n", sum_bytes);
	fmt::fprintf(stdout, "Docu=%lld\n", num_docs);
	fmt::fprintf(stdout, "Line=%lld\n", num_lines);
	fmt::fprintf(stdout, "SVM data set building process completed.\n");

	close(dst_fd);
	std::system("./svm-scale -l 0 -u 1 -s train.svm.scale.cfg output/train.svm > output/train.svm.scale");
    std::system("./svm-scale -r train.svm.scale.cfg output/test.svm > output/test.svm.scale");
    std::system("./train -s 1 -c 0.125 output/train.svm.scale output/train.svm.scale.model");
    std::system("./predict output/test.svm.scale output/train.svm.scale.model output/train.svm.scale.predict");
	return;
}
