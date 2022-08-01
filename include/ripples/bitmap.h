/**
 *  @file Huffman.h
 *  @author Sheng Di
 *  @date Aug., 2016
 *  @brief Header file for the exponential segment constructor.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef RIPPLES_BITMAP_H
#define RIPPLES_BITMAP_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <set>

namespace ripples {

void prtbits(unsigned int a){
    for (int i = 0; i < 32; i++) {
        printf("%d", !!((a << i) & 0x80000000));
    }
    printf("\n");
}

template <typename InItr>
void encodeRR0(InItr in_begin, size_t local_idx, size_t length, size_t n_ints, unsigned int* code_array){
	size_t i = 0;
	size_t vtx_id = 0, byte_offset=0, bit_offset=0;
	unsigned int m = 1;
	for (i = 0;i<length;i++) 
	{
		vtx_id = *(in_begin->begin()+i);
		byte_offset = local_idx / 32;
		bit_offset  = local_idx % 32; 
		m = 1 << bit_offset;
		code_array[vtx_id*n_ints+byte_offset] |= m; //vtx start from zero
	}
}


void countRR0(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, std::vector<size_t> n_ints,
			 size_t *local_m, size_t *local_v, size_t &maxvtx, size_t &maxcnt){
	size_t num_threads = omp_get_max_threads();
	size_t n_xs = n_ints.size();
	size_t* globalcnt=(size_t*)malloc(n_vtx*sizeof(size_t));
	memset(globalcnt, 0, n_vtx*sizeof(size_t));
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t i = 0, j = 0, x = 0; // k=0
  		size_t rank = omp_get_thread_num();
	    size_t* localcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	    size_t local_max = 0, local_vtx = 0;
		// #pragma omp for schedule(static)  		    
		// for(k=0; k<num_threads; k++)
		// {
	    for (i = 0;i<n_vtx;i++) {
			for(x = 0; x < n_xs; x++){
				for (j=0; j<n_ints[x]; j++){
					localcnt[i] += __builtin_popcount(blockR[x][rank][i*n_ints[x] + j]);
				}
			}
		// }
		// // }
		// for (i = 0;i<n_vtx;i++) {
			if (localcnt[i] > local_max){
				local_max = localcnt[i];
				local_vtx = i;
			}
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
		#pragma omp barrier
		#pragma omp critical
	    {	
			std::set<size_t> local_idx_set;
			size_t tmp_key = 0;
			for(int r=0;r<num_threads;r++){
				tmp_key = local_v[r];
				if(local_idx_set.find(tmp_key)==local_idx_set.end()){
					std::cout<<tmp_key<<","<<localcnt[tmp_key]<<", ";
					local_idx_set.insert(tmp_key);
					globalcnt[tmp_key]+=localcnt[tmp_key];
				}
			}
			std::cout<<std::endl;
	    }  
	    free(localcnt);		
	}
    maxvtx=0;
    maxcnt=0;
    for(int i=0;i<n_vtx;i++){
    	if(globalcnt[i]>0){
    		std::cout<<"coutRR0: global[i]="<<i<<", "<<globalcnt[i]<<std::endl;
    	}
        if(globalcnt[i]>maxcnt)  {
          maxcnt=globalcnt[i];
          maxvtx=i;
        }
    }
    std::cout<<"coutRR0: maxk="<<maxvtx<<" maxv="<<maxcnt<<std::endl;
    free(globalcnt);
}

void countRR02(std::vector<std::vector<unsigned int*>> &blockR1, std::vector<unsigned int*> &blockR2, 
			const size_t n_vtx, std::vector<size_t> n_ints1, size_t n_ints2,
			size_t *local_m, size_t *local_v, size_t &maxvtx, size_t &maxcnt){
	size_t num_threads = omp_get_max_threads();
	size_t n_xs = n_ints1.size();
	size_t* globalcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	// memset(globalcnt, 0, n_vtx*sizeof(size_t));
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t i = 0, j = 0,  x = 0; //k = 0,
  		size_t rank = omp_get_thread_num();
	    size_t* localcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	    size_t local_max = 0, local_vtx = 0;
		// #pragma omp for schedule(static)  		    
		// for(k=0; k<num_threads; k++)
		// {
	    for (i = 0;i<n_vtx;i++) {
			for(x = 0; x < n_xs; x++){
				for (j=0; j<n_ints1[x]; j++){
					localcnt[i] += __builtin_popcount(blockR1[x][rank][i*n_ints1[x] + j]);
				}
			}
		// }
		// for (i = 0;i<n_vtx;i++) {
			for (j=0; j<n_ints2; j++){
				localcnt[i] += __builtin_popcount(blockR2[rank][i*n_ints2 + j]);
			}
		}
		// }
		for (i = 0;i<n_vtx;i++) {
			if (localcnt[i] > local_max){
				local_max = localcnt[i];
				local_vtx = i;
			}
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
		#pragma omp barrier
		#pragma omp critical
	    {	
			std::set<size_t> local_idx_set;
			size_t tmp_key = 0;
			for(int r=0;r<num_threads;r++){
				tmp_key = local_v[r];
				if(local_idx_set.find(tmp_key)==local_idx_set.end()){
					// std::cout<<tmp_key<<","<<localcnt[tmp_key]<<", ";
					local_idx_set.insert(tmp_key);
					globalcnt[tmp_key]+=localcnt[tmp_key];
				}
			}
			// std::cout<<std::endl;
	    }  
	    free(localcnt);		
	}
    maxvtx=0;
    maxcnt=0;
    for(int i=0;i<n_vtx;i++){
    	if(globalcnt[i]>0){
    		std::cout<<"coutRR02: global[i]="<<i<<", "<<globalcnt[i]<<std::endl;
    	}
        if(globalcnt[i]>maxcnt)  {
          maxcnt=globalcnt[i];
          maxvtx=i;
        }
    }
    std::cout<<"coutRR02: maxk="<<maxvtx<<" maxv="<<maxcnt<<std::endl;
    free(globalcnt);
}

void selectRR20(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, std::vector<size_t> n_ints,
			 size_t *local_m, size_t *local_v, std::vector<bool *> &deleteflag, size_t &maxk, size_t &maxv){
	size_t num_threads = omp_get_max_threads();
	size_t n_xs = n_ints.size();
	
	size_t* globalcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	std::cout<<" selectRR20 mvtx="<<maxk<<" mmax="<<maxv << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close)  shared(maxk)
  	{
  		size_t i = 0, j = 0, x = 0; // k = 0, 
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int tmp1 = 0, tmp2 = 0, tmp3 = 0;
	    size_t local_max = 0, local_vtx = 0;
		deleteflag[rank][maxk] = 1;    
// #pragma omp for schedule(static)  		    
// 		for(k=0; k<num_threads; k++) 
// 		{
		for(x = 0; x < n_xs; x++){
			for (j=0; j<n_ints[x]; j++){
				blockR[x][rank][n_vtx*n_ints[x] + j] = blockR[x][rank][n_vtx*n_ints[x] + j] | blockR[x][rank][maxk*n_ints[x] + j];
			}
		}
		for (i = 0;i<n_vtx;i++){
			if(deleteflag[rank][i]== 0){	
				for(x = 0; x < n_xs; x++){
					for (j=0; j<n_ints[x]; j++){
						tmp1 = blockR[x][rank][i*n_ints[x] + j] ^ blockR[x][rank][maxk*n_ints[x] + j];
						blockR[x][rank][i*n_ints[x] + j] &= tmp1;
						localcnt[i] += __builtin_popcount(blockR[x][rank][i*n_ints[x] + j]);
					}
				}
			}
			if (localcnt[i] > local_max){
				local_max = localcnt[i];
				local_vtx = i;
			}
			if(localcnt[i]==0){
				deleteflag[rank][i]=1;
			}
		}
		// }
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
		#pragma omp barrier
		#pragma omp critical
	    {	
			std::set<size_t> local_idx_set;
			size_t tmp_key = 0;
			for(int r=0;r<num_threads;r++){
				tmp_key = local_v[r];
				if(local_idx_set.find(tmp_key)==local_idx_set.end()){
					// std::cout<<tmp_key<<","<<localcnt[tmp_key]<<", ";
					local_idx_set.insert(tmp_key);
					globalcnt[tmp_key]+=localcnt[tmp_key];
				}
			}
			// std::cout<<std::endl;
	    }
	    free(localcnt);		
	}
	maxk=0;
    maxv=0;
    for(int i=0;i<n_vtx;i++){
        if(globalcnt[i]>maxv)  {
          maxk=i;
          maxv=globalcnt[i];
        }
    }
    free(globalcnt);
}

void selectRR202(std::vector<std::vector<unsigned int*>> &blockR1, std::vector<unsigned int*> &blockR2, 
			 const size_t n_vtx, std::vector<size_t> n_ints1, const size_t n_ints2,
			 size_t *local_m, size_t *local_v, std::vector<bool *> &deleteflag, size_t &maxk, size_t &maxv){
	size_t num_threads = omp_get_max_threads();
	size_t n_xs = n_ints1.size();
	
	size_t* globalcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	std::cout<<" selectRR202 mvtx="<<maxk<<" mmax="<<maxv << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) shared(maxk)
  	{
  		size_t i = 0, j = 0, x = 0; //k = 0
  		size_t rank = omp_get_thread_num();
	    size_t* localcnt=(size_t*)calloc(n_vtx,sizeof(size_t));
	    unsigned int tmp1 = 0, tmp2 = 0;
	    size_t local_max = 0, local_vtx = 0;
	    deleteflag[rank][maxk] = 1;
		for (i = 0;i<n_vtx;i++) 
		{
			if(deleteflag[rank][i]== 0){
				for(x = 0; x < n_xs; x++){
					for (j=0; j<n_ints1[x]; j++){
						tmp1 = blockR1[x][rank][i*n_ints1[x] + j] ^ blockR1[x][rank][maxk*n_ints1[x] + j];
						blockR1[x][rank][i*n_ints1[x] + j] &= tmp1;
						localcnt[i] += __builtin_popcount(blockR1[x][rank][i*n_ints1[x] + j]);
					}
				}
				for(j=0; j<n_ints2; j++){
					tmp1 = blockR2[rank][i*n_ints2 + j] ^ blockR2[rank][maxk*n_ints2 + j];
					blockR2[rank][i*n_ints2 + j] &= tmp1;
					localcnt[i] += __builtin_popcount(blockR2[rank][i*n_ints2 + j]);
				}
				if (localcnt[i] > local_max){
					local_max = localcnt[i];
					local_vtx = i;
				}
				if(localcnt[i]==0){
					deleteflag[rank][i]=1;
				}
			}	
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
		#pragma omp barrier
		#pragma omp critical
	    {	
			std::set<size_t> local_idx_set;
			size_t tmp_key = 0;
			for(int r=0;r<num_threads;r++){
				tmp_key = local_v[r];
				if(local_idx_set.find(tmp_key)==local_idx_set.end()){
					// std::cout<<tmp_key<<","<<localcnt[tmp_key]<<", ";
					local_idx_set.insert(tmp_key);
					globalcnt[tmp_key]+=localcnt[tmp_key];
				}
			}
			// std::cout<<std::endl;
	    }
	    free(localcnt);		
	}
	maxk=0;
    maxv=0;
    for(int i=0;i<n_vtx;i++){
        if(globalcnt[i]>maxv)  {
          maxk=i;
          maxv=globalcnt[i];
        }
    }
    free(globalcnt);
}

template <typename vertex_type, typename RRRset>
void bitmapRRRSets0(std::vector<RRRset> &RRRsets, 
				   const int blockoffset,
				   std::vector<size_t> &blockR_pointer, 
				   std::vector<unsigned int*> &blockR,
				   const size_t n_ints) {
  	size_t s1 = RRRsets.size(); //i: current RR's index
  	// std::cout<<" s1="<<s1<<" block-offset="<<blockoffset << std::endl;
  	size_t total_rrr_size = 0;
  	size_t num_threads = omp_get_max_threads();
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t rank = omp_get_thread_num();
#pragma omp for reduction(+:total_rrr_size) schedule(static)//(dynamic)//
	    for (size_t i=blockoffset; i<s1; i++) {
	        auto in_begin=RRRsets.begin();
	        std::advance(in_begin, i);
	        size_t s2=std::distance(in_begin->begin(),in_begin->end());
	        encodeRR0(in_begin, blockR_pointer[rank], s2, n_ints, blockR[rank]);
	        blockR_pointer[rank] += 1;
	        (*in_begin).clear();	//# check why block+compress is faster than original sampling
	        (*in_begin).shrink_to_fit(); //# check why block+compress is faster than original sampling
	        total_rrr_size += s2;
	    }
	}
}

} //// namespace ripples
#endif //RIPPLES_BITMAP_H
