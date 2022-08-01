/**
 *  @file io.h
 *  @author Sheng Di
 *  @date April, 2015
 *  @brief Header file for the whole io interface.
 *  (C) 2015 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _IO_H
#define _IO_H

#include <stdio.h>
#include <stdint.h>
#include "defines.h"
#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef union lint32
{
	int ivalue;
	unsigned int uivalue;
	unsigned char byte[4];
} lint32;

extern int dataEndianType; //*endian type of the data read from disk
extern int sysEndianType; //*sysEndianType is actually set automatically.

void symTransform_2bytes(unsigned char data[2]);
void symTransform_4bytes(unsigned char data[4]);

int bytesToInt_bigEndian(unsigned char* bytes);
void intToBytes_bigEndian(unsigned char *b, unsigned int num);
void longToBytes_bigEndian(unsigned char *b, unsigned long num);

unsigned char *readByteData(char *srcFilePath, size_t *byteLength, int *status);
uint32_t *readUInt32Data(char *srcFilePath, size_t *nbEle, int *status);

uint32_t *readUInt32Data_systemEndian(char *srcFilePath, size_t *nbEle, int *status);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _IO_H  ----- */
