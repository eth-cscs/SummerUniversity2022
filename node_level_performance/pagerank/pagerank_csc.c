#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
//#include <mpi.h>
#include "mmio.h"



/**
 * @brief      Debuging print matrix
 *
 * @param      V        values
 * @param      col_ptr  The col pointer
 * @param      row_ind  The row ind
 * @param[in]  n        size of matrix
 */
void print_matrix(double* V, int* col_ptr, int* row_ind, int n) {

	double A[n][n];
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i][j] = 0.0;
		}
	}

	for (int j = 0; j < n; ++j) {
		for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
			A[row_ind[k]][j] = V[k];
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
}

/* This is a serial sparse implementation of PageRank algorithm.
  We are using the Compressed Sparse Column Format (CSC) storage technique
  for the arrays indices, indptr, data:
  - All non zero elements will be stored as G vector
  - indices is array of row indices
  - G is array of corresponding nonzero values
  - indptr points to column starts in indices and data
  - length is n_col + 1, last item = number of values = length of both indices and data */

int main(int argc, char *argv[]) {


	// Aryan Eftkehari
	FILE *fp;
	char *filename;
	MM_typecode matcode;
	int ret_code;
	int colindex, link, i, j = 0, col, colmatch = 0, localc = 0;
	int k, it = 0;

	int nodes;  // number of webpages = number of nodes in the graph
	int nedges; // number of links in the webgraph = number of nonzeros in the matrix
	double p = 0.85; // probability that a user follows an outgoing link

	// use G as the scaled nonzero adjacency matrix stored in compress column format
	double *G = NULL;
	int    *indices = NULL, *colptr = NULL;

	double *xs = NULL; // will be used to save the current solution vector x
	double * x = NULL; // new solution vector x
	int    * c = NULL; // number of nonzeros per column in G (out-degree)
	double * z = NULL; //

	double err = 1.e-9;
	double norm = 0.0, norm_sq = 0.0, time_start = 0.0;
	char ch;

	// Check if a filename has been specified in the command
	if (argc < 2) {
		printf("Missing Filename\n");
		return (1);
	} else {
		filename    = argv[1];
	}



	fp = fopen(filename, "r");

	if (mm_read_banner(fp, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
	        mm_is_sparse(matcode) ) {
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	/* find out size of sparse matrix .... */
	if ((ret_code = mm_read_mtx_crd_size(fp, &nodes, &nodes, &nedges)) != 0)
		exit(1);


	// assumption is that the matrix is stored in one-based index
	printf("[HPC for CSE] Number of nodes %d and edges %d in webgraph %s \n", nodes, nedges, filename) ;


	G       = (double*) calloc(nedges, sizeof(double));
	indices = (int*)    calloc(nedges, sizeof(int));
	colptr  = (int*)    calloc(nodes + 1, sizeof(int));
	xs      = (double*) malloc(nodes * sizeof(double)); // will be used to save the current solution vector x
	x       = (double*) calloc(nodes, sizeof(double));  // new solution vector x
	z       = (double*) malloc(nodes * sizeof(double)); //
	c       = (int*)    calloc(nodes, sizeof(int));     // number of nonzeros per column in G (out-degree)

	int ii;
	for (i = 0; i < nedges; i++) {

		if (fscanf(fp, "%d %d", &link, &colindex));

		link -= 1; // matrix transfered to zero index
		colindex -= 1; // matrix transfered to zero index

		// @Aryan Eftekhari:
		if (  (i >=  0) && (nedges) > i  ) {

			ii = i;

			indices[ii] = link;

			if (colmatch == colindex) {
				localc += 1;
				G[ii] = 1.0;
			} else {
				while ( (colmatch + 1) != colindex) {
					c[j] = localc;
					colptr[j + 1] = colptr[j] + localc; //index of G where new column starts
					j += 1;
					localc = 0; // no new element
					c[j] = localc;
					colptr[j + 1] = colptr[j]; //index of G where new column starts
					colmatch += 1; //new column
					localc = 0; //new localc
				}
				c[j] = localc;
				colptr[j + 1] = colptr[j] + localc; //index of G where new column starts
				localc = 1; //new localc
				j += 1;
				colmatch = colindex; //new column
				G[ii] = 1.0;
			}
		}
	}

	c[j] = localc;       // entry for the last column in compressed column format
	colptr[j + 1] = nedges; // toal number of nonzeros in the matrix G
	fclose(fp);


	// Aryan Eftekhari
	// Sum overall process the column sums. Note with this we do not need
	// to change any of the code for construction pGD or z
	// The scaling factor is used as we will sum over all processes.
	double scale = 1.0;

	// build scaled nonzero adjacency matrix G = p*G*D
	for (i = 0; i < nodes; i++) {
		for (j = colptr[i]; j < colptr[i + 1] ; j++) {
			G[j] = p * G[j] / c[i];
		}
	}

	for (i = 0; i < nodes; i++) { // This is to normalize the array of non zeros
		if (c[i] != 0 )
			z[i] = (1 - p) / (double)nodes * scale;
		else
			z[i] = 1.0 / (double)nodes * scale;
	}
	for (i = 0; i < nodes; i++)
		xs[i] = 1.0 / ( (double ) nodes ); // inital guess for x (similar as in the Matlab code

	// ************************************************
	// ********* Computing PageRank *******************
	// ************************************************
	time_start = omp_get_wtime();


	// @Aryan Eftekhari
	// For debugging
	if (0) {
		// Debugging
		printf("//////////////////// \n");
		for (int i = 0; i < nedges; ++i) {
			printf("G[%i]=%f\n", i, G[i] );
		}

		for (int i = 0; i < nedges; ++i) {
			printf("indices[%i]=%i\n", i, indices[i] );
		}


		for (int i = 0; i < nodes + 1; ++i) {
			printf("colptr[%i]=%i\n", i, colptr[i] );
		}

		for (int i = 0; i < nodes; ++i) {
			printf("c[%i]=%i\n", i, c[i] );
		}

		print_matrix(G, colptr, indices, nodes);
	}

	do {
		// compute x = p*G*D*x
		memset(x, 0, nodes * sizeof(double)); // set new solution x to 0
		for (col = 0; col < nodes; col++) {
			for (j = colptr[col]; j < colptr[col + 1]; j++) {

				x[indices[j]] += G[j] * xs[col];
			}
		}

		// x = p*G*D*x + e*(z*x)
		norm = 0.0;
		for (i = 0; i < nodes; i++) {
			norm += z[i] * xs[i];
		}

		// x = p*G*D*x + e*(z*x)
		for (i = 0; i < nodes; i++) {
			x[i] = x[i] + norm;
		}

		norm_sq = 0.0;
		for (i = 0; i < nodes; i++) {
			norm_sq += (x[i] - xs[i]) * (x[i] - xs[i]);
			xs[i] = x[i];
		}
		norm = sqrt(norm_sq);
		it += 1;
	} while (norm > err );
	printf("[HPC for CSE] %d PageRank iterations with norm of %e computed in %f sec. \n", it, norm, omp_get_wtime() - time_start);

	// @Aryan Eftekhari
	// Bring Solution to all nodes & Save
	double temp = 0;
	for (int i = 0; i < nodes; ++i) {
		temp += x[i] * x[i];
	}
	printf("//******************* \n");
	printf("// Sol. Norm Squared : %e \n", temp);
	printf("//******************* \n");

	fp = fopen("PageRank.dat", "w");
	for (j = 0; j < nodes; j++) {
		fprintf(fp, "%e\n", x[j]);
	}
	fclose(fp);

	free(G);
	free(indices);
	free(colptr);
	free(xs);
	free(x);
	free(z);
	free(c);


	return 0;
}
