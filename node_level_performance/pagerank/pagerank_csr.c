#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
//#include <mpi.h>
#include "mmio.h"


void spmv(int nodes, int* rowptr, int* indices, double* xs, double* G, double* x);
double dotprod(int nodes, double* z, double* xs);
double devsq(int nodes, double* x, double* xs);
void nadd(int nodes, double* x, double norm);
  void convert_matrix (int , int , double *, int *, int *, int *);

/* This is a serial sparse implementation of PageRank algorithm.
  We are using the Compressed Sparse Row Format (CSR) storage technique
  for the arrays indices, indptr, data:
  - All non zero elements will be stored as G vector
  - indices is array of column indices
  - G is array of corresponding nonzero values
  - rowptr points to row starts in indices and G
  - length is nodes + 1, last item = number of values = length of both indices and G */

int main(int argc, char *argv[]) {


	// Aryan Eftkehari
	// MPI stuff
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
	double *G  = NULL;
	double *G1 = NULL;
	int    *indices = NULL, *colptr = NULL;
	int    *rowptr = NULL;

	double *xs = NULL; // will be used to save the current solution vector x
	double * x = NULL; // new solution vector x
	int    * c = NULL; // number of nonzeros per column in G (out-degree)
	double * z = NULL; //

	double err = 1.e-9;
	double norm = 0.0, norm_sq = 0.0, time_start = 0.0;
	char ch;
	char line[21];

	// Check if a filename has been specified in the command
	if (argc < 2) {
		printf("Missing Filename\n");
		return (1);
	} else {
		filename = argv[1];
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
	rowptr  = (int*)    calloc(nodes + 1, sizeof(int)); // used to store matrix in compressed sparse row format, row pointer
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

	//
	// transform matrix to compressed sparse row format for pagerank computation
	//
	convert_matrix (nodes, nedges, G, indices, colptr, rowptr);


	// Aryan Eftekhari
	// Sum overall process the column sums. Note with this we do not need
	// to change any of the code for construction pGD or z
	// The scaling factor is used as we will sum over all processes.
	double scale = 1.0;

	// build scaled nonzero adjacency matrix G = p*G*D
	for (i = 0; i < nodes; i++) {
		for (j = rowptr[i]; j < rowptr[i + 1] ; j++) {
			G[j] = p * G[j] / c[indices[j]];
		}
	}

	for (i = 0; i < nodes; i++) { // This is to normalize the array of non zeros
		if (c[i] != 0 )
			z[i] = (1 - p) / (double)nodes * scale;
		else
			z[i] = 1.0 / (double)nodes * scale;
	}
	for (i = 0; i < nodes; i++)
		xs[i] =  1.0 / ( (double ) nodes ); // inital guess for x (similar as in the Matlab code

	// ************************************************
	// ********* Computing PageRank *******************
	// ************************************************
	time_start = omp_get_wtime();
	do {
		// compute x = p*G*D*x
		memset(x, 0, nodes * sizeof(double)); // set new solution x to 0
		spmv(nodes, rowptr, indices, xs, G, x);

		// x = p*G*D*x + e*(z*x)
		norm = dotprod(nodes, z, xs);

		// x = p*G*D*x + e*(z*x)
		nadd(nodes, x, norm);
		/*
		for (i = 0; i < nodes; i++) {
			x[i] = x[i] + norm;
		}
		*/
		norm_sq = devsq(nodes, x, xs);

		norm = sqrt(norm_sq);

		it += 1;
	} while (norm > err);
	printf("[HPC for CSE] %d PageRank iterations with norm of %e computed in %.2lf sec. \n",
	       it, norm, omp_get_wtime() - time_start);

	// ************************************************
	// ********* Save PageRank vector x ***************
	// ************************************************

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
	free(rowptr);
	free(xs);
	free(x);
	free(z);
	free(c);

	return 0;
}


void convert_matrix (int nodes, int nedges, double *G, int *indices, int *colptr, int *rowptr) {
	int *ja     = (int*)   calloc(nedges, sizeof(int));  // used to store matrix in compressed sparse row format,  column elements
	int *sumrow = (int*)   calloc(nodes + 1, sizeof(int)); // used to store matrix in compressed sparse row format,  column elements
	double *G1  = (double*)calloc(nedges, sizeof(double));

	for (int i = 0; i < nodes; i++) {
		for (int j = colptr[i]; j < colptr[i + 1] ; j++) {
			sumrow[indices[j]] += 1;
		}
	}
	rowptr[0] = 0;
	for (int i = 1; i < nodes + 1; i++) {
		rowptr[i] = rowptr[i - 1] + sumrow[i - 1];
	}
	for (int i = 0; i < nodes; i++) {
		sumrow[i] = rowptr[i];
	}
	for (int i = 0; i < nodes; i++) {
		for (int j = colptr[i]; j < colptr[i + 1] ; j++) {
			ja[sumrow[indices[j]]] = i;
			G1[sumrow[indices[j]]] = G[j];
			sumrow[indices[j]] += 1;
		}
	}
	for (int i = 0; i < nodes; i++) {
		for (int j = rowptr[i]; j < rowptr[i + 1] ; j++) {
			G[j] = G1[j];
			indices[j] = ja[j];
		}
	}
	free(ja);
	free(sumrow);
	free(G1);

}

void spmv(int nodes, int* rowptr, int* indices, double* xs, double* G, double* x) {
  for (int row = 0; row < nodes; row++) {
    for (int j = rowptr[row]; j < rowptr[row + 1]; j++) {
      x[row] += G[j] * xs[indices[j]];
    }
  }
}

double dotprod(int nodes, double* z, double* xs) {
  double sp = 0.0;
  for (int i = 0; i < nodes; i++) {
    sp += z[i] * xs[i];
  }
  return sp;
}

double devsq(int nodes, double* x, double* xs) {
  double ds = 0.0;
  for (int i = 0; i < nodes; i++) {
    ds += (x[i] - xs[i]) * (x[i] - xs[i]);
    xs[i] = x[i];
  }
  return ds;
}

void nadd(int nodes, double* x, double norm) {
  for (int i = 0; i < nodes; i++) {
    x[i] = x[i] + norm;
  }
}

