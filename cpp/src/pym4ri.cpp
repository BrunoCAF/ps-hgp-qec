#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <m4ri/m4ri.h>
#include <numpy/ndarrayobject.h>

#include <cstdlib>

#undef NDEBUG
#include <assert.h>


std::pair<std::pair<mzd_t *, mzd_t *>, std::pair<mzp_t *, mzp_t *>>
clean_pluq(mzd_t *A) {
	auto [m, n] = std::make_pair(A->nrows, A->ncols);
	mzp_t *P = mzp_init(m), *Q = mzp_init(n);
	rci_t r = mzd_pluq(A, P, Q, 0);

	mzd_t *L_sq = mzd_init(r, r), *L_low;
	mzd_t *U_sq = mzd_init(r, r), *U_right;
	L_sq = mzd_extract_l(L_sq, mzd_submatrix(L_sq, A, 0, 0, r, r));
	L_low = mzd_submatrix(NULL, A, r, 0, m, r);
	U_sq = mzd_extract_u(U_sq, mzd_submatrix(U_sq, A, 0, 0, r, r));
	U_right = mzd_submatrix(NULL, A, 0, r, r, n);

	mzd_t *L = mzd_stack(NULL, L_sq, L_low);
	mzd_t *U = mzd_concat(NULL, U_sq, U_right);

	mzd_free(L_sq), mzd_free(U_sq), mzd_free(L_low), mzd_free(U_right);

	auto LU = std::make_pair(L, U);
	auto PQ = std::make_pair(P, Q);
	return std::make_pair(LU, PQ);
}

mzd_t *_gen2chk(mzd_t *G) {
	// G is assumed to be of the form [I | A]
	auto [m, n] = std::make_pair(G->nrows, G->ncols);
	rci_t r = m;

	mzd_t *A = mzd_submatrix(NULL, G, 0, r, m, n);
	mzd_t *At = mzd_transpose(NULL, A);
	mzd_t *I = mzd_init(n - r, n - r);
	for (int i = 0; i < n - r; i++)
		mzd_write_bit(I, i, i, 1);
	mzd_t *H = mzd_concat(NULL, At, I);

	mzd_free(A), mzd_free(At), mzd_free(I);

	// Returns H = [A^t | I]
	return H;
}

mzd_t *gen2chk(mzd_t *G) {
	auto [L_U, P_Q] = clean_pluq(G);
	auto [L, U] = L_U;
	auto [P, Q] = P_Q;
	mzd_free(L), mzp_free(P);

	mzd_echelonize_pluq(U, 1);
	mzd_t *H = _gen2chk(U);
	mzd_apply_p_right_trans(H, Q);
	mzd_free(U), mzp_free(Q);

	return H;
}

mzd_t *chk2gen(mzd_t *H) { return mzd_kernel_left_pluq(H, 0); }

rci_t rank(mzd_t *M) { return mzd_echelonize_m4ri(M, 0, 0); }

mzd_t *PyArray_ToMzd(PyObject *array_obj) {
	PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(
		array_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY); 
	if (!array)
		return NULL;

	npy_intp *shape = PyArray_SHAPE(array);
	mzd_t *matrix = mzd_init(shape[0], shape[1]);

	for (npy_intp i = 0; i < shape[0]; i++) {
		for (npy_intp j = 0; j < shape[1]; j++) {
			uint8_t value = *(uint8_t *)PyArray_GETPTR2(array, i, j);
			mzd_write_bit(matrix, i, j, value);
		}
	}
	Py_DECREF(array);
	return matrix;
}

PyObject *MzdToPyArray(mzd_t *matrix) {
	npy_intp dims[2] = {matrix->nrows, matrix->ncols};
	PyObject *array_obj = PyArray_SimpleNew(2, dims, NPY_UINT8);
	PyArrayObject *array = (PyArrayObject *)array_obj;

	for (int i = 0; i < matrix->nrows; i++) {
		for (int j = 0; j < matrix->ncols; j++) {
			uint8_t value = mzd_read_bit(matrix, i, j);
			*(uint8_t *)PyArray_GETPTR2(array, i, j) = value;
		}
	}
	return array_obj;
}

// extern "C" 
static PyObject *gen2chk(PyObject *self, PyObject *args) {
	(void)self;
	PyObject *G_obj;
	if (!PyArg_ParseTuple(args, "O", &G_obj))
		return NULL;

	mzd_t *G = PyArray_ToMzd(G_obj);
	mzd_t *H = gen2chk(G);

	PyObject *H_obj = MzdToPyArray(H);
	mzd_free(G), mzd_free(H);

	return H_obj;
}

// extern "C" 
static PyObject *chk2gen(PyObject *self, PyObject *args) {
	(void)self;
	PyObject *H_obj;
	if (!PyArg_ParseTuple(args, "O", &H_obj))
		return NULL;

	mzd_t *H = PyArray_ToMzd(H_obj);
	mzd_t *G = chk2gen(H);

	PyObject *G_obj = MzdToPyArray(G);
	mzd_free(H), mzd_free(G);

	return G_obj;
}

// extern "C" 
static PyObject *rank(PyObject *self, PyObject *args) {
	(void)self;
	PyObject *M_obj;
	if (!PyArg_ParseTuple(args, "O", &M_obj))
		return NULL;
	mzd_t *M = PyArray_ToMzd(M_obj);
	rci_t r = rank(M);
	mzd_free(M);
	return Py_BuildValue("i", r);
}

int main() {
	const rci_t m = 3, n = 5;

	rci_t example[m][n] = {
		{0, 0, 1, 1, 1},
		{1, 1, 0, 0, 0},
		{0, 0, 0, 0, 0},
	};
	mzd_t *H = mzd_init(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mzd_write_bit(H, i, j, example[i][j]);

	printf("Behold, H:\n");
	mzd_print(H);
	printf("----------\n");

	// mzd_t *H = gen2chk(G);
	mzd_t *HH = mzd_copy(NULL, H), *G = chk2gen(HH);
    mzd_free(HH);

	printf("Behold, G:\n");
	mzd_print(G);
	printf("----------\n");

	mzd_t *HG = mzd_mul(NULL, H, G, 0);

	printf("Behold, HG:\n");
	mzd_print(HG);
	printf("----------\n");

	assert(mzd_is_zero(HG));
	mzd_free(G), mzd_free(H), mzd_free(HG);

	return 0;
}

static PyMethodDef cssutils[] = {
	{"gen2chk", gen2chk, METH_VARARGS,
	 "Find a parity-check matrix from a generator matrix."},
	{"chk2gen", chk2gen, METH_VARARGS,
	 "Find a generator matrix from a parity-check matrix."},
	{"rank", rank, METH_VARARGS, "Computes the rank of a matrix in GF(2)."},
	{NULL, NULL, 0, NULL},
};

static struct PyModuleDef m4rimodule = {
	PyModuleDef_HEAD_INIT,
	"pym4ri",
	NULL, // module_doc,
	-1,	  // size of the state of the module
	cssutils,
    NULL, // m_slots
    NULL, // m_traverse
    NULL, // m_clear
    NULL  // m_free
};

PyMODINIT_FUNC PyInit_pym4ri(void) {
	import_array();
	return PyModule_Create(&m4rimodule);
}