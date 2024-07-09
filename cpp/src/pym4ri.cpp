#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <m4ri/m4ri.h>
#include <numpy/ndarrayobject.h>
#include <cstdlib>
#include <string>
#include <assert.h>

inline void safe_mzd_print(const mzd_t *M) {
    if (M && M->nrows && M->ncols)
        mzd_print(M);
    else
        printf("\n");
}

void behold(mzd_t *M, std::string C) {
    const char *c = C.c_str();
    printf("Is %s NULL? <<%d>>\n", c, M == NULL);
    printf("Behold %s (%d x %d): \n", c, M->nrows, M->ncols);
    safe_mzd_print(M);
    printf("--------------------\n");
}

mzd_t *safe_mzd_copy(mzd_t *DST, const mzd_t *A) {
    return (A->nrows && A->ncols) ? mzd_copy(DST, A) : mzd_init(A->nrows, A->ncols);
}

mzd_t *safe_mzd_transpose(mzd_t *DST, const mzd_t *A) {
    return (A->nrows && A->ncols) ? mzd_transpose(DST, A) : mzd_init(A->ncols, A->nrows);
}

mzd_t *safe_mzd_submatrix(mzd_t *S, const mzd_t *M, rci_t lowr, rci_t lowc, rci_t highr, rci_t highc) {
    if ((highr - lowr) && (highc - lowc))
        return mzd_submatrix(S, M, lowr, lowc, highr, highc);
    return mzd_init(highr - lowr, highc - lowc);
}

std::pair<std::pair<mzd_t *, mzd_t *>, std::pair<mzp_t *, mzp_t *>> clean_pluq(mzd_t *A) {
    auto [m, n] = std::make_pair(A->nrows, A->ncols);
    mzp_t *P = mzp_init(m), *Q = mzp_init(n);
    rci_t r = mzd_pluq(A, P, Q, 0);

    mzd_t *L_sq = mzd_init(r, r), *L_low;
    mzd_t *U_sq = mzd_init(r, r), *U_right;
    L_sq = mzd_extract_l(L_sq, safe_mzd_submatrix(L_sq, A, 0, 0, r, r));
    U_sq = mzd_extract_u(U_sq, safe_mzd_submatrix(U_sq, A, 0, 0, r, r));

    L_low = safe_mzd_submatrix(NULL, A, r, 0, m, r);
    U_right = safe_mzd_submatrix(NULL, A, 0, r, r, n);

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

    mzd_t *A = safe_mzd_submatrix(NULL, G, 0, r, m, n);
    mzd_t *At = safe_mzd_transpose(NULL, A);
    mzd_t *I = mzd_init(n - r, n - r);
    for (int i = 0; i < n - r; i++)
        mzd_write_bit(I, i, i, 1);
    mzd_t *H = mzd_concat(NULL, At, I);

    mzd_free(A), mzd_free(At), mzd_free(I);

	// Returns H = [A^t | I]
    return H;
}

mzd_t *gen2chk(mzd_t *G) {
	// Safety check against empty matrices
    if (G->nrows && G->ncols) {
        auto [L_U, P_Q] = clean_pluq(G);
        auto [L, U] = L_U;
        auto [P, Q] = P_Q;
        mzd_free(L), mzp_free(P);

        mzd_echelonize_pluq(U, 1);

        mzd_t *H = _gen2chk(U);
        mzd_apply_p_right(H, Q);
        mzd_free(U), mzp_free(Q);

        return H;
    }

    mzd_t *I = mzd_init(G->ncols, G->ncols);
    for (int i = 0; i < G->ncols; i++)
        mzd_write_bit(I, i, i, 1);
    return I;
}

mzd_t *chk2gen(mzd_t *H) {
    if (H->nrows && H->ncols) {
        mzd_t *X = mzd_kernel_left_pluq(H, 0);
        return X ? X : mzd_init(H->ncols, 0);
    }

    mzd_t *I = mzd_init(H->ncols, H->ncols);
    for (int i = 0; i < H->ncols; i++)
        mzd_write_bit(I, i, i, 1);
    return I;
}

rci_t rank(mzd_t *M) { return mzd_echelonize_m4ri(M, 0, 0); }

mzd_t *PyArray_ToMzd(PyObject *array_obj) {
    if (!PyArray_Check(array_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array");
        return NULL;
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (!array)
        return NULL;

    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected a 2D numpy array");
        Py_DECREF(array);
        return NULL;
    }

    npy_intp *shape = PyArray_SHAPE(array);
    mzd_t *matrix = mzd_init(shape[0], shape[1]);

    for (npy_intp i = 0; i < shape[0]; i++) {
        for (npy_intp j = 0; j < shape[1]; j++) {
            bool value = *(bool *)PyArray_GETPTR2(array, i, j);
            mzd_write_bit(matrix, i, j, value);
        }
    }
    Py_DECREF(array);
    return matrix;
}

PyObject *MzdToPyArray(mzd_t *matrix) {
    npy_intp dims[2] = {matrix->nrows, matrix->ncols};
    PyObject *array_obj = PyArray_SimpleNew(2, dims, NPY_BOOL);
    PyArrayObject *array = (PyArrayObject *)array_obj;

    for (int i = 0; i < matrix->nrows; i++) {
        for (int j = 0; j < matrix->ncols; j++) {
            bool value = mzd_read_bit(matrix, i, j);
            *(bool *)PyArray_GETPTR2(array, i, j) = value;
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
    if (!G) return NULL;

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
    if (!H) return NULL;

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
    if (!M) return NULL;

    rci_t r = rank(M);
    mzd_free(M);
    return Py_BuildValue("i", r);
}

// extern "C"
static PyObject *gf2_mul(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *A_obj, *B_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &B_obj))
        return NULL;

    mzd_t *A = PyArray_ToMzd(A_obj);
    if (!A) return NULL;

    mzd_t *B = PyArray_ToMzd(B_obj);
    if (!B) {
        mzd_free(A);
        return NULL;
    }

    if (A->ncols != B->nrows) {
        PyErr_SetString(PyExc_ValueError, "Incompatible dimensions for matrix multiplication");
        mzd_free(A);
        mzd_free(B);
        return NULL;
    }

    mzd_t *C = mzd_mul(NULL, A, B, 0);
    PyObject *C_obj = MzdToPyArray(C);
    mzd_free(A), mzd_free(B), mzd_free(C);

    return C_obj;
}

void test_chk2gen(mzd_t *H) {
	printf(":: CHK2GEN TEST ::\n");
	// behold H
	behold(H, "H");

	// apply chk2gen
	mzd_t *HH = safe_mzd_copy(NULL, H);
	mzd_t *G = chk2gen(HH);
	mzd_free(HH);

	// behold G
	behold(G, "G");

	// verify condition
	mzd_t *X = mzd_mul(NULL, H, G, 0);

	behold(X, "H*G");

	if (X->nrows && X->ncols)
		assert(mzd_is_zero(X));

	// clear
	mzd_free(X);
	mzd_free(G);

	printf(":: CHK2GEN DONE ::\n");
}

void test_gen2chk(mzd_t *H) {
	printf(":: GEN2CHK TEST ::\n");
	// behold H
	behold(H, "H");

	// apply gen2chk
	mzd_t *HH = safe_mzd_copy(NULL, H);
	mzd_t *G = gen2chk(HH);
	mzd_free(HH);

	// behold G
	behold(G, "G");

	// verify condition
	mzd_t *Ht = safe_mzd_transpose(NULL, H), *X = mzd_mul(NULL, G, Ht, 0);
	mzd_free(Ht);

	behold(X, "G*H^t");

	if (X->nrows && X->ncols)
		assert(mzd_is_zero(X));

	// clear
	mzd_free(X);
	mzd_free(G);

	printf(":: GEN2CHK DONE ::\n");
}

void test_rank(mzd_t *H) {
	behold(H, "H");
	printf("rank(H) = %d\n", rank(H));
}

int main() {
	const rci_t m = 3, n = 5;

	rci_t example[m][n] = {
		{0, 1, 1, 0, 1},
		{0, 0, 0, 0, 0},
		{0, 1, 0, 0, 1},
	};
	mzd_t *H = mzd_init(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mzd_write_bit(H, i, j, example[i][j]);

	test_chk2gen(H);
	test_gen2chk(H);
	test_rank(H);

	mzd_free(H);

	return 0;
}

PyDoc_STRVAR(gen2chk_doc, 
    "gen2chk(G: numpy.ndarray[bool]) -> numpy.ndarray[bool]\n"
    "\n"
    "Find a parity-check matrix from a generator matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "G : numpy.ndarray of bool\n"
    "    The generator matrix.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray of bool\n"
    "    The parity-check matrix.\n"
);

PyDoc_STRVAR(chk2gen_doc, 
    "chk2gen(H: numpy.ndarray[bool]) -> numpy.ndarray[bool]\n"
    "\n"
    "Find a generator matrix from a parity-check matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "H : numpy.ndarray of bool\n"
    "    The parity-check matrix.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray of bool\n"
    "    The generator matrix.\n"
);

PyDoc_STRVAR(rank_doc, 
    "rank(M: numpy.ndarray[bool]) -> int\n"
    "\n"
    "Computes the rank of a matrix in GF(2).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "M : numpy.ndarray of bool\n"
    "    The matrix.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "int\n"
    "    The rank of the matrix.\n"
);

PyDoc_STRVAR(gf2_mul_doc, 
    "gf2_mul(A: numpy.ndarray[bool], B: numpy.ndarray[bool]) -> numpy.ndarray[bool]\n"
    "\n"
    "GF(2) matrix multiplication.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "A : numpy.ndarray of bool\n"
    "    The first matrix.\n"
    "B : numpy.ndarray of bool\n"
    "    The second matrix.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray of bool\n"
    "    The result of multiplying A by B in GF(2).\n"
);

static PyMethodDef cssutils[] = {
    {"gen2chk", gen2chk, METH_VARARGS, gen2chk_doc},
    {"chk2gen", chk2gen, METH_VARARGS, chk2gen_doc},
	{"rank", rank, METH_VARARGS, rank_doc},
	{"gf2_mul", gf2_mul, METH_VARARGS, gf2_mul_doc},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef m4rimodule = {
    PyModuleDef_HEAD_INIT,
    "pym4ri",
    "A Python interface to the M4RI library for working with binary matrices.",
    -1,
    cssutils,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pym4ri(void) {
    import_array();
	return PyModule_Create(&m4rimodule);
}
