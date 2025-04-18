#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <m4ri/m4ri.h>
#include <numpy/ndarrayobject.h>
#include <cstdlib>
#include <string>
#include <ranges>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>

/**
 * @brief Utility functions for testing and debugging. 
 */
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

/**
 * @brief Safe versions of some M4RI utilities. 
 * 
 * @note Same interface as usual M4RI functions, 
 * but with guardrails for 0-sizes matrices. 
 */
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

bool safe_mzd_is_zero(const mzd_t *A) { return (A->nrows && A->ncols) ? mzd_is_zero(A) : true; }

/**
 * @brief Core CSS code utilities
 * 
 * @note Methods to compute rank, the eta and gamma conversions, 
 * and to build the CSS code from the classical Tanner graph. 
 */
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

inline void build_HxHz_from_edges(mzd_t *Hx, mzd_t *Hz, std::pair<int, int> &shape, 
                                  std::vector<std::pair<int, int>> &edges){
    auto &[m, n] = shape;
    // Fill out first block Im x H' in Hx
    for(int k = 0; k < m; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(Hx, k*n + (v-m), k*m + u, 1);
        
    // Fill out second block H x In in Hx
    for(int k = 0; k < n; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(Hx, u*n + k, m*m + (v-m)*n + k, 1);
        
    // Fill out first block H' x Im in Hz
    for(int k = 0; k < m; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(Hz, (v-m)*m + k, u*m + k, 1);
        
    // Fill out second block In x H in Hz
    for(int k = 0; k < n; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(Hz, k*m + u, m*m + k*n + (v-m), 1);
}

inline void build_HxTHzT_from_edges(mzd_t *HxT, mzd_t *HzT, std::pair<int, int> &shape, 
                                  std::vector<std::pair<int, int>> &edges){
    auto &[m, n] = shape;
    // Fill out first block Im x H in HxT
    for(int k = 0; k < m; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(HxT, k*m + u, k*n + (v-m), 1);
        
    // Fill out second block H'x In in HxT
    for(int k = 0; k < n; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(HxT, m*m + (v-m)*n + k, u*n + k, 1);
        
    // Fill out first block H x Im in HzT
    for(int k = 0; k < m; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(HzT, u*m + k, (v-m)*m + k, 1);
        
    // Fill out second block In x H' in HzT
    for(int k = 0; k < n; k++) 
        for(auto &[u, v] : edges) 
            mzd_write_bit(HzT, m*m + k*n + (v-m), k*m + u, 1);
}

/**
 * @brief Functions for simulating the erasure channel. 
 */
int sample_erasure(double p, mzp_t *select_erased_cols) {
    // Set RNG
    static std::random_device rd; static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    // Reset permutation to identity
    mzp_set_ui(select_erased_cols, 1);
    
    // Sample erasure
    int e_weight = 0;
    for(int i = 0; i < select_erased_cols->length; i++) 
        if(dis(gen) < p) 
            select_erased_cols->values[e_weight++] = i;
            
    return e_weight;
}

inline int erasure_dim_gap_from(mzd_t *HT, mzd_t *canvas, mzd_t *erasure_window, mzp_t *select_erased_cols, 
int e_weight, int &rank_Ebar) {
    // Select erased and intact columns of H (rows of HT)
    mzd_copy(canvas, HT);
    mzd_apply_p_left(canvas, select_erased_cols);
    
    // Set window to H_erased
    erasure_window->nrows = e_weight;
    erasure_window->data = canvas->data;
    int dimension_gap = -rank(erasure_window);
    
    // Set window to H_intact
    erasure_window->nrows = canvas->nrows - e_weight;
    erasure_window->data = canvas->data + e_weight * canvas->rowstride;
    rank_Ebar = rank(erasure_window);
    dimension_gap += rank_Ebar;

    return dimension_gap;
}

bool logical_error_within_erasure(mzd_t *eta, mzd_t *H, mzd_t *eta_canvas, mzd_t *canvas, 
                                         mzp_t *select_erased_cols, int e_weight) {
    // Compute gamma(H^E) 
    for(int i = 0; i < H->nrows; i++)
        for(int j = 0; j < e_weight; j++) 
            mzd_write_bit(canvas, i, j, mzd_read_bit(H, i, select_erased_cols->values[j]));
    mzd_t *canvas_erasure_window = mzd_init_window(canvas, 0, 0, canvas->nrows, e_weight);
    mzd_t *gamma_H_E = chk2gen(canvas_erasure_window);

    // Initialize window to access the erased columns of eta
    for(int i = 0; i < eta->nrows; i++)
        for(int j = 0; j < e_weight; j++) 
            mzd_write_bit(eta_canvas, i, j, mzd_read_bit(eta, i, select_erased_cols->values[j]));
    const mzd_t *eta_erasure_window = mzd_init_window_const(eta_canvas, 0, 0, eta->nrows, e_weight);
    
    // Compute the condition for the erasure to contain an X/Z type logical error
    mzd_t *correctability_matrix = mzd_mul(NULL, eta_erasure_window, gamma_H_E, 0);
    bool logical_error = !safe_mzd_is_zero(correctability_matrix);

    // Free allocated matrices, eta window and revert the permutation on eta(Hx)
    mzd_free_window(canvas_erasure_window);
    mzd_free(gamma_H_E);
    mzd_free_window((mzd_t *)eta_erasure_window);
    mzd_free(correctability_matrix);

    return logical_error;
}

/**
 * @brief Monte Carlo simulations of the logical error rate for the erasure channel. 
 * 
 * @note You can choose whether to compute the failure condition through rank-based 
 * conditions (lower bound) or by the eta-gamma approach (exact).
 */
inline void MC_erasure_plog_rank(int num_trials, std::vector<double> &p_vals, mzd_t *Hx, mzd_t *Hz, 
mzp_t *select_erased_cols, PyArrayObject *means, PyArrayObject *stds, PyArrayObject *rank_stats) {
    // Construct HxT and HzT
    mzd_t *HxT = safe_mzd_transpose(NULL, Hx), *HzT = safe_mzd_transpose(NULL, Hz);
    
    // Preallocate space to hold the matrices whose ranks shall be computed
    mzd_t *canvas = mzd_init(HxT->nrows, HxT->ncols);

    // Use a window to access the submatrices corresponding to erased and intact bits
    mzd_t *erasure_window = mzd_init_window(canvas, 0, 0, canvas->nrows, canvas->ncols);

    // Precompute rank(H)
    rci_t rank_H = rank(Hx) + rank(Hz);

    // Loop over all p_values and do MC simulation
    for(std::vector<double>::size_type idx = 0; idx < p_vals.size(); idx++){
        double p = p_vals[idx]; int failures = 0; int full_rank = 0;
        for(int t = 0; t < num_trials; t++){
            // Sample erasure
            int e_weight = sample_erasure(p, select_erased_cols);
            
            // Compute dimension gap condition
            int dimension_gap = 2*e_weight - rank_H; // - rank_H_E + rank_H_E_
            // Add dimension gap due to the X part of the CSS code
            int rankEbar_X = 0;
            dimension_gap += erasure_dim_gap_from(HxT, canvas, erasure_window, select_erased_cols, e_weight, rankEbar_X);
            // Add dimension gap due to the Z part of the CSS code
            int rankEbar_Z = 0;
            dimension_gap += erasure_dim_gap_from(HzT, canvas, erasure_window, select_erased_cols, e_weight, rankEbar_Z);
            
            // Admit failure only when the dimension gap necessary condition fails
            if(dimension_gap > 0) failures++;

            if((rankEbar_X + rankEbar_Z) == MIN(Hx->nrows + Hz->nrows, 2*(canvas->nrows - e_weight))) full_rank++;
        }
        // Estimate failure rate and estimator variance
        long long M = num_trials, m = failures;
        *(double *)PyArray_GETPTR1(means, idx) = (double) m / M;
        *(double *)PyArray_GETPTR1(stds, idx) = sqrt( (double) (m*(M - m)) / (M*(M - 1)) );
        *(double *)PyArray_GETPTR1(rank_stats, idx) = (double) full_rank / M;
    }
    // Cleanup
    mzd_free_window(erasure_window), mzd_free(canvas), mzd_free(HxT), mzd_free(HzT); 
}

inline void MC_erasure_plog_rank_only_X(int num_trials, std::vector<double> &p_vals, mzd_t *Hx, 
mzp_t *select_erased_cols, PyArrayObject *means, PyArrayObject *stds, PyArrayObject *rank_stats) {
    // Construct HxT and HzT
    mzd_t *HxT = safe_mzd_transpose(NULL, Hx);
    
    // Preallocate space to hold the matrices whose ranks shall be computed
    mzd_t *canvas = mzd_init(HxT->nrows, HxT->ncols);

    // Use a window to access the submatrices corresponding to erased and intact bits
    mzd_t *erasure_window = mzd_init_window(canvas, 0, 0, canvas->nrows, canvas->ncols);

    // Precompute rank(H)
    rci_t rank_Hx = rank(Hx);

    // Loop over all p_values and do MC simulation
    for(std::vector<double>::size_type idx = 0; idx < p_vals.size(); idx++){
        double p = p_vals[idx]; int failures = 0; int full_rank = 0;
        for(int t = 0; t < num_trials; t++){
            // Sample erasure
            int e_weight = sample_erasure(p, select_erased_cols);
            
            // Compute dimension gap condition
            int dimension_gap = e_weight - rank_Hx; // - rank_H_E + rank_H_E_
            // Add dimension gap due to the X part of the CSS code
            int rankEbar_X = 0;
            dimension_gap += erasure_dim_gap_from(HxT, canvas, erasure_window, select_erased_cols, e_weight, rankEbar_X);
            
            // Admit failure only when the dimension gap necessary condition fails
            if(dimension_gap > 0) failures++;

            if((rankEbar_X) == MIN(Hx->nrows, canvas->nrows - e_weight)) full_rank++;
        }
        // Estimate failure rate and estimator variance
        long long M = num_trials, m = failures;
        *(double *)PyArray_GETPTR1(means, idx) = (double) m / M;
        *(double *)PyArray_GETPTR1(stds, idx) = sqrt( (double) (m*(M - m)) / (M*(M - 1)) );
        *(double *)PyArray_GETPTR1(rank_stats, idx) = (double) full_rank / M;
    }
    // Cleanup
    mzd_free_window(erasure_window), mzd_free(canvas), mzd_free(HxT); 
}

inline void MC_erasure_plog_eta_gamma(int num_trials, std::vector<double> &p_vals, 
mzd_t *Hx, mzd_t *Hz, mzp_t *select_erased_cols, PyArrayObject *means, PyArrayObject *stds) {
    // Preallocate space to hold the submatrices or copies of Hx/Hz
    mzd_t *canvas = mzd_init(Hx->nrows, Hx->ncols);
    
    // Precompute eta(Hx), eta(Hz)
    mzd_copy(canvas, Hx); mzd_t *eta_Hx = gen2chk(canvas);
    mzd_copy(canvas, Hz); mzd_t *eta_Hz = gen2chk(canvas);
    mzd_t *eta_canvas = mzd_init(MAX(eta_Hx->nrows, eta_Hz->nrows), MAX(eta_Hx->ncols, eta_Hz->ncols));
    
    // Loop over all p_values and do MC simulation
    for(std::vector<double>::size_type idx = 0; idx < p_vals.size(); idx++){
        double p = p_vals[idx]; int failures = 0;
        for(int t = 0; t < num_trials; t++){            
            // Sample erasure
            int e_weight = sample_erasure(p, select_erased_cols);
            
            // Check for the existence of a X type logical error within the erasure
            if(logical_error_within_erasure(eta_Hx, Hz, eta_canvas, canvas, select_erased_cols, e_weight)) {
                failures++;
                continue;
            }

            // Check for the existence of a Z type logical error within the erasure
            if(logical_error_within_erasure(eta_Hz, Hx, eta_canvas, canvas, select_erased_cols, e_weight)) 
                failures++;
        }
        // Estimate failure rate and estimator variance
        long long M = num_trials, m = failures;
        *(double *)PyArray_GETPTR1(means, idx) = (double) m / M;
        *(double *)PyArray_GETPTR1(stds, idx) = sqrt( (double) (m*(M - m)) / (M*(M - 1)) );
    }
    // Cleanup
    mzd_free(canvas), mzd_free(eta_canvas), mzd_free(eta_Hx), mzd_free(eta_Hz); 
}

inline void MC_erasure_plog_eta_gamma_only_X(int num_trials, std::vector<double> &p_vals, 
mzd_t *Hx, mzd_t *Hz, mzp_t *select_erased_cols, PyArrayObject *means, PyArrayObject *stds) {
    // Preallocate space to hold the submatrices or copies of Hz
    mzd_t *canvas = mzd_init(Hz->nrows, Hz->ncols);
    
    // Precompute eta(Hx)
    mzd_copy(canvas, Hx); mzd_t *eta_Hx = gen2chk(canvas);
    mzd_t *eta_canvas = mzd_init(eta_Hx->nrows, eta_Hx->ncols);
    
    // Loop over all p_values and do MC simulation
    for(std::vector<double>::size_type idx = 0; idx < p_vals.size(); idx++){
        double p = p_vals[idx]; int failures = 0;
        for(int t = 0; t < num_trials; t++){
            // Sample erasure
            int e_weight = sample_erasure(p, select_erased_cols);
            
            // Check for the existence of a X type logical error within the erasure
            if(logical_error_within_erasure(eta_Hx, Hz, eta_canvas, canvas, select_erased_cols, e_weight))
                failures++;
        }
        // Estimate failure rate and estimator variance
        long long M = num_trials, m = failures;
        *(double *)PyArray_GETPTR1(means, idx) = (double) m / M;
        *(double *)PyArray_GETPTR1(stds, idx) = sqrt( (double) (m*(M - m)) / (M*(M - 1)) );
    }
    // Cleanup
    mzd_free(canvas), mzd_free(eta_Hx); 
}

/**
 * @brief Conversion methods between Python objects and data structures from M4RI and C/C++. 
 */
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

int parse_edgelist(PyObject *edgelist, void *vec){
    std::vector<std::pair<int, int>> *V = (std::vector<std::pair<int, int>> *) vec;
    if(!PySequence_Check(edgelist)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence for the edgelist");
        return 0;
    }
    for(Py_ssize_t i = 0; i < PySequence_Length(edgelist); i++){  
        PyObject *tuple_obj = PySequence_GetItem(edgelist, i);
        if (!PyTuple_Check(tuple_obj) || PyTuple_Size(tuple_obj) != 2) {
            PyErr_SetString(PyExc_TypeError, "Iterable must contain tuples of size 2");
            return 0;
        }
        PyObject *u = PyTuple_GetItem(tuple_obj, 0), *v = PyTuple_GetItem(tuple_obj, 1);
        if (PyArray_IsIntegerScalar(u) && PyArray_IsIntegerScalar(v)) {
            V->emplace_back((int)PyLong_AsLong(u), (int)PyLong_AsLong(v));
        } else {
            PyErr_SetString(PyExc_TypeError, "Tuples must contain integers");
            return 0;
        }
    }
    return 1;
}

int parse_list(PyObject *list, void *vec){
    std::vector<double> *V = (std::vector<double> *) vec;
    if(!PySequence_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence for the p_vals list");
        return 0;
    }

    for(Py_ssize_t i = 0; i < PySequence_Length(list); i++){
        PyObject *item_obj = PySequence_GetItem(list, i);
        if (!PyFloat_Check(item_obj)) {
            PyErr_SetString(PyExc_TypeError, "p values must be float");
            return 0;
        }
        V->emplace_back(PyFloat_AsDouble(item_obj));
    }
    return 1;
}

/**
 * @brief Python wrapped CSS code utilities. 
 */
static PyObject *gen2chk(PyObject *Py_UNUSED(self), PyObject *args) {
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

static PyObject *chk2gen(PyObject *Py_UNUSED(self), PyObject *args) {
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

static PyObject *rank(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *M_obj;
    if (!PyArg_ParseTuple(args, "O", &M_obj))
        return NULL;

    mzd_t *M = PyArray_ToMzd(M_obj);
    if (!M) return NULL;

    rci_t r = rank(M);
    mzd_free(M);
    return Py_BuildValue("i", r);
}

static PyObject *gf2_mul(PyObject *Py_UNUSED(self), PyObject *args) {
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

    mzd_t *C = (!A->nrows || !A->ncols || !B->ncols) ? mzd_init(A->ncols, B->nrows) : mzd_mul(NULL, A, B, 0);    
    PyObject *C_obj = MzdToPyArray(C);
    mzd_free(A), mzd_free(B), mzd_free(C);

    return C_obj;
}

static PyObject *gf2_linsolve(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *A_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &b_obj))
        return NULL;

    mzd_t *A = PyArray_ToMzd(A_obj);
    if (!A) return NULL;

    mzd_t *b = PyArray_ToMzd(b_obj);
    if (!b) {
        mzd_free(A);
        return NULL;
    }

    if (A->nrows != b->nrows) {
        PyErr_SetString(PyExc_ValueError, "Number of rows of A and b must be equal");
        mzd_free(A);
        mzd_free(b);
        return NULL;
    }

    if(mzd_solve_left(A, b, 0, 1) == -1){
        PyErr_SetString(PyExc_ValueError, "No solution was found for Ax = b");
        mzd_free(A);
        mzd_free(b);
        return NULL;        
    }
    PyObject *x_obj = MzdToPyArray(b);
    mzd_free(A), mzd_free(b);

    return x_obj;
}

static PyObject *MC_erasure_plog(PyObject *Py_UNUSED(self), PyObject *args) {
    // Parse all arguments to C/C++ data structures
    std::pair<int, int> shape;
    std::vector<std::pair<int, int>> edges;
    int num_trials;
    std::vector<double> p_vals;
    int rank_method, only_X;
    if (!PyArg_ParseTuple(args, "(ii)O&iO&pp", &(shape.first), &(shape.second), 
                        parse_edgelist, (void *)&edges, &num_trials, 
                        parse_list, (void *)&p_vals, 
                        &rank_method, &only_X)) return NULL;
    
    // Prepare np.arrays to be returned
    npy_intp dims[1] = {(npy_intp) p_vals.size()};
    PyArrayObject *means = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *stds = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *rank_stats = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    
    // Construct Hx = [Im x H'| H x In] and Hz = [H'x Im | In x H]
    auto &[m, n] = shape;
    rci_t num_checks = m*n, num_qubits = m*m + n*n;
    mzd_t *Hx = mzd_init(num_checks, num_qubits), *Hz = mzd_init(num_checks, num_qubits);
    build_HxHz_from_edges(Hx, Hz, shape, edges);

    // Prepare column permutation to select erased columns
    mzp_t *select_erased_cols = mzp_init(num_qubits);        

    // Run Monte Carlo estimation of the logical error rate for the erasure channel
    if(rank_method)
        only_X ? MC_erasure_plog_rank_only_X(num_trials, p_vals, Hx, select_erased_cols, means, stds, rank_stats) :
                 MC_erasure_plog_rank(num_trials, p_vals, Hx, Hz, select_erased_cols, means, stds, rank_stats); 
    else
        only_X ? MC_erasure_plog_eta_gamma_only_X(num_trials, p_vals, Hx, Hz, select_erased_cols, means, stds) :
                 MC_erasure_plog_eta_gamma(num_trials, p_vals, Hx, Hz, select_erased_cols, means, stds); 

    // Cleanup
    mzd_free(Hx), mzd_free(Hz), mzp_free(select_erased_cols); 

    // Wrap results in a dict
    PyObject* result_dict;
    Py_INCREF(means), Py_INCREF(stds);
    if(rank_method) {
        Py_INCREF(rank_stats);
        result_dict = Py_BuildValue("{s:O, s:O, s:O}", "mean", means, "std", stds, "rank_stats", rank_stats);
        Py_DECREF(rank_stats);
    } else {
        result_dict = Py_BuildValue("{s:O, s:O}", "mean", means, "std", stds);
    }
    Py_DECREF(means), Py_DECREF(stds);

    return result_dict;    
}

/**
 * @brief Testing routines. 
 */
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
	const rci_t m = 5, n = 5;

	rci_t example[m][n] = {
		{1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 0, 1, 0},
		{0, 0, 0, 0, 1},
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

/**
 * @brief Docstrings for the PyModule methods.
 */
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

PyDoc_STRVAR(gf2_linsolve_doc, 
    "chk2gen(A: numpy.ndarray[bool], b: numpy.ndarray[bool]) -> numpy.ndarray[bool]\n"
    "\n"
    "Solve the linear system Ax = b for x.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "A : numpy.ndarray of bool\n"
    "    The matrix of the linear system, must have shape (m, n).\n"
    "b : numpy.ndarray of bool\n"
    "    The right-hand side vector, must have shape (m, 1).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray of bool\n"
    "    The unique solution of the system Ax = b, of shape (n, 1).\n"
    "    If the system has no solution or multiple solutions, an error is triggered.\n"
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

PyDoc_STRVAR(MC_erasure_plog_doc, 
    "MC_erasure_plog(shape: tuple[int, int], edges: list[tuple[int, int]], num_trials: int, "
    "p_vals: list[double], rank_method: bool) -> dict[np.array, np.array]\n"
    "\n"
    "Estimates the logical error rate for the erasure channel, "
    "by checking the existence of logical errors within the erasure support.\n"
    "If rank_method is True, estimates a lower bound on the logical error rate "
    "based on a rank-related necessary condition."
    "\n"
    "Parameters\n"
    "----------\n"
    "shape: tuple[ii]; number of check/bit nodes in the classical Tanner graph. \n"
    "edges: list[tuple[ii]]; list of edges of the classical Tanner graph. \n"
    "num_trials: int; number of trials in the MC simulation. \n"
    "p_vals: list[double]; physical erasure rates for which to compute the logical error rate. \n"

    "    The matrix.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "dict{'mean': np.array, 'std': np.array}: means and std of the failure rate for each p_val. \n"
    "\n"
    "Example call: \n"
    "MC_erasure_plog(shape=(m, n), edges=[(u1, v1), ..., (uk, vk)], num_trials=1000, p_vals=[0.1, 0.2, 0.3])\n"
    "\n"
    "Note: \n"
    "The edges assume that the nodes are numbered from [0..m+n-1], i.e., check nodes are labeled in [0..m-1],"
    "whereas bit nodes are labeled in [m..m+n-1]. \n"
);

/**
 * @brief PyModule definition boilerplate.
 */
static PyMethodDef cssutils[] = {
    {"gen2chk", gen2chk, METH_VARARGS, gen2chk_doc},
    {"chk2gen", chk2gen, METH_VARARGS, chk2gen_doc},
	{"gf2_mul", gf2_mul, METH_VARARGS, gf2_mul_doc},
	{"gf2_linsolve", gf2_linsolve, METH_VARARGS, gf2_linsolve_doc},
	{"rank", rank, METH_VARARGS, rank_doc},
    {"MC_erasure_plog", MC_erasure_plog, METH_VARARGS, MC_erasure_plog_doc},
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
