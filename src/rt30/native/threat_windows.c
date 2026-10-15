/* Exact native scan for search.mcts_v5._threat_windows (docs/web-ser5.md §9.8, fast3 X1-A).
 *
 * windows(board, player, minimum, flat_index, windows) -> list
 *   board       the game's board: a list of 15 lists of 15 ints (-1, 0, 1), read in place
 *   flat_index  bytes of 572*5 cell indices (r * 15 + c), in WINDOWS order
 *   windows     the WINDOWS tuple; the result holds its own window objects, in WINDOWS order
 * A window is kept when none of its cells is -player and at least `minimum` cells are player:
 * the frozen generator's condition. Nothing is cached and the board is never written.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SIZE 15
#define CELLS (SIZE * SIZE)

static PyObject *windows(PyObject *self, PyObject *args) {
    PyObject *board, *index, *wins;
    long player, minimum;
    if (!PyArg_ParseTuple(args, "OllSO", &board, &player, &minimum, &index, &wins)) return NULL;
    if (!PyList_Check(board) || PyList_GET_SIZE(board) != SIZE || !PyTuple_Check(wins)) {
        PyErr_SetString(PyExc_TypeError, "board must be a 15x15 list of lists, windows a tuple");
        return NULL;
    }
    Py_ssize_t n = PyTuple_GET_SIZE(wins);
    if (PyBytes_GET_SIZE(index) != n * 5) {
        PyErr_SetString(PyExc_ValueError, "flat_index must hold 5 cells per window");
        return NULL;
    }
    signed char cell[CELLS];
    for (int r = 0; r < SIZE; r++) {
        PyObject *row = PyList_GET_ITEM(board, r);
        if (!PyList_Check(row) || PyList_GET_SIZE(row) != SIZE) {
            PyErr_SetString(PyExc_TypeError, "board rows must be lists of 15 ints");
            return NULL;
        }
        for (int c = 0; c < SIZE; c++) {
            long v = PyLong_AsLong(PyList_GET_ITEM(row, c));
            if (v == -1 && PyErr_Occurred()) return NULL;
            /* player's stone -> 1, the opponent's -> -1, empty 0 */
            cell[r * SIZE + c] = (signed char)(v * player);
        }
    }
    const unsigned char *ix = (const unsigned char *)PyBytes_AS_STRING(index);
    PyObject *out = PyList_New(0);
    if (out == NULL) return NULL;
    for (Py_ssize_t w = 0; w < n; w++) {
        const unsigned char *p = ix + 5 * w;
        int own = 0, bad = 0;
        for (int k = 0; k < 5; k++) {
            signed char v = cell[p[k]];
            if (v < 0) { bad = 1; break; }
            own += v;
        }
        if (!bad && own >= minimum) {
            if (PyList_Append(out, PyTuple_GET_ITEM(wins, w)) < 0) { Py_DECREF(out); return NULL; }
        }
    }
    return out;
}

static PyMethodDef methods[] = {
    {"windows", windows, METH_VARARGS, "Threat windows of player with at least minimum stones."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_threat_windows", NULL, -1, methods};

PyMODINIT_FUNC PyInit__threat_windows(void) { return PyModule_Create(&module); }
