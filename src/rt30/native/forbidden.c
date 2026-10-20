/* Exact native renju.rules.forbidden_reason (H7 P1a-1, docs/h6-value.md; native-core-plan X1-B).
 *
 * forbidden(board, row, col) -> None | '장목' | '사사' | '삼삼'
 *   board  the game's board: a list of 15 lists of 15 ints (-1, 0, 1), read in place, never written
 * The same function as the Python one, step by step: the same quiet test on the per-axis counts, then the
 * same exact classifier on a private copy of the board (overline / five per axis where possible, fours as
 * distinct four-stone sets, open threes with the recursive legality of their straight-four extension),
 * with the same order of checks, so the same reason when several apply. Occupied or outside points raise
 * ValueError like the Python function. Nothing is cached.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SIZE 15
#define CELLS (SIZE * SIZE)
#define EMPTY 0
#define BLACK 1

static const int DR[4] = {1, 0, 1, 1};
static const int DC[4] = {0, 1, 1, -1};

/* Line geometry per (cell, axis): the cells of the full line through it in order, and its index. */
static unsigned char LINE[CELLS][4][SIZE];
static unsigned char LINE_LEN[CELLS][4];
static unsigned char LINE_IDX[CELLS][4];

static PyObject *OVERLINE, *DOUBLE_FOUR, *DOUBLE_THREE, *OUTSIDE_MESSAGE;

static int inside(int r, int c) { return 0 <= r && r < SIZE && 0 <= c && c < SIZE; }

static void build_lines(void) {
    for (int row = 0; row < SIZE; row++) {
        for (int col = 0; col < SIZE; col++) {
            for (int a = 0; a < 4; a++) {
                int r = row, c = col;
                while (inside(r - DR[a], c - DC[a])) { r -= DR[a]; c -= DC[a]; }
                int n = 0, idx = 0;
                while (inside(r, c)) {
                    if (r == row && c == col) idx = n;
                    LINE[row * SIZE + col][a][n++] = (unsigned char)(r * SIZE + c);
                    r += DR[a];
                    c += DC[a];
                }
                LINE_LEN[row * SIZE + col][a] = (unsigned char)n;
                LINE_IDX[row * SIZE + col][a] = (unsigned char)idx;
            }
        }
    }
}

/* rules.run_length on axis a: the run of the color at p through p. */
static int run_length(const signed char *b, int p, int a) {
    int color = b[p], count = 1, r0 = p / SIZE, c0 = p % SIZE;
    for (int sign = -1; sign <= 1; sign += 2) {
        int r = r0 + sign * DR[a], c = c0 + sign * DC[a];
        while (inside(r, c) && b[r * SIZE + c] == color) {
            count++;
            r += sign * DR[a];
            c += sign * DC[a];
        }
    }
    return count;
}

/* rules._axis_counts: (wide, near) per axis; ``get`` reads a cell. */
typedef struct { int wide[4], near[4]; } Counts;

static void axis_counts_local(const signed char *b, int p, Counts *out) {
    for (int a = 0; a < 4; a++) {
        const unsigned char *line = LINE[p][a];
        int n = LINE_LEN[p][a], i = LINE_IDX[p][a], near = 0, wide;
        for (int k = i - 3; k <= i + 3; k++)
            if (k != i && k >= 0 && k < n && b[line[k]] == BLACK) near++;
        wide = near;
        if (i - 4 >= 0 && b[line[i - 4]] == BLACK) wide++;
        if (i + 4 < n && b[line[i + 4]] == BLACK) wide++;
        out->wide[a] = wide;
        out->near[a] = near;
    }
}

/* rules._quiet_counts */
static int quiet_counts(const Counts *k) {
    int two = 0;
    for (int a = 0; a < 4; a++) {
        if (k->wide[a] > 2) return 0;
        if (k->near[a] == 2 && ++two > 1) return 0;
    }
    return 1;
}

/* rules._fours: the number of distinct four-stone sets (with an exact-five completion) through p on axis a. */
static int fours(signed char *b, int p, int a) {
    const unsigned char *line = LINE[p][a];
    int n = LINE_LEN[p][a], mi = LINE_IDX[p][a];
    int start_min = mi - 4 > 0 ? mi - 4 : 0, start_max = mi < n - 5 ? mi : n - 5;
    int sets[5][4], count = 0;
    for (int s = start_min; s <= start_max; s++) {
        int black[5], nb = 0, blank = -1, nblank = 0;
        for (int k = s; k < s + 5; k++) {
            int v = b[line[k]];
            if (v == BLACK) black[nb++] = line[k];
            else if (v == EMPTY) { blank = line[k]; nblank++; }
        }
        if (nb != 4 || nblank != 1) continue;
        b[blank] = BLACK;
        int exact = run_length(b, blank, a) == 5;
        b[blank] = EMPTY;
        if (!exact) continue;
        /* the window's black cells are in line order, so equal sets are equal arrays */
        int seen = 0;
        for (int j = 0; j < count && !seen; j++)
            seen = sets[j][0] == black[0] && sets[j][1] == black[1] && sets[j][2] == black[2] && sets[j][3] == black[3];
        if (!seen) {
            for (int j = 0; j < 4; j++) sets[count][j] = black[j];
            count++;
        }
    }
    return count;
}

/* rules._straight_four_after_extension: move and extension are already placed. */
static int straight_four_after_extension(signed char *b, int move, int extension, int a) {
    const unsigned char *line = LINE[move][a];
    int n = LINE_LEN[move][a], mi = LINE_IDX[move][a], ei = LINE_IDX[extension][a];
    int later = mi > ei ? mi : ei, earlier = mi < ei ? mi : ei;
    int start_min = later - 3 > 0 ? later - 3 : 0, start_max = earlier < n - 4 ? earlier : n - 4;
    for (int s = start_min; s <= start_max; s++) {
        int all = 1;
        for (int k = s; k < s + 4; k++)
            if (b[line[k]] != BLACK) { all = 0; break; }
        if (!all) continue;
        int g0 = line[s], g3 = line[s + 3];
        int br = g0 / SIZE - DR[a], bc = g0 % SIZE - DC[a], ar = g3 / SIZE + DR[a], ac = g3 % SIZE + DC[a];
        if (!inside(br, bc) || !inside(ar, ac)) continue;
        int before = br * SIZE + bc, after = ar * SIZE + ac;
        if (b[before] != EMPTY || b[after] != EMPTY) continue;
        b[before] = BLACK;
        int first = run_length(b, before, a) == 5;
        b[before] = EMPTY;
        b[after] = BLACK;
        int second = run_length(b, after, a) == 5;
        b[after] = EMPTY;
        if (first && second) return 1;
    }
    return 0;
}

static int classify(signed char *b, int p, const Counts *k);

static int forbidden_after_black_move(signed char *b, int p) {
    Counts k;
    axis_counts_local(b, p, &k);
    return classify(b, p, &k);
}

/* rules._open_three */
static int open_three(signed char *b, int move, int a) {
    const unsigned char *line = LINE[move][a];
    int n = LINE_LEN[move][a], i = LINE_IDX[move][a], near = 0;
    for (int k = i - 3; k <= i + 3; k++)
        if (k != i && k >= 0 && k < n && b[line[k]] == BLACK) near++;
    if (near < 2) return 0;
    int lo = i - 4 > 0 ? i - 4 : 0, hi = i + 5 < n ? i + 5 : n;
    for (int k = lo; k < hi; k++) {
        int pos = line[k];
        if (b[pos] != EMPTY) continue;
        b[pos] = BLACK;
        int five = 0;
        for (int v = 0; v < 4 && !five; v++) five = run_length(b, pos, v) == 5;
        int ok = 0;
        if (!five && straight_four_after_extension(b, move, pos, a))
            ok = forbidden_after_black_move(b, pos) == 0;
        b[pos] = EMPTY;
        if (ok) return 1;
    }
    return 0;
}

/* rules._classify_black_stone: 0 = None, 1 = overline, 2 = double four, 3 = double three. */
static int classify(signed char *b, int p, const Counts *k) {
    int lengths[4], nl = 0;
    for (int a = 0; a < 4; a++)
        if (k->wide[a] >= 4) lengths[nl++] = run_length(b, p, a);
    for (int j = 0; j < nl; j++)
        if (lengths[j] == 5) return 0;
    for (int j = 0; j < nl; j++)
        if (lengths[j] >= 6) return 1;
    int total = 0;
    for (int a = 0; a < 4; a++) {
        if (k->wide[a] >= 3) {
            total += fours(b, p, a);
            if (total >= 2) return 2;
        }
    }
    int candidates[4], nc = 0;
    for (int a = 0; a < 4; a++)
        if (k->near[a] >= 2) candidates[nc++] = a;
    int threes = 0;
    for (int index = 0; index < nc; index++) {
        if (threes + nc - index < 2) return 0;
        if (open_three(b, p, candidates[index])) {
            if (++threes >= 2) return 3;
        }
    }
    return 0;
}

/* Read the Python board into ``out``; 0 on success, -1 with an exception set. */
static int read_board(PyObject *board, signed char *out) {
    if (!PyList_Check(board) || PyList_GET_SIZE(board) != SIZE) {
        PyErr_SetString(PyExc_TypeError, "board must be a 15x15 list of lists");
        return -1;
    }
    for (int r = 0; r < SIZE; r++) {
        PyObject *row = PyList_GET_ITEM(board, r);
        if (!PyList_Check(row) || PyList_GET_SIZE(row) != SIZE) {
            PyErr_SetString(PyExc_TypeError, "board rows must be lists of 15 ints");
            return -1;
        }
        for (int c = 0; c < SIZE; c++) {
            long v = PyLong_AsLong(PyList_GET_ITEM(row, c));
            if (v == -1 && PyErr_Occurred()) return -1;
            out[r * SIZE + c] = (signed char)v;
        }
    }
    return 0;
}

/* One cell of the Python board (the board was checked by check_board); -2 with an exception set. */
static int cell(PyObject *board, int p) {
    long v = PyLong_AsLong(PyList_GET_ITEM(PyList_GET_ITEM(board, p / SIZE), p % SIZE));
    if (v == -1 && PyErr_Occurred()) return -2;
    return (int)v;
}

static int check_board(PyObject *board) {
    if (!PyList_Check(board) || PyList_GET_SIZE(board) != SIZE) {
        PyErr_SetString(PyExc_TypeError, "board must be a 15x15 list of lists");
        return -1;
    }
    for (int r = 0; r < SIZE; r++) {
        PyObject *row = PyList_GET_ITEM(board, r);
        if (!PyList_Check(row) || PyList_GET_SIZE(row) != SIZE) {
            PyErr_SetString(PyExc_TypeError, "board rows must be lists of 15 ints");
            return -1;
        }
    }
    return 0;
}

/* rules._axis_counts read straight from the Python board (only the +/-4 cells). */
static int axis_counts_board(PyObject *board, int p, Counts *out) {
    for (int a = 0; a < 4; a++) {
        const unsigned char *line = LINE[p][a];
        int n = LINE_LEN[p][a], i = LINE_IDX[p][a], near = 0, wide, v;
        for (int k = i - 3; k <= i + 3; k++) {
            if (k == i || k < 0 || k >= n) continue;
            if ((v = cell(board, line[k])) == -2) return -1;
            near += v == BLACK;
        }
        wide = near;
        if (i - 4 >= 0) {
            if ((v = cell(board, line[i - 4])) == -2) return -1;
            wide += v == BLACK;
        }
        if (i + 4 < n) {
            if ((v = cell(board, line[i + 4])) == -2) return -1;
            wide += v == BLACK;
        }
        out->wide[a] = wide;
        out->near[a] = near;
    }
    return 0;
}

static PyObject *forbidden(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "forbidden(board, row, col)");
        return NULL;
    }
    long row = PyLong_AsLong(args[1]), col = PyLong_AsLong(args[2]);
    if ((row == -1 || col == -1) && PyErr_Occurred()) return NULL;
    if (check_board(args[0]) < 0) return NULL;
    if (!inside((int)row, (int)col)) {
        PyErr_SetObject(PyExc_ValueError, OUTSIDE_MESSAGE);
        return NULL;
    }
    int p = (int)(row * SIZE + col), here = cell(args[0], p);
    if (here == -2) return NULL;
    if (here != EMPTY) {
        PyErr_SetObject(PyExc_ValueError, OUTSIDE_MESSAGE);
        return NULL;
    }
    Counts k;
    if (axis_counts_board(args[0], p, &k) < 0) return NULL;
    if (quiet_counts(&k)) Py_RETURN_NONE;
    signed char b[CELLS];  /* the exact classifier writes, so it works on a private copy */
    if (read_board(args[0], b) < 0) return NULL;
    b[p] = BLACK;
    int reason = classify(b, p, &k);
    switch (reason) {
        case 1: return Py_NewRef(OVERLINE);
        case 2: return Py_NewRef(DOUBLE_FOUR);
        case 3: return Py_NewRef(DOUBLE_THREE);
        default: Py_RETURN_NONE;
    }
}

static PyMethodDef methods[] = {
    {"forbidden", (PyCFunction)(void (*)(void))forbidden, METH_FASTCALL,
     "renju.rules.forbidden_reason(board, row, col), exact."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_forbidden", NULL, -1, methods};

PyMODINIT_FUNC PyInit__forbidden(void) {
    build_lines();
    OVERLINE = PyUnicode_FromString("\xec\x9e\xa5\xeb\xaa\xa9");          /* 장목 */
    DOUBLE_FOUR = PyUnicode_FromString("\xec\x82\xac\xec\x82\xac");       /* 사사 */
    DOUBLE_THREE = PyUnicode_FromString("\xec\x82\xbc\xec\x82\xbc");      /* 삼삼 */
    OUTSIDE_MESSAGE = PyUnicode_FromString(
        "\xeb\xb9\x88 \xeb\xb3\xb4\xeb\x93\x9c\xec\x9d\x98 \xeb\xb2\x94\xec\x9c\x84 \xeb\x82\xb4 "
        "\xec\xa2\x8c\xed\x91\x9c\xea\xb0\x80 \xed\x95\x84\xec\x9a\x94\xed\x95\xa9\xeb\x8b\x88\xeb\x8b\xa4");
    if (!OVERLINE || !DOUBLE_FOUR || !DOUBLE_THREE || !OUTSIDE_MESSAGE) return NULL;
    return PyModule_Create(&module);
}
