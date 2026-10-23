/* The frozen VCF search in C (X2-F, P1a-2; docs/h6-value.md, "TB-V3 실행 최적화").
 *
 * vcf(board, attacker, max_fours, node_limit, table) ->
 *     (found, attack_moves, completion_points, fours, nodes, exhausted, hits, stores)
 *   board      the game's board: 15 lists of 15 ints (-1, 0, 1), read once, never written
 *   table      0: search.mcts_v7._find_vcf_with_stats; 1: analysis.vcf_tt_v1 (a call-local table of
 *              exact no-win states: (board, attacker, max_fours - fours_used), stored only for a search
 *              that returned no VCF without a cut, answered at no node cost)
 *   moves are flat cell indices row * 15 + col, attack_moves in line order, completion_points ascending
 * The same recursion as the Python one, step by step: the same candidates in the same order (the empty
 * cells of the 5-cell windows with no opponent stone and >= 3 attacker stones, ascending), one node per
 * candidate taken before its legality check, the same legality (black forbidden points by the frozen
 * native classifier of rt30/native/forbidden.c, included below), the same five / four / block / reverse
 * four tests and the same returns, so the same result, witness, node count and cut. Nothing is cached
 * across calls.
 */
#include "../native/forbidden.c"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WHITE (-1)
#define MAX_DEPTH (CELLS + 2)

/* The 5-cell windows of search.mcts_v5.WINDOWS (cells inside the board, 4 directions), and per cell the
 * windows that contain it (at most 20). */
#define MAX_WINDOWS (CELLS * 4)
static int WINDOW[MAX_WINDOWS][5];
static int N_WINDOWS = 0;
static int CELL_WINDOWS[CELLS][20];
static int N_CELL_WINDOWS[CELLS];
static uint64_t ZOBRIST[CELLS][3];

/* A board with the stone counts of every window (index 0 black, 1 white), kept by place / unplace, and per
 * colour the number of windows with >= 4 of its stones and none of the other's (where a five can be). */
typedef struct {
    signed char b[CELLS];
    unsigned char count[MAX_WINDOWS][2];
    int four_windows[2];
} Board;

static int side(int player) { return player == BLACK ? 0 : 1; }

static int fourish(const Board *g, int w, int player) {
    return g->count[w][side(-player)] == 0 && g->count[w][side(player)] >= 4;
}

static void board_init(Board *g) {
    g->four_windows[0] = g->four_windows[1] = 0;
    for (int w = 0; w < N_WINDOWS; w++) {
        g->count[w][0] = g->count[w][1] = 0;
        for (int i = 0; i < 5; i++) {
            int v = g->b[WINDOW[w][i]];
            if (v == BLACK) g->count[w][0]++;
            else if (v == WHITE) g->count[w][1]++;
        }
        g->four_windows[0] += fourish(g, w, BLACK);
        g->four_windows[1] += fourish(g, w, WHITE);
    }
}

/* Put ``player`` on the empty cell p (delta +1) or take it back (delta -1), keeping the counts. */
static void board_set(Board *g, int p, int player, int delta) {
    for (int k = 0; k < N_CELL_WINDOWS[p]; k++) {
        int w = CELL_WINDOWS[p][k];
        g->four_windows[0] -= fourish(g, w, BLACK);
        g->four_windows[1] -= fourish(g, w, WHITE);
        g->count[w][side(player)] = (unsigned char)(g->count[w][side(player)] + delta);
        g->four_windows[0] += fourish(g, w, BLACK);
        g->four_windows[1] += fourish(g, w, WHITE);
    }
    g->b[p] = delta > 0 ? (signed char)player : EMPTY;
}

static void build_windows(void) {
    for (int r = 0; r < SIZE; r++)
        for (int c = 0; c < SIZE; c++)
            for (int a = 0; a < 4; a++) {
                int er = r + 4 * DR[a], ec = c + 4 * DC[a];
                if (!inside(er, ec)) continue;
                for (int i = 0; i < 5; i++) {
                    int p = (r + i * DR[a]) * SIZE + (c + i * DC[a]);
                    WINDOW[N_WINDOWS][i] = p;
                    CELL_WINDOWS[p][N_CELL_WINDOWS[p]++] = N_WINDOWS;
                }
                N_WINDOWS++;
            }
    uint64_t x = 0x9E3779B97F4A7C15ULL;  /* splitmix64: fixed keys, the table compares whole boards anyway */
    for (int p = 0; p < CELLS; p++)
        for (int v = 0; v < 3; v++) {
            uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            ZOBRIST[p][v] = z ^ (z >> 31);
        }
}

/* search.mcts_v5._window_candidates: ascending distinct empty cells; returns the count. */
static int window_candidates(const Board *g, int player, int minimum, int *out) {
    unsigned char mark[CELLS];
    memset(mark, 0, sizeof mark);
    int mine_side = side(player), other_side = side(-player);
    for (int w = 0; w < N_WINDOWS; w++) {
        if (g->count[w][other_side] || g->count[w][mine_side] < minimum) continue;
        const int *win = WINDOW[w];
        for (int i = 0; i < 5; i++)
            if (g->b[win[i]] == EMPTY) mark[win[i]] = 1;
    }
    int n = 0;
    for (int p = 0; p < CELLS; p++)
        if (mark[p]) out[n++] = p;
    return n;
}

/* renju.rules.forbidden_reason(...) is not None, on an empty cell (rt30/native/forbidden.c). */
static int black_forbidden(signed char *b, int p) {
    Counts k;
    axis_counts_local(b, p, &k);
    if (quiet_counts(&k)) return 0;
    b[p] = BLACK;
    int reason = classify(b, p, &k);
    b[p] = EMPTY;
    return reason != 0;
}

/* search.mcts._is_legal_for_player */
static int legal(signed char *b, int player, int p) {
    if (b[p] != EMPTY) return 0;
    return player == WHITE || !black_forbidden(b, p);
}

/* search.mcts._wins_for_player: black wins with exactly five, white with five or more. */
static int wins(signed char *b, int player, int p) {
    signed char old = b[p];
    b[p] = (signed char)player;
    int win = 0;
    for (int a = 0; a < 4 && !win; a++) {
        int n = run_length(b, p, a);
        win = player == BLACK ? n == 5 : n >= 5;
    }
    b[p] = old;
    return win;
}

/* search.mcts_v5._winning_moves (ascending); with ``first_only`` stops at the first one. Its candidates are
 * the empty cells of the windows counted in ``four_windows``, so with none there is no winning move. */
static int winning_moves(Board *g, int player, int *out, int first_only) {
    if (g->four_windows[side(player)] == 0) return 0;
    signed char *b = g->b;
    int cand[CELLS], n = window_candidates(g, player, 4, cand), k = 0;
    for (int i = 0; i < n; i++) {
        int p = cand[i];
        if (b[p] == EMPTY && wins(b, player, p) && legal(b, player, p)) {
            out[k++] = p;
            if (first_only) break;
        }
    }
    return k;
}

/* rt30.fast_extensions.winning_extensions_in_direction (= the frozen one), marking ``mark``. */
static void extensions(const signed char *b, int player, int anchor, int a, unsigned char *mark) {
    int row = anchor / SIZE, col = anchor % SIZE, dr = DR[a], dc = DC[a];
    if (b[anchor] != player) return;
    int back = 0, br = row - dr, bc = col - dc;
    while (inside(br, bc) && b[br * SIZE + bc] == player) { back++; br -= dr; bc -= dc; }
    int ahead = 0, ar = row + dr, ac = col + dc;
    while (inside(ar, ac) && b[ar * SIZE + ac] == player) { ahead++; ar += dr; ac += dc; }
    int ends[2][5] = {{br, bc, back + 1, -dr, -dc}, {ar, ac, ahead + 1, dr, dc}};
    for (int e = 0; e < 2; e++) {
        int er = ends[e][0], ec = ends[e][1], offset = ends[e][2], sr = ends[e][3], sc = ends[e][4];
        if (offset > 4 || !inside(er, ec) || b[er * SIZE + ec] != EMPTY) continue;
        int beyond = 0, xr = er + sr, xc = ec + sc;
        while (inside(xr, xc) && b[xr * SIZE + xc] == player) { beyond++; xr += sr; xc += sc; }
        int length = back + ahead + 2 + beyond;
        if (player == BLACK ? length == 5 : length >= 5) mark[er * SIZE + ec] = 1;
    }
}

/* search.mcts_v7._legal_completions: the legal winning extensions through ``anchor``, ascending. */
static int legal_completions(signed char *b, int player, int anchor, int *out) {
    unsigned char mark[CELLS];
    memset(mark, 0, sizeof mark);
    for (int a = 0; a < 4; a++) extensions(b, player, anchor, a, mark);
    int n = 0;
    for (int p = 0; p < CELLS; p++)
        if (mark[p] && legal(b, player, p)) out[n++] = p;
    return n;
}

/* ---- the call-local no-win table (analysis.vcf_tt_v1) ---- */
typedef struct {
    uint64_t hash;
    int remaining;  /* -1: empty slot */
    signed char board[CELLS];
} Entry;

typedef struct {
    Entry *slots;
    size_t mask, count;
    long hits;
} Table;

static int table_init(Table *t) {
    t->mask = 1023;
    t->count = 0;
    t->hits = 0;
    t->slots = malloc((t->mask + 1) * sizeof(Entry));
    if (!t->slots) return -1;
    for (size_t i = 0; i <= t->mask; i++) t->slots[i].remaining = -1;
    return 0;
}

static Entry *table_find(Table *t, uint64_t hash, int remaining, const signed char *b, int *found) {
    size_t i = (size_t)hash & t->mask;
    for (;;) {
        Entry *e = &t->slots[i];
        if (e->remaining < 0) { *found = 0; return e; }
        if (e->hash == hash && e->remaining == remaining && memcmp(e->board, b, CELLS) == 0) { *found = 1; return e; }
        i = (i + 1) & t->mask;
    }
}

static int table_store(Table *t, uint64_t hash, int remaining, const signed char *b) {
    if ((t->count + 1) * 2 > t->mask + 1) {  /* grow at half load */
        size_t old_mask = t->mask;
        Entry *old = t->slots;
        t->mask = old_mask * 2 + 1;
        t->slots = malloc((t->mask + 1) * sizeof(Entry));
        if (!t->slots) { t->slots = old; t->mask = old_mask; return -1; }
        for (size_t i = 0; i <= t->mask; i++) t->slots[i].remaining = -1;
        for (size_t i = 0; i <= old_mask; i++) {
            if (old[i].remaining < 0) continue;
            int found;
            Entry *e = table_find(t, old[i].hash, old[i].remaining, old[i].board, &found);
            *e = old[i];
        }
        free(old);
    }
    int found;
    Entry *e = table_find(t, hash, remaining, b, &found);
    if (!found) {
        e->hash = hash;
        e->remaining = remaining;
        memcpy(e->board, b, CELLS);
        t->count++;
    }
    return 0;
}

/* ---- the recursion ---- */
typedef struct {
    Board g;
    int attacker, max_fours;
    long limit, nodes;
    int exhausted, error;
    uint64_t hash;
    Table *table;           /* NULL: the plain search */
    int path[MAX_DEPTH];    /* the attacker's four at each depth */
    int block[MAX_DEPTH];   /* the defender's forced block at each depth */
    /* the result */
    int moves[MAX_DEPTH + 1], n_moves, fours;
    unsigned char completions[CELLS];
} Search;

static int take_node(Search *s) {
    if (s->nodes >= s->limit) {
        s->exhausted = 1;
        return 0;
    }
    s->nodes++;
    return 1;
}

static void place(Search *s, int p, int player) {
    board_set(&s->g, p, player, +1);
    s->hash ^= ZOBRIST[p][player + 1] ^ ZOBRIST[p][1];
}

static void unplace(Search *s, int p, int player) {
    board_set(&s->g, p, player, -1);
    s->hash ^= ZOBRIST[p][player + 1] ^ ZOBRIST[p][1];
}

/* The result at depth ``depth``: the path's fours, then ``last``; the path's blocks plus ``comps``. */
static int succeed(Search *s, int depth, int last, const int *comps, int n_comps, int fours) {
    for (int d = 0; d < depth; d++) s->moves[d] = s->path[d];
    s->moves[depth] = last;
    s->n_moves = depth + 1;
    memset(s->completions, 0, CELLS);
    for (int d = 0; d < depth; d++) s->completions[s->block[d]] = 1;
    for (int i = 0; i < n_comps; i++) s->completions[comps[i]] = 1;
    s->fours = fours;
    return 1;
}

static int recurse(Search *s, int fours_used);

/* search.mcts_v7._find_vcf_recursive after its exhausted check (vcf_tt_v1._body). */
static int body(Search *s, int fours_used) {
    int attacker = s->attacker, defender = -attacker, immediate[1];
    signed char *b = s->g.b;
    if (winning_moves(&s->g, attacker, immediate, 1))
        return succeed(s, fours_used, immediate[0], NULL, 0, fours_used);
    if (fours_used >= s->max_fours) return 0;
    int cand[CELLS], n = window_candidates(&s->g, attacker, 3, cand);
    for (int i = 0; i < n; i++) {
        int move = cand[i];
        if (!take_node(s)) return 0;
        if (!legal(b, attacker, move)) continue;
        if (wins(b, attacker, move)) return succeed(s, fours_used, move, NULL, 0, fours_used);
        place(s, move, attacker);
        int comps[CELLS], nc = legal_completions(b, attacker, move, comps), result = 0, done = 0;
        int dummy[1];
        if (nc == 0 || winning_moves(&s->g, defender, dummy, 1)) {
            /* no four, or the defender has a five: next candidate */
        } else if (nc >= 2) {
            result = succeed(s, fours_used, move, comps, nc, fours_used + 1);
            done = 1;
        } else {
            int blk = comps[0];
            if (!legal(b, defender, blk)) {
                result = succeed(s, fours_used, move, comps, nc, fours_used + 1);
                done = 1;
            } else if (!wins(b, defender, blk)) {
                place(s, blk, defender);
                int rev[CELLS];
                if (!legal_completions(b, defender, blk, rev)) {
                    s->path[fours_used] = move;
                    s->block[fours_used] = blk;
                    if (recurse(s, fours_used + 1)) result = done = 1;
                }
                unplace(s, blk, defender);
            }
        }
        unplace(s, move, attacker);
        if (s->error) return 0;
        if (done) return result;
    }
    return 0;
}

static int recurse(Search *s, int fours_used) {
    if (s->exhausted) return 0;
    if (s->table == NULL) return body(s, fours_used);
    int remaining = s->max_fours - fours_used, found;
    table_find(s->table, s->hash, remaining, s->g.b, &found);
    if (found) {
        s->table->hits++;
        return 0;
    }
    int result = body(s, fours_used);
    if (!result && !s->exhausted && !s->error && table_store(s->table, s->hash, remaining, s->g.b) < 0) s->error = 1;
    return result;
}

static PyObject *cell_tuple(int p) { return Py_BuildValue("(ii)", p / SIZE, p % SIZE); }

static PyObject *vcf(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "vcf(board, attacker, max_fours, node_limit, table)");
        return NULL;
    }
    Search *s = malloc(sizeof(Search));
    if (!s) return PyErr_NoMemory();
    if (read_board(args[0], s->g.b) < 0) { free(s); return NULL; }
    board_init(&s->g);
    s->attacker = (int)PyLong_AsLong(args[1]);
    s->max_fours = (int)PyLong_AsLong(args[2]);
    s->limit = PyLong_AsLong(args[3]);
    int use_table = PyObject_IsTrue(args[4]);
    if (PyErr_Occurred()) { free(s); return NULL; }
    s->nodes = 0;
    s->exhausted = s->error = 0;
    s->hash = 0;
    for (int p = 0; p < CELLS; p++) s->hash ^= ZOBRIST[p][s->g.b[p] + 1];
    Table table;
    s->table = NULL;
    if (use_table) {
        if (table_init(&table) < 0) { free(s); return PyErr_NoMemory(); }
        s->table = &table;
    }
    int found = recurse(s, 0);
    long hits = use_table ? table.hits : 0, stores = use_table ? (long)table.count : 0;
    if (use_table) free(table.slots);
    if (s->error) { free(s); return PyErr_NoMemory(); }
    PyObject *moves = PyTuple_New(found ? s->n_moves : 0), *comps = NULL, *out = NULL;
    if (!moves) goto done;
    for (int i = 0; found && i < s->n_moves; i++) {
        PyObject *t = cell_tuple(s->moves[i]);
        if (!t) goto done;
        PyTuple_SET_ITEM(moves, i, t);
    }
    int nc = 0;
    if (found)
        for (int p = 0; p < CELLS; p++) nc += s->completions[p];
    comps = PyTuple_New(nc);
    if (!comps) goto done;
    for (int p = 0, i = 0; found && p < CELLS; p++) {
        if (!s->completions[p]) continue;
        PyObject *t = cell_tuple(p);
        if (!t) goto done;
        PyTuple_SET_ITEM(comps, i++, t);
    }
    out = Py_BuildValue("(OOOilill)", found ? Py_True : Py_False, moves, comps, found ? s->fours : 0, s->nodes,
                        s->exhausted, hits, stores);
done:
    Py_XDECREF(moves);
    Py_XDECREF(comps);
    free(s);
    return out;
}

/* G0 probes: the helpers one at a time (cells as flat indices). */
static PyObject *probe(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "probe(board, what, player, cell)");
        return NULL;
    }
    static Board g;
    signed char *b = g.b;
    if (read_board(args[0], b) < 0) return NULL;
    board_init(&g);
    long what = PyLong_AsLong(args[1]), player = PyLong_AsLong(args[2]), cell = PyLong_AsLong(args[3]);
    if (PyErr_Occurred()) return NULL;
    int out[CELLS], n;
    switch (what) {
        case 0: n = window_candidates(&g, (int)player, (int)cell, out); break;     /* cell = minimum */
        case 1: n = winning_moves(&g, (int)player, out, 0); break;
        case 2: n = legal_completions(b, (int)player, (int)cell, out); break;
        case 3: return PyBool_FromLong(legal(b, (int)player, (int)cell));
        case 4: return PyBool_FromLong(wins(b, (int)player, (int)cell));
        default: PyErr_SetString(PyExc_ValueError, "what: 0 windows, 1 fives, 2 completions, 3 legal, 4 wins");
                 return NULL;
    }
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *t = cell_tuple(out[i]);
        if (!t) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, t);
    }
    return list;
}

static PyMethodDef vcf_methods[] = {
    {"vcf", (PyCFunction)(void (*)(void))vcf, METH_FASTCALL,
     "search.mcts_v7._find_vcf_with_stats (table=0) or analysis.vcf_tt_v1 (table=1), exact."},
    {"probe", (PyCFunction)(void (*)(void))probe, METH_FASTCALL, "the helpers, for the differential tests."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef vcf_module = {PyModuleDef_HEAD_INIT, "_vcf_core", NULL, -1, vcf_methods};

PyMODINIT_FUNC PyInit__vcf_core(void) {
    build_lines();
    build_windows();
    return PyModule_Create(&vcf_module);
}
