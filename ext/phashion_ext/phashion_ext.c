#include "ruby.h"
#include "pHash.h"
#ifdef HAVE_RUBY_THREAD_H
#include <ruby/thread.h>
#else
#include <ruby/intern.h>
void *
rb_thread_call_without_gvl(void *(*func)(void *data), void *data1,
			    rb_unblock_function_t *ubf, void *data2) {
  return (void *)rb_thread_blocking_region(
      (rb_blocking_function_t *)func, data1,
      (rb_unblock_function_t *)ubf, data2);
}
#endif

/* ── phash64: 64-bit DCT perceptual hash (libpHash default) ───────────────────
 *
 * Wraps ph_dct_imagehash() from libpHash 0.9.6:
 *   1. Load image with CImg
 *   2. Resize to 32×32
 *   3. Convert to grayscale
 *   4. 2D DCT-II
 *   5. Top-left 8×8 submatrix (64 coefficients)
 *   6. Compare each coefficient to the mean → 64-bit hash (ulong64)
 *
 * Hamming distance via ph_hamming_distance() (popcount of XOR).
 */

struct nogvl_hash_args {
  const char * filename;
  ulong64 hash;
  int retval;
};

static void * nogvl_hash(struct nogvl_hash_args * args) {
  ulong64 hash;

  args->retval = ph_dct_imagehash(args->filename, hash);
  args->hash = hash;

  return NULL;
}

static VALUE image_hash_for(VALUE self, VALUE _filename) {
    ulong64 hash;
    struct nogvl_hash_args args;

    args.filename = StringValuePtr(_filename);
    args.retval = -1;

    rb_thread_call_without_gvl((void *(*)(void *))nogvl_hash,
        (void *)&args, RUBY_UBF_PROCESS, 0);

    if (-1 == args.retval) {
      rb_raise(rb_eRuntimeError, "Unknown pHash error");
    }
    return ULL2NUM(args.hash);
}


static VALUE hamming_distance(VALUE self, VALUE a, VALUE b) {
    int result = 0;
    result = ph_hamming_distance(NUM2ULL(a), NUM2ULL(b));
    if (-1 == result) {
      rb_raise(rb_eRuntimeError, "Unknown pHash error");
    }
    return INT2NUM(result);
}

static VALUE hamming_distance2(VALUE self, VALUE a, VALUE b) {
	double distance;
	uint8_t* hashA;
	uint8_t* hashB;
	int lenA = RARRAY_LEN(a);
	int lenB = RARRAY_LEN(b);

	hashA = (uint8_t*)xcalloc(lenA, sizeof(uint8_t));
	hashB = (uint8_t*)xcalloc(lenB, sizeof(uint8_t));

	for(int i = 0; i < lenA; i++) {
		hashA[i] = NUM2INT(rb_ary_entry(a, i));
	}
	for(int i = 0; i < lenB; i++) {
		hashB[i] = NUM2INT(rb_ary_entry(b, i));
	}

	distance = ph_hammingdistance2(hashA, lenA, hashB, lenB);

	xfree(hashA);
	xfree(hashB);

	return DBL2NUM(distance);
}

static VALUE mh_hash_for(VALUE self, VALUE filename, VALUE alpha, VALUE lvl) {
	uint8_t* result;
	int n;
	VALUE array;
	result = ph_mh_imagehash(StringValuePtr(filename),
			n,
			NUM2DBL(alpha),
			NUM2DBL(lvl));
	array = rb_ary_new2(n);

	for(int i = 0; i < n; i++) {
		rb_ary_push(array, INT2FIX(result[i]));
	}

	xfree(result);

	return array;
}

static VALUE texthash_for(VALUE self, VALUE file) {
    int nbpoints, i;
    VALUE list;
    VALUE point_class;

    TxtHashPoint *points = ph_texthash(StringValuePtr(file), &nbpoints);

    point_class = rb_const_get(self, rb_intern("TextHashPoint"));

    list = rb_ary_new2((long)nbpoints);

    for(i = 0; i < nbpoints; i++) {
      VALUE point;
      VALUE args[2];

      args[0] = ULL2NUM(points[i].hash);
      args[1] = ULL2NUM(points[i].index);

      point = rb_class_new_instance(2, args, point_class);
      rb_ary_push(list, point);
    }

    free(points);

    return list;
}

static TxtHashPoint * rb2phash_points(VALUE list) {
    int i;
    TxtHashPoint * txt_list;

    txt_list = (TxtHashPoint *)xcalloc(RARRAY_LEN(list), sizeof(TxtHashPoint));

    for(i = 0; i < RARRAY_LEN(list); i++) {
      VALUE elem = rb_ary_entry(list, i);
      txt_list[i].hash = NUM2ULL(rb_funcall(elem, rb_intern("hash"), 0));
      txt_list[i].index = NUM2INT(rb_funcall(elem, rb_intern("index"), 0));
    }

    return txt_list;
}

static VALUE textmatches_for(VALUE self, VALUE list1, VALUE list2) {
    int nbmatches, i;
    VALUE list;
    VALUE match_class;
    TxtHashPoint *txt_list1;
    TxtHashPoint *txt_list2;

    txt_list1 = rb2phash_points(list1);
    txt_list2 = rb2phash_points(list2);

    TxtMatch *matches = ph_compare_text_hashes(txt_list1, RARRAY_LEN(list1),
                                               txt_list2, RARRAY_LEN(list2),
                                               &nbmatches);

    xfree(txt_list1);
    xfree(txt_list2);

    match_class = rb_const_get(self, rb_intern("TextMatch"));

    list = rb_ary_new2((long)nbmatches);

    for(i = 0; i < nbmatches; i++) {
      VALUE match;
      VALUE args[3];

      args[0] = INT2NUM(matches[i].first_index);
      args[1] = INT2NUM(matches[i].second_index);
      args[2] = INT2NUM(matches[i].length);

      match = rb_class_new_instance(3, args, match_class);
      rb_ary_push(list, match);
    }

    free(matches);

    return list;
}

/* ── phash256: 256-bit DCT perceptual hash (hash_size=16) ──────────────────────
 *
 * Algorithm mirrors Python imagehash.phash(img, hash_size=16):
 *   1. Convert to grayscale (ITU-R 601)
 *   2. Resize to 64×64 with Lanczos resampling (PIL ANTIALIAS-compatible)
 *   3. 2D DCT-II (matches scipy.fftpack.dct type=2, norm=None)
 *   4. Top-left 16×16 submatrix (256 coefficients)
 *   5. Compare each coefficient to the median → 256-bit hash
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PHASH256_SIZE     16
#define PHASH256_IMG_SIZE (PHASH256_SIZE * 4)            /* 64  */
#define PHASH256_BITS     (PHASH256_SIZE * PHASH256_SIZE) /* 256 */
#define PHASH256_BYTES    (PHASH256_BITS / 8)            /* 32  */

#define LANCZOS_SUPPORT_RADIUS 3.0

/* ITU-R BT.601 luma coefficients for RGB → grayscale conversion */
#define LUMA_R 0.2126
#define LUMA_G 0.7152
#define LUMA_B 0.0722

/* Maximum image dimension we're willing to process (guards against overflow) */
#define MAX_IMAGE_DIMENSION 65536

static double lanczos3_kernel(double x) {
    if (x == 0.0) return 1.0;
    if (x < 0.0) x = -x;
    if (x >= LANCZOS_SUPPORT_RADIUS) return 0.0;
    double xpi = x * M_PI;
    return LANCZOS_SUPPORT_RADIUS * sin(xpi) * sin(xpi / LANCZOS_SUPPORT_RADIUS) / (xpi * xpi);
}

/*
 * Perform one separable 1D Lanczos-3 pass (either horizontal or vertical).
 *
 * Parameters:
 *   input       - source pixel buffer
 *   output      - destination pixel buffer
 *   src_len     - source dimension along the resampling axis
 *   dst_len     - destination dimension along the resampling axis
 *   other_len   - dimension along the non-resampling axis
 *   src_stride  - distance between consecutive elements along the resampling axis in input
 *   dst_stride  - distance between consecutive elements along the resampling axis in output
 *   src_row     - distance between consecutive rows (the non-resampling axis) in input
 *   dst_row     - distance between consecutive rows (the non-resampling axis) in output
 *   clamp_output - if true, clamp and round output values to [0, 255]
 */
static void lanczos_pass(const double *input, double *output,
                          int src_len, int dst_len, int other_len,
                          int src_stride, int dst_stride,
                          int src_row, int dst_row,
                          int clamp_output) {
    double scale = (double)src_len / dst_len;
    double norm  = scale > 1.0 ? scale : 1.0;
    double support = LANCZOS_SUPPORT_RADIUS * norm;

    for (int row = 0; row < other_len; row++) {
        for (int d = 0; d < dst_len; d++) {
            double center = (d + 0.5) * scale - 0.5;
            int s0 = (int)ceil(center - support);
            int s1 = (int)floor(center + support);
            if (s0 < 0) s0 = 0;
            if (s1 >= src_len) s1 = src_len - 1;

            double weight_sum = 0.0, value = 0.0;
            for (int s = s0; s <= s1; s++) {
                double weight = lanczos3_kernel((s - center) / norm);
                value      += input[row * src_row + s * src_stride] * weight;
                weight_sum += weight;
            }

            double result = (weight_sum > 0.0) ? (value / weight_sum) : 0.0;
            if (clamp_output) {
                if (result < 0.0) result = 0.0;
                else if (result > 255.0) result = 255.0;
                result = round(result);
            }
            output[row * dst_row + d * dst_stride] = result;
        }
    }
}

/*
 * Anti-aliased Lanczos resize matching Pillow's ANTIALIAS/LANCZOS filter.
 *
 * Pillow scales the filter support by the downsampling ratio (scale > 1),
 * so for a 1080→64 resize (scale ≈ 17) the effective support radius is
 * 3 × 17 = 51 pixels. This averages out scattered pixel noise that would
 * otherwise flip hash bits. CImg's built-in flag=6 Lanczos uses a fixed
 * 3-pixel support and misses this anti-aliasing, causing near-identical
 * images to get different hashes.
 *
 * Two separable 1D passes (horizontal then vertical). Final output pixels
 * are clamped and rounded to [0, 255] to match PIL's uint8 storage.
 *
 * Returns 0 on success, -1 on allocation failure.
 */
static int lanczos_resize_pil(const double *src, int src_width, int src_height,
                               double *dst,      int dst_width, int dst_height) {
    double *tmp = (double *)malloc((size_t)dst_width * src_height * sizeof(double));
    if (!tmp) return -1;

    /* Horizontal pass: src_width×src_height → dst_width×src_height */
    lanczos_pass(src, tmp,
                 src_width, dst_width, src_height,
                 1, 1,                        /* stride along resampling axis */
                 src_width, dst_width,        /* row stride */
                 0);                          /* no clamping on intermediate */

    /* Vertical pass: dst_width×src_height → dst_width×dst_height, with clamping */
    lanczos_pass(tmp, dst,
                 src_height, dst_height, dst_width,
                 dst_width, dst_width,        /* stride along vertical axis */
                 1, 1,                        /* "row" stride = element stride */
                 1);                          /* clamp final output */

    free(tmp);
    return 0;
}

static void dct1d(double *data, int n) {
    double tmp[PHASH256_IMG_SIZE];
    const double pi_2n = M_PI / (2.0 * n);
    for (int k = 0; k < n; k++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += data[i] * cos(pi_2n * k * (2 * i + 1));
        tmp[k] = sum + sum;
    }
    memcpy(data, tmp, (size_t)n * sizeof(double));
}

static void dct2d(double *data, int n) {
    double col[PHASH256_IMG_SIZE];
    for (int r = 0; r < n; r++)
        dct1d(data + r * n, n);
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < n; r++) col[r] = data[r * n + c];
        dct1d(col, n);
        for (int r = 0; r < n; r++) data[r * n + c] = col[r];
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

struct nogvl_hash256_args {
    const char *filename;
    uint8_t     bytes[PHASH256_BYTES];
    int         retval;
};

static void *nogvl_hash256(struct nogvl_hash256_args *args) {
    try {
        CImg<float> src(args->filename);

        int src_width  = src.width();
        int src_height = src.height();

        if (src_width <= 0 || src_height <= 0 ||
            src_width > MAX_IMAGE_DIMENSION || src_height > MAX_IMAGE_DIMENSION) {
            args->retval = -1;
            return NULL;
        }

        /* Guard against size_t overflow: width * height * sizeof(double) */
        size_t pixel_count = (size_t)src_width * (size_t)src_height;
        if (pixel_count / (size_t)src_width != (size_t)src_height) {
            args->retval = -1;
            return NULL;
        }

        /* 1. Grayscale (ITU-R BT.601 luma), matching PIL's convert("L") → uint8 */
        double *gray_flat = (double *)malloc(pixel_count * sizeof(double));
        if (!gray_flat) {
            args->retval = -1;
            return NULL;
        }

        if (src.spectrum() >= 3) {
            cimg_forXY(src, x, y) {
                double v = LUMA_R * (double)src(x, y, 0, 0)
                         + LUMA_G * (double)src(x, y, 0, 1)
                         + LUMA_B * (double)src(x, y, 0, 2);
                if (v < 0.0) v = 0.0;
                else if (v > 255.0) v = 255.0;
                gray_flat[y * src_width + x] = round(v);
            }
        } else {
            cimg_forXY(src, x, y) {
                double v = (double)src(x, y, 0, 0);
                if (v < 0.0) v = 0.0;
                else if (v > 255.0) v = 255.0;
                gray_flat[y * src_width + x] = round(v);
            }
        }

        /* 2. Resize 64×64 with PIL-compatible anti-aliased Lanczos */
        double resized[PHASH256_IMG_SIZE * PHASH256_IMG_SIZE];
        if (lanczos_resize_pil(gray_flat, src_width, src_height,
                               resized, PHASH256_IMG_SIZE, PHASH256_IMG_SIZE) != 0) {
            free(gray_flat);
            args->retval = -1;
            return NULL;
        }
        free(gray_flat);

        /* 3. 2D DCT-II (matches scipy.fftpack.dct type=2 norm=None) */
        dct2d(resized, PHASH256_IMG_SIZE);

        /* 4. Top-left 16×16 submatrix */
        double low[PHASH256_BITS];
        for (int r = 0; r < PHASH256_SIZE; r++)
            for (int c = 0; c < PHASH256_SIZE; c++)
                low[r * PHASH256_SIZE + c] = resized[r * PHASH256_IMG_SIZE + c];

        /* 5. Median of 256 values (average of two middle elements) */
        double sorted[PHASH256_BITS];
        memcpy(sorted, low, sizeof(sorted));
        qsort(sorted, PHASH256_BITS, sizeof(double), cmp_double);
        double med = (sorted[PHASH256_BITS / 2 - 1] + sorted[PHASH256_BITS / 2]) / 2.0;

        /* 6. Pack bits MSB-first: bit i set iff low[i] > median */
        for (int i = 0; i < PHASH256_BYTES; i++) {
            uint8_t byte_val = 0;
            for (int bit = 0; bit < 8; bit++) {
                if (low[i * 8 + bit] > med)
                    byte_val |= (uint8_t)(1u << (7 - bit));
            }
            args->bytes[i] = byte_val;
        }

        args->retval = 0;
    } catch (...) {
        args->retval = -1;
    }
    return NULL;
}

static VALUE image_hash256_for(VALUE self, VALUE _filename) {
    struct nogvl_hash256_args args;
    args.filename = StringValuePtr(_filename);
    args.retval   = -1;

    rb_thread_call_without_gvl((void *(*)(void *))nogvl_hash256,
        (void *)&args, RUBY_UBF_PROCESS, 0);

    if (args.retval == -1)
        rb_raise(rb_eRuntimeError, "pHash256 error computing hash");

    return rb_integer_unpack(args.bytes, PHASH256_BYTES, 1, 0, INTEGER_PACK_BIG_ENDIAN);
}

static VALUE hamming_distance256(VALUE self, VALUE a, VALUE b) {
    uint8_t ha[PHASH256_BYTES] = {0}, hb[PHASH256_BYTES] = {0};
    rb_integer_pack(a, ha, PHASH256_BYTES, 1, 0, INTEGER_PACK_BIG_ENDIAN);
    rb_integer_pack(b, hb, PHASH256_BYTES, 1, 0, INTEGER_PACK_BIG_ENDIAN);

    int dist = 0;
    for (int i = 0; i < PHASH256_BYTES; i++)
        dist += __builtin_popcount((unsigned char)(ha[i] ^ hb[i]));

    return INT2NUM(dist);
}

#ifdef __cplusplus
extern "C" {
#endif
  void Init_phashion_ext() {
    VALUE c = rb_cObject;
    c = rb_const_get(c, rb_intern("Phashion"));

    rb_define_singleton_method(c, "hamming_distance", (VALUE(*)(ANYARGS))hamming_distance, 2);
    rb_define_singleton_method(c, "image_hash_for", (VALUE(*)(ANYARGS))image_hash_for, 1);
    rb_define_singleton_method(c, "image_hash256_for", (VALUE(*)(ANYARGS))image_hash256_for, 1);
    rb_define_singleton_method(c, "hamming_distance256", (VALUE(*)(ANYARGS))hamming_distance256, 2);

    rb_define_singleton_method(c, "_mh_hash_for", (VALUE(*)(ANYARGS))mh_hash_for, 3);
    rb_define_singleton_method(c, "hamming_distance2", (VALUE(*)(ANYARGS))hamming_distance2, 2);

    rb_define_singleton_method(c, "texthash_for", (VALUE(*)(ANYARGS))texthash_for, 1);
    rb_define_singleton_method(c, "textmatches_for", (VALUE(*)(ANYARGS))textmatches_for, 2);
  }

#ifdef HAVE_SQLITE3EXT_H
#include <sqlite3ext.h>

SQLITE_EXTENSION_INIT1

static void hamming_distance(sqlite3_context * ctx, int agc, sqlite3_value **argv)
{
  sqlite3_int64 hashes[4];
  ulong64 left, right;
  int i, result;

  for(i = 0; i < 4; i++) {
    if (SQLITE_INTEGER == sqlite3_value_type(argv[i])) {
      hashes[i] = sqlite3_value_int64(argv[i]);
    } else {
      hashes[i] = 0;
    }
  }

  left = (hashes[0] << 32) + hashes[1];
  right = (hashes[2] << 32) + hashes[3];
  result = ph_hamming_distance(left, right);
  sqlite3_result_int(ctx, result);
}

int sqlite3_extension_init(
  sqlite3 *db,
  char **pzErrMsg,
  const sqlite3_api_routines *pApi
){
  SQLITE_EXTENSION_INIT2(pApi);

  sqlite3_create_function(
      db,
      "hamming_distance",
      4,
      SQLITE_UTF8,
      NULL,
      hamming_distance,
      NULL,
      NULL
  );
  return SQLITE_OK;
}

#endif

#ifdef __cplusplus
}
#endif
