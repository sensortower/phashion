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

#define PHASH256_SIZE     16
#define PHASH256_IMG_SIZE (PHASH256_SIZE * 4)          /* 64  */
#define PHASH256_BITS     (PHASH256_SIZE * PHASH256_SIZE) /* 256 */
#define PHASH256_BYTES    (PHASH256_BITS / 8)          /* 32  */

/* Lanczos-3 kernel matching Pillow's filter_lanczos (support = 3.0) */
static double lanczos3_kernel(double x) {
    if (x == 0.0) return 1.0;
    if (x < 0.0) x = -x;
    if (x >= 3.0) return 0.0;
    double xpi = x * M_PI;
    return 3.0 * sin(xpi) * sin(xpi / 3.0) / (xpi * xpi);
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
 * Two separable 1D passes (horizontal then vertical). Output pixels are
 * clamped and rounded to [0, 255] to match PIL's uint8 storage.
 */
static void lanczos_resize_pil(const float *src, int sw, int sh,
                                float *dst,       int dw, int dh) {
    double scale_x = (double)sw / dw;
    double scale_y = (double)sh / dh;
    double norm_x  = scale_x > 1.0 ? scale_x : 1.0;
    double norm_y  = scale_y > 1.0 ? scale_y : 1.0;
    double supp_x  = 3.0 * norm_x;
    double supp_y  = 3.0 * norm_y;

    /* Horizontal pass: sw×sh → dw×sh */
    float *tmp = (float *)malloc((size_t)dw * sh * sizeof(float));
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < dw; x++) {
            double center = (x + 0.5) * scale_x - 0.5;
            int x0 = (int)ceil(center - supp_x);
            int x1 = (int)floor(center + supp_x);
            if (x0 < 0)   x0 = 0;
            if (x1 >= sw) x1 = sw - 1;
            double wsum = 0.0, val = 0.0;
            for (int sx = x0; sx <= x1; sx++) {
                double w = lanczos3_kernel((sx - center) / norm_x);
                val  += src[y * sw + sx] * w;
                wsum += w;
            }
            tmp[y * dw + x] = (wsum > 0.0) ? (float)(val / wsum) : 0.0f;
        }
    }

    /* Vertical pass: dw×sh → dw×dh, clamp+round to uint8 */
    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            double center = (y + 0.5) * scale_y - 0.5;
            int y0 = (int)ceil(center - supp_y);
            int y1 = (int)floor(center + supp_y);
            if (y0 < 0)   y0 = 0;
            if (y1 >= sh) y1 = sh - 1;
            double wsum = 0.0, val = 0.0;
            for (int sy = y0; sy <= y1; sy++) {
                double w = lanczos3_kernel((sy - center) / norm_y);
                val  += tmp[sy * dw + x] * w;
                wsum += w;
            }
            float v = (wsum > 0.0) ? (float)(val / wsum) : 0.0f;
            if (v < 0.0f)   v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            dst[y * dw + x] = roundf(v);
        }
    }
    free(tmp);
}

static void dct1d(double *data, int n) {
    double tmp[PHASH256_IMG_SIZE];
    const double pi_2n = M_PI / (2.0 * n);
    for (int k = 0; k < n; k++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += data[i] * cos(pi_2n * k * (2*i + 1));
        tmp[k] = 2.0 * sum;
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

        /* 1. Grayscale (ITU-R 601), matching PIL's convert("L") → uint8 */
        int sw = src.width(), sh = src.height();
        float *gray_flat = (float *)malloc((size_t)sw * sh * sizeof(float));
        if (src.spectrum() >= 3) {
            cimg_forXY(src, x, y) {
                float v = 0.299f * src(x, y, 0, 0)
                        + 0.587f * src(x, y, 0, 1)
                        + 0.114f * src(x, y, 0, 2);
                if (v < 0.0f) v = 0.0f;
                else if (v > 255.0f) v = 255.0f;
                gray_flat[y * sw + x] = std::round(v);
            }
        } else {
            cimg_forXY(src, x, y)
                gray_flat[y * sw + x] = src(x, y, 0, 0);
        }

        /* 2. Resize 64×64 with PIL-compatible anti-aliased Lanczos */
        float resized[PHASH256_IMG_SIZE * PHASH256_IMG_SIZE];
        lanczos_resize_pil(gray_flat, sw, sh,
                           resized, PHASH256_IMG_SIZE, PHASH256_IMG_SIZE);
        free(gray_flat);

        /* 3. Pixels → row-major double array */
        double pixels[PHASH256_IMG_SIZE * PHASH256_IMG_SIZE];
        for (int i = 0; i < PHASH256_IMG_SIZE * PHASH256_IMG_SIZE; i++)
            pixels[i] = (double)resized[i];

        /* 4. 2D DCT-II (matches scipy.fftpack.dct type=2 norm=None, axis=0 then axis=1) */
        dct2d(pixels, PHASH256_IMG_SIZE);

        /* 5. Top-left 16×16 submatrix */
        double low[PHASH256_BITS];
        for (int r = 0; r < PHASH256_SIZE; r++)
            for (int c = 0; c < PHASH256_SIZE; c++)
                low[r * PHASH256_SIZE + c] = pixels[r * PHASH256_IMG_SIZE + c];

        /* 6. Median of 256 values (average of two middle elements) */
        double sorted[PHASH256_BITS];
        memcpy(sorted, low, sizeof(sorted));
        qsort(sorted, PHASH256_BITS, sizeof(double), cmp_double);
        double med = (sorted[PHASH256_BITS/2 - 1] + sorted[PHASH256_BITS/2]) / 2.0;

        /* 7. Pack bits MSB-first: bit i set iff low[i] > median */
        memset(args->bytes, 0, PHASH256_BYTES);
        for (int i = 0; i < PHASH256_BITS; i++)
            if (low[i] > med)
                args->bytes[i / 8] |= (uint8_t)(1u << (7 - (i % 8)));

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
