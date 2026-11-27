// Zero-libc Linux x86_64 N-API addon for exact makefun-diff 1.0.0 (v7.7).
// Architecture:
// - Single-pass find_quote on `old` ONLY (zero pre-scan of `new` -> 1x DRAM stream!)
// - Single-pass fused 64B/iter AVX2 lockstep Block-Sync + 1-mul Fibonacci hash
// - Packed 2-bit L2-hot uniqueness bitset (`seen`, BF_SHIFT=18) + short boilerplate hash skip
// - Proven-safe inline 1-line 16B SIMD fast-resync on open-ended `new` (~60% of gaps, zero stack frame!)
// - Compact noinline 2-tier resync fallback (batched peek, incremental Tier 1.5, O(1) hash Tier 2)
// - Exact O(N) Kadane-Phi bridge decomposition + O(1) run-tail skip + expanded safe unique-suffix trim
// - Exact O(n)/O(m) solver for BOTH disjoint AND matched 1xM / Nx1 subproblems!
// - Compact Tiny-Myers (n,m <= 12) + old-only HT Big Myers with O(1) disjoint skip + AVX2 16-wide snake
// - Zero-copy MAP_SHARED|MAP_POPULATE input + tight single-mmap arena + scratch-aliased output
// - Compact ~40 KB stripped ELF + direct init_module execution + SYS_exit_group(0)

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <immintrin.h>
#ifdef STANDALONE_TEST
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
static inline double _t_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
static uint64_t cyc_run = 0, cyc_fastgap = 0, cyc_gap = 0, cyc_sub_tiny = 0, cyc_sub_big = 0;
static uint64_t cnt_fg = 0, cnt_sg = 0, cnt_big = 0, cnt_1x_match = 0;
#endif

#define COLD_OPT __attribute__((noinline, cold, optimize("Os")))
#define SSE2_OPT __attribute__((noinline, optimize("Os")))

#ifndef BF_SHIFT
#define BF_SHIFT 18u
#endif
#ifndef SHORT_HASH_MIN
#define SHORT_HASH_MIN 8u
#endif
#ifndef ARENA_POPULATE
#define ARENA_POPULATE 1
#endif

#if BF_SHIFT <= 16
typedef uint16_t oh_t;
#else
typedef uint32_t oh_t;
#endif

// ---------------------------------------------------------------------------
// Minimal N-API ABI (4 symbols only, resolved from host `node`)
// ---------------------------------------------------------------------------

typedef struct napi_env__* napi_env;
typedef struct napi_value__* napi_value;
typedef int napi_status;

typedef struct {
    int nm_version;
    unsigned int nm_flags;
    const char* nm_filename;
    napi_value (*nm_register_func)(napi_env env, napi_value exports);
    const char* nm_modname;
    void* nm_priv;
    void* reserved[4];
} napi_module;

#ifndef STANDALONE_TEST
extern void napi_module_register(napi_module* mod);
extern napi_status napi_has_named_property(napi_env env, napi_value object, const char* utf8name, bool* result);
extern napi_status napi_get_named_property(napi_env env, napi_value object, const char* utf8name, napi_value* result);
extern napi_status napi_get_value_string_utf8(napi_env env, napi_value value, char* buf, size_t bufsize, size_t* result);
#endif

// ---------------------------------------------------------------------------
// Direct Linux x86_64 syscalls (zero libc!)
// ---------------------------------------------------------------------------

#define SYS_read       0
#define SYS_write      1
#define SYS_close      3
#define SYS_lseek      8
#define SYS_mmap       9
#define SYS_munmap     11
#define SYS_pread64    17
#define SYS_fchmod     91
#define SYS_exit_group 231
#define SYS_openat     257

#define AT_FDCWD     (-100)
#define O_RDONLY     0
#define O_WRONLY     1
#define O_CREAT      0100
#define O_TRUNC      01000
#define SEEK_END     2
#define PROT_READ    1
#define PROT_WRITE   2
#define MAP_SHARED   0x01
#define MAP_PRIVATE  0x02
#define MAP_ANON     0x20
#define MAP_POPULATE 0x08000

static inline long sys1(long n, long a1) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1) : "rcx", "r11", "memory");
    return ret;
}

static inline long sys2(long n, long a1, long a2) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2) : "rcx", "r11", "memory");
    return ret;
}

static inline long sys3(long n, long a1, long a2, long a3) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2), "d"(a3) : "rcx", "r11", "memory");
    return ret;
}

static inline long sys4(long n, long a1, long a2, long a3, long a4) {
    long ret;
    register long r10 __asm__("r10") = a4;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10) : "rcx", "r11", "memory");
    return ret;
}

static inline long sys6(long n, long a1, long a2, long a3, long a4, long a5, long a6) {
    long ret;
    register long r10 __asm__("r10") = a4;
    register long r8  __asm__("r8")  = a5;
    register long r9  __asm__("r9")  = a6;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
    return ret;
}

static inline void* os_mmap(size_t len) {
#if ARENA_POPULATE
    long p = sys6(SYS_mmap, 0, (long)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_POPULATE, -1, 0);
    if (p < 0 && p > -4096) p = sys6(SYS_mmap, 0, (long)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
    long p = sys6(SYS_mmap, 0, (long)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
    if (p < 0 && p > -4096) return NULL;
    return (void*)p;
}

static inline void os_munmap(void* p, size_t len) {
    if (p) sys2(SYS_munmap, (long)p, (long)len);
}

// ---------------------------------------------------------------------------
// Runtime CPUID + XGETBV check for AVX2 + BMI1
// ---------------------------------------------------------------------------

static int cpu_has_avx2(void) {
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0), "c"(0));
    if (eax < 7) return 0;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
    if (((ecx & (1u << 27)) == 0) || ((ecx & (1u << 28)) == 0)) return 0;
    uint32_t xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    if ((xlo & 6u) != 6u) return 0;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    return ((ebx & (1u << 5)) != 0) && ((ebx & (1u << 3)) != 0);
}

// ---------------------------------------------------------------------------
// Low-level memory / SIMD helpers (zero libc)
// ---------------------------------------------------------------------------

static inline uint64_t read_u64(const uint8_t* p) { uint64_t v; __builtin_memcpy(&v, p, 8); return v; }
static inline uint32_t read_u32(const uint8_t* p) { uint32_t v; __builtin_memcpy(&v, p, 4); return v; }
static inline uint16_t read_u16(const uint8_t* p) { uint16_t v; __builtin_memcpy(&v, p, 2); return v; }

static inline void mem_zero_sse2(void* dst, size_t n) {
    uint8_t* p = (uint8_t*)dst;
    __m128i z = _mm_setzero_si128();
    while (n >= 64) {
        _mm_storeu_si128((__m128i*)(p), z);
        _mm_storeu_si128((__m128i*)(p + 16), z);
        _mm_storeu_si128((__m128i*)(p + 32), z);
        _mm_storeu_si128((__m128i*)(p + 48), z);
        p += 64; n -= 64;
    }
    while (n >= 16) { _mm_storeu_si128((__m128i*)p, z); p += 16; n -= 16; }
    while (n--) *p++ = 0;
}

__attribute__((target("avx2")))
static inline void mem_zero_avx2(void* dst, size_t n) {
    uint8_t* p = (uint8_t*)dst;
    __m256i z = _mm256_setzero_si256();
    while (n >= 128) {
        _mm256_storeu_si256((__m256i*)(p), z);
        _mm256_storeu_si256((__m256i*)(p + 32), z);
        _mm256_storeu_si256((__m256i*)(p + 64), z);
        _mm256_storeu_si256((__m256i*)(p + 96), z);
        p += 128; n -= 128;
    }
    while (n >= 32) { _mm256_storeu_si256((__m256i*)p, z); p += 32; n -= 32; }
    while (n--) *p++ = 0;
}

static inline void mem_copy_bytes(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n >= 32) {
        _mm_storeu_si128((__m128i*)d, _mm_loadu_si128((const __m128i*)s));
        _mm_storeu_si128((__m128i*)(d + 16), _mm_loadu_si128((const __m128i*)(s + 16)));
        d += 32; s += 32; n -= 32;
    }
    while (n >= 8) { uint64_t v = read_u64(s); __builtin_memcpy(d, &v, 8); d += 8; s += 8; n -= 8; }
    while (n--) *d++ = *s++;
}

static inline uint8_t* copy_short_sse2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    if (__builtin_expect(len >= 16, 1)) {
        if (len <= 32) {
            _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
            _mm_storeu_si128((__m128i*)(end - 16), _mm_loadu_si128((const __m128i*)(src + len - 16)));
            return end;
        }
        size_t i = 0;
        for (; i + 32 <= len; i += 32) {
            _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((const __m128i*)(src + i)));
            _mm_storeu_si128((__m128i*)(dst + i + 16), _mm_loadu_si128((const __m128i*)(src + i + 16)));
        }
        if (i + 16 <= len) _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((const __m128i*)(src + i)));
        _mm_storeu_si128((__m128i*)(end - 16), _mm_loadu_si128((const __m128i*)(src + len - 16)));
        return end;
    }
    if (len >= 8) {
        uint64_t v0 = read_u64(src), v1 = read_u64(src + len - 8);
        __builtin_memcpy(dst, &v0, 8); __builtin_memcpy(end - 8, &v1, 8);
        return end;
    }
    if (len >= 4) {
        uint32_t v0 = read_u32(src), v1 = read_u32(src + len - 4);
        __builtin_memcpy(dst, &v0, 4); __builtin_memcpy(end - 4, &v1, 4);
        return end;
    }
    for (size_t i = 0; i < len; i++) dst[i] = src[i];
    return end;
}

__attribute__((target("avx2"), always_inline))
static inline uint8_t* copy_line_over_avx2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
    if (__builtin_expect(len <= 32, 1)) return end;
    _mm256_storeu_si256((__m256i*)(dst + 32), _mm256_loadu_si256((const __m256i*)(src + 32)));
    if (__builtin_expect(len <= 64, 1)) return end;
    size_t i = 64;
    for (; i + 32 <= len; i += 32) {
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((const __m256i*)(src + i)));
    }
    if (i < len) _mm256_storeu_si256((__m256i*)(end - 32), _mm256_loadu_si256((const __m256i*)(src + len - 32)));
    return end;
}

__attribute__((target("avx2"), always_inline))
static inline uint8_t* copy_short_avx2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    if (__builtin_expect(len >= 32, 1)) {
        if (len <= 64) {
            _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
            _mm256_storeu_si256((__m256i*)(end - 32), _mm256_loadu_si256((const __m256i*)(src + len - 32)));
            return end;
        }
        size_t i = 0;
        for (; i + 64 <= len; i += 64) {
            _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((const __m256i*)(src + i)));
            _mm256_storeu_si256((__m256i*)(dst + i + 32), _mm256_loadu_si256((const __m256i*)(src + i + 32)));
        }
        if (i + 32 <= len) _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((const __m256i*)(src + i)));
        _mm256_storeu_si256((__m256i*)(end - 32), _mm256_loadu_si256((const __m256i*)(src + len - 32)));
        return end;
    }
    return copy_short_sse2(dst, src, len);
}

static inline int eq_bytes_sse2(const uint8_t* a, const uint8_t* b, size_t len) {
    if (len <= 16) {
        if (len >= 8) return ((read_u64(a) ^ read_u64(b)) | (read_u64(a + len - 8) ^ read_u64(b + len - 8))) == 0;
        if (len >= 4) return ((read_u32(a) ^ read_u32(b)) | (read_u32(a + len - 4) ^ read_u32(b + len - 4))) == 0;
        if (len >= 2) return ((read_u16(a) ^ read_u16(b)) | (a[len - 1] ^ b[len - 1])) == 0;
        return len == 0 || a[0] == b[0];
    }
    if (len <= 32) {
        __m128i x0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b));
        __m128i x1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(a + len - 16)), _mm_loadu_si128((const __m128i*)(b + len - 16)));
        return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_or_si128(x0, x1), _mm_setzero_si128())) == 0xFFFF;
    }
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(b + i)))) != 0xFFFF) return 0;
    }
    return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)(a + len - 16)), _mm_loadu_si128((const __m128i*)(b + len - 16)))) == 0xFFFF;
}

__attribute__((target("avx2")))
static int eq_bytes_avx2(const uint8_t* a, const uint8_t* b, size_t len) {
    if (len <= 32) {
        if (len >= 16) {
            __m128i x0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b));
            __m128i x1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(a + len - 16)), _mm_loadu_si128((const __m128i*)(b + len - 16)));
            __m128i o = _mm_or_si128(x0, x1);
            return _mm_testz_si128(o, o);
        }
        if (len >= 8) return ((read_u64(a) ^ read_u64(b)) | (read_u64(a + len - 8) ^ read_u64(b + len - 8))) == 0;
        if (len >= 4) return ((read_u32(a) ^ read_u32(b)) | (read_u32(a + len - 4) ^ read_u32(b + len - 4))) == 0;
        if (len >= 2) return ((read_u16(a) ^ read_u16(b)) | (a[len - 1] ^ b[len - 1])) == 0;
        return len == 0 || a[0] == b[0];
    }
    if (len <= 64) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)a), _mm256_loadu_si256((const __m256i*)b));
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + len - 32)), _mm256_loadu_si256((const __m256i*)(b + len - 32)));
        __m256i o = _mm256_or_si256(x0, x1);
        return _mm256_testz_si256(o, o);
    }
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i x = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + i)), _mm256_loadu_si256((const __m256i*)(b + i)));
        if (!_mm256_testz_si256(x, x)) return 0;
    }
    __m256i xt = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + len - 32)), _mm256_loadu_si256((const __m256i*)(b + len - 32)));
    return _mm256_testz_si256(xt, xt);
}

// ---------------------------------------------------------------------------
// Ultra-fast 1-multiply Fibonacci O(1) hash + packed 2-bit uniqueness filter
// ---------------------------------------------------------------------------

#define K1 0x9E3779B97F4A7C15ULL
#define K2 0xBF58476D1CE4E5B9ULL

#define BF_BUCKETS (1u << BF_SHIFT)
#define BF_MASK    (BF_BUCKETS - 1u)
#define BF_WORDS   (BF_BUCKETS >> 5)

static inline uint32_t hash_o1_ge8(const uint8_t* p, size_t len) {
    uint64_t a = read_u64(p);
    uint64_t b = read_u64(p + len - 8);
    uint64_t h = (a ^ ((b << 29) | (b >> 35)) ^ (uint64_t)len) * K1;
    return (uint32_t)(h >> (64u - BF_SHIFT));
}

static inline void mark_sync_2b(uint64_t* seen, uint32_t h) {
    uint32_t w = h >> 5;
    uint32_t sh = (h & 31u) << 1;
    uint64_t v = seen[w];
    uint64_t b1 = 1ULL << sh;
    seen[w] = v | b1 | ((v & b1) << 1);
}

static inline void mark_gap_2b(uint64_t* seen, uint32_t h) {
    seen[h >> 5] |= (3ULL << ((h & 31u) << 1));
}

static inline uint32_t is_uniq_2b(const uint64_t* seen, uint32_t h) {
    return (uint32_t)(((seen[h >> 5] >> ((h & 31u) << 1)) & 2u) == 0);
}

static inline oh_t record_sync_line(uint64_t* seen, const uint8_t* p, size_t len) {
    if (__builtin_expect(len >= SHORT_HASH_MIN, 1)) {
        uint32_t h = hash_o1_ge8(p, len);
        mark_sync_2b(seen, h);
        return (oh_t)h;
    }
    return (oh_t)0; // bucket 0 is pre-marked 3 (non-unique) -> zero hash, zero RMW!
}

static inline oh_t record_gap_old(uint64_t* seen, const uint8_t* p, size_t len) {
    if (__builtin_expect(len >= SHORT_HASH_MIN, 1)) {
        uint32_t h = hash_o1_ge8(p, len);
        mark_gap_2b(seen, h);
        return (oh_t)h;
    }
    return (oh_t)0;
}

static inline void record_gap_new(uint64_t* seen, const uint8_t* p, size_t len) {
    if (__builtin_expect(len >= SHORT_HASH_MIN, 1)) mark_gap_2b(seen, hash_o1_ge8(p, len));
}

static inline uint64_t hash_64(const uint8_t* p, size_t len) {
    if (__builtin_expect(len >= 16, 1)) {
        uint64_t a = read_u64(p);
        uint64_t b = read_u64(p + (len >> 2));
        uint64_t c = read_u64(p + (len >> 1) - 4);
        uint64_t d = read_u64(p + len - 8);
        uint64_t h = ((a ^ (len * K2)) * K1 ^ b) * K2 ^ (c * K1) ^ d;
        if (__builtin_expect(len > 128, 0)) {
            for (size_t off = 16; off + 16 <= len - 16; off += 16) {
                h = (h ^ read_u64(p + off)) * K1 ^ read_u64(p + off + 8);
            }
        }
        return h ^ (h >> 32);
    }
    if (len >= 4) {
        uint64_t a = read_u32(p);
        uint64_t b = read_u32(p + len - 4);
        uint64_t c = len >= 8 ? (uint64_t)read_u32(p + 4) : 0;
        uint64_t h = ((a | (b << 32)) ^ (c * K2) ^ len) * K1;
        return h ^ (h >> 32);
    }
    uint64_t v = len ? ((uint64_t)p[0] | ((uint64_t)p[len - 1] << 8)) : 0;
    uint64_t h = (v ^ (len << 16)) * K1;
    return h ^ (h >> 32);
}

// ---------------------------------------------------------------------------
// Compact single-mmap arena
// ---------------------------------------------------------------------------

#define MAX_ARENA_BLOCKS 16

typedef struct {
    void* blocks[MAX_ARENA_BLOCKS];
    size_t sizes[MAX_ARENA_BLOCKS];
    int n_blocks;
    uint8_t* cur;
    size_t used;
    size_t cap;
} Arena;

static int arena_init(Arena* a, size_t cap) {
    cap = (cap + 4095) & ~(size_t)4095;
    uint8_t* p = (uint8_t*)os_mmap(cap);
    if (!p) return -1;
    a->blocks[0] = p; a->sizes[0] = cap; a->n_blocks = 1;
    a->cur = p; a->used = 0; a->cap = cap;
    return 0;
}

static void arena_free(Arena* a) {
    for (int i = 0; i < a->n_blocks; i++) os_munmap(a->blocks[i], a->sizes[i]);
    a->n_blocks = 0;
}

static void* arena_alloc(Arena* a, size_t sz, size_t align) {
    size_t off = (a->used + align - 1) & ~(align - 1);
    if (off + sz <= a->cap) { void* p = a->cur + off; a->used = off + sz; return p; }
    if (a->n_blocks >= MAX_ARENA_BLOCKS) return NULL;
    size_t ncap = a->cap * 2;
    if (ncap < sz + align) ncap = sz + align;
    ncap = (ncap + 4095) & ~(size_t)4095;
    uint8_t* nb = (uint8_t*)os_mmap(ncap);
    if (!nb) return NULL;
    a->blocks[a->n_blocks] = nb; a->sizes[a->n_blocks] = ncap; a->n_blocks++;
    a->cur = nb; a->used = sz; a->cap = ncap;
    return nb;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

#define DEAD (-2)
#define MAX_SUB_LINES 65532u

typedef struct { uint32_t o, p, dels, ins; } Chg;
typedef struct { uint32_t a0, b0, len; } SyncRun;
typedef struct { uint32_t preA, preB, preUV, sufA, sufB, sufUV, uniq; } RunPhi;

typedef struct {
    Arena arena;

    uint32_t* old_off; size_t old_off_cap; size_t a_len;
    uint32_t* new_off; size_t new_off_cap; size_t b_len;
    Chg* chg;          size_t chg_len;     size_t chg_cap;
    int32_t* v;        size_t v_half;

    uint8_t* scratch_base;
    size_t scratch_cap;

    oh_t* old_h;       size_t old_h_cap;
    SyncRun* runs;     size_t runs_len;    size_t runs_cap;
    RunPhi* rphi;      size_t rphi_cap;
    uint64_t* seen;

    uint32_t* ht;      size_t ht_cap;
    uint32_t* ent_off; size_t ent_cap;
    uint32_t* ent_len;
    uint16_t* a_ids;   size_t a_ids_cap;
    uint16_t* b_ids;   size_t b_ids_cap;
    uint16_t* last_a;  size_t last_cap;
    uint16_t* last_b;
    int32_t* tr_x;     size_t tr_cap;
    uint8_t* tr_c;
    uint32_t* row_base; size_t row_cap;
    int32_t* row_lo;
    uint32_t* seg_ops; size_t seg_cap;

    uint8_t* out;      size_t out_len;     size_t out_cap;
    uint8_t* out_overflow; size_t out_overflow_cap;
} WS;

COLD_OPT
static int ensure_u32(Arena* a, uint32_t** ptr, size_t* cap, size_t used, size_t need) {
    if (need <= *cap) return 0;
    size_t nc = *cap ? *cap * 2 : ((need + 63) & ~(size_t)63);
    while (nc < need) nc *= 2;
    uint32_t* np = (uint32_t*)arena_alloc(a, nc * 4, 32);
    if (!np) return -1;
    if (used > 0) mem_copy_bytes(np, *ptr, used * 4);
    *ptr = np; *cap = nc;
    return 0;
}

COLD_OPT
static int ensure_u16(Arena* a, uint16_t** ptr, size_t* cap, size_t used, size_t need) {
    if (need <= *cap) return 0;
    size_t nc = *cap ? *cap * 2 : ((need + 63) & ~(size_t)63);
    while (nc < need) nc *= 2;
    uint16_t* np = (uint16_t*)arena_alloc(a, nc * 2, 32);
    if (!np) return -1;
    if (used > 0) mem_copy_bytes(np, *ptr, used * 2);
    *ptr = np; *cap = nc;
    return 0;
}

static inline int ensure_oh(Arena* a, oh_t** ptr, size_t* cap, size_t used, size_t need) {
#if BF_SHIFT <= 16
    return ensure_u16(a, ptr, cap, used, need);
#else
    return ensure_u32(a, ptr, cap, used, need);
#endif
}

COLD_OPT
static int ensure_runs(WS* ws, size_t need) {
    if (need <= ws->runs_cap) return 0;
    size_t nc = ws->runs_cap ? ws->runs_cap * 2 : 4096;
    while (nc < need) nc *= 2;
    SyncRun* np = (SyncRun*)arena_alloc(&ws->arena, nc * sizeof(SyncRun), 32);
    if (!np) return -1;
    if (ws->runs_len > 0) mem_copy_bytes(np, ws->runs, ws->runs_len * sizeof(SyncRun));
    ws->runs = np; ws->runs_cap = nc;
    return 0;
}

COLD_OPT
static int ensure_rphi(WS* ws, size_t need) {
    if (need <= ws->rphi_cap) return 0;
    size_t nc = ws->rphi_cap ? ws->rphi_cap * 2 : 4096;
    while (nc < need) nc *= 2;
    RunPhi* np = (RunPhi*)arena_alloc(&ws->arena, nc * sizeof(RunPhi), 32);
    if (!np) return -1;
    ws->rphi = np; ws->rphi_cap = nc;
    return 0;
}

COLD_OPT
static int ensure_chg(WS* ws, size_t need) {
    if (need <= ws->chg_cap) return 0;
    size_t nc = ws->chg_cap ? ws->chg_cap * 2 : 4096;
    while (nc < need) nc *= 2;
    Chg* np = (Chg*)arena_alloc(&ws->arena, nc * sizeof(Chg), 32);
    if (!np) return -1;
    if (ws->chg_len > 0) mem_copy_bytes(np, ws->chg, ws->chg_len * sizeof(Chg));
    ws->chg = np; ws->chg_cap = nc;
    return 0;
}

COLD_OPT
static int ensure_out_grow(WS* ws, size_t need) {
    if (ws->out_overflow && need <= ws->out_overflow_cap) {
        if (ws->out != ws->out_overflow) {
            if (ws->out_len > 0) mem_copy_bytes(ws->out_overflow, ws->out, ws->out_len);
            ws->out = ws->out_overflow; ws->out_cap = ws->out_overflow_cap;
        }
        return 0;
    }
    size_t nc = ws->out_cap ? ws->out_cap * 2 : ((need + 4095) & ~(size_t)4095);
    while (nc < need) nc *= 2;
    uint8_t* np = (uint8_t*)arena_alloc(&ws->arena, nc, 32);
    if (!np) return -1;
    if (ws->out_len > 0) mem_copy_bytes(np, ws->out, ws->out_len);
    ws->out = np; ws->out_cap = nc;
    ws->out_overflow = np; ws->out_overflow_cap = nc;
    return 0;
}

static inline int ensure_out(WS* ws, size_t need) {
    if (__builtin_expect(need <= ws->out_cap, 1)) return 0;
    return ensure_out_grow(ws, need);
}

COLD_OPT
static int ensure_v(WS* ws, size_t half_need) {
    if (half_need <= ws->v_half) return 0;
    size_t nh = ws->v_half ? ws->v_half * 2 : 1024;
    while (nh < half_need) nh *= 2;
    size_t nlen = nh * 2 + 4;
    int32_t* np = (int32_t*)arena_alloc(&ws->arena, nlen * 4, 32);
    if (!np) return -1;
    for (size_t i = 0; i < nlen; i++) np[i] = DEAD;
    if (ws->v_half > 0) {
        size_t old_len = ws->v_half * 2 + 4;
        mem_copy_bytes(np + (nh - ws->v_half), ws->v, old_len * 4);
    }
    ws->v = np; ws->v_half = nh;
    return 0;
}

COLD_OPT
static int ensure_trace(WS* ws, size_t tr_used, size_t tr_need, size_t row_used, size_t row_need) {
    if (tr_need > ws->tr_cap) {
        size_t nc = ws->tr_cap ? ws->tr_cap * 2 : 4096;
        while (nc < tr_need) nc *= 2;
        int32_t* nx = (int32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        uint8_t* nch = (uint8_t*)arena_alloc(&ws->arena, nc, 32);
        if (!nx || !nch) return -1;
        if (tr_used > 0) { mem_copy_bytes(nx, ws->tr_x, tr_used * 4); mem_copy_bytes(nch, ws->tr_c, tr_used); }
        ws->tr_x = nx; ws->tr_c = nch; ws->tr_cap = nc;
    }
    if (row_need > ws->row_cap) {
        size_t nc = ws->row_cap ? ws->row_cap * 2 : 1024;
        while (nc < row_need) nc *= 2;
        uint32_t* nb = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        int32_t* nl = (int32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        if (!nb || !nl) return -1;
        if (row_used > 0) { mem_copy_bytes(nb, ws->row_base, row_used * 4); mem_copy_bytes(nl, ws->row_lo, row_used * 4); }
        ws->row_base = nb; ws->row_lo = nl; ws->row_cap = nc;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Ultra-fast JSON string closing-quote locator (64B/iter AVX2 & compact SSE2)
// ---------------------------------------------------------------------------

static inline size_t skip_ws(const uint8_t* base, size_t p, size_t len) {
    while (p < len && (base[p] == ' ' || base[p] == '\n' || base[p] == '\r' || base[p] == '\t')) p++;
    return p;
}

__attribute__((target("avx2,bmi"), noinline))
static size_t find_quote_end_avx2(const uint8_t* base, size_t start, size_t len) {
    __m256i v_qt = _mm256_set1_epi8('"');
    size_t p = start;
    while (p + 64 <= len) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(base + p));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(base + p + 32));
        uint32_t m0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt));
        uint32_t m1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt));
        if ((m0 | m1) == 0) { p += 64; continue; }
        size_t hit = m0 ? (p + (size_t)__builtin_ctz(m0)) : (p + 32 + (size_t)__builtin_ctz(m1));
        size_t bs = 0;
        while (hit > start + bs && base[hit - 1 - bs] == '\\') bs++;
        if ((bs & 1) == 0) return hit;
        p = hit + 1;
    }
    while (p < len) {
        if (base[p] == '"') {
            size_t bs = 0;
            while (p > start + bs && base[p - 1 - bs] == '\\') bs++;
            if ((bs & 1) == 0) return p;
        }
        p++;
    }
    return (size_t)-1;
}

SSE2_OPT
static size_t find_quote_end_sse2(const uint8_t* base, size_t start, size_t len) {
    __m128i v_qt = _mm_set1_epi8('"');
    size_t p = start;
    while (p + 16 <= len) {
        uint32_t m = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)(base + p)), v_qt));
        if (m == 0) { p += 16; continue; }
        size_t hit = p + (size_t)__builtin_ctz(m);
        size_t bs = 0;
        while (hit > start + bs && base[hit - 1 - bs] == '\\') bs++;
        if ((bs & 1) == 0) return hit;
        p = hit + 1;
    }
    while (p < len) {
        if (base[p] == '"') {
            size_t bs = 0;
            while (p > start + bs && base[p - 1 - bs] == '\\') bs++;
            if ((bs & 1) == 0) return p;
        }
        p++;
    }
    return (size_t)-1;
}

COLD_OPT
static size_t skip_json_value(const uint8_t* base, size_t p, size_t len, int use_avx2) {
    p = skip_ws(base, p, len);
    if (p >= len) return (size_t)-1;
    uint8_t c = base[p];
    if (c == '"') {
        size_t qe = use_avx2 ? find_quote_end_avx2(base, p + 1, len) : find_quote_end_sse2(base, p + 1, len);
        return qe == (size_t)-1 ? (size_t)-1 : qe + 1;
    }
    if (c == '{') {
        p = skip_ws(base, p + 1, len);
        if (p < len && base[p] == '}') return p + 1;
        for (;;) {
            p = skip_ws(base, p, len);
            if (p >= len || base[p] != '"') return (size_t)-1;
            size_t qe = use_avx2 ? find_quote_end_avx2(base, p + 1, len) : find_quote_end_sse2(base, p + 1, len);
            if (qe == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, qe + 1, len);
            if (p >= len || base[p] != ':') return (size_t)-1;
            p = skip_json_value(base, p + 1, len, use_avx2);
            if (p == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, p, len);
            if (p >= len) return (size_t)-1;
            if (base[p] == ',') { p++; continue; }
            if (base[p] == '}') return p + 1;
            return (size_t)-1;
        }
    }
    if (c == '[') {
        p = skip_ws(base, p + 1, len);
        if (p < len && base[p] == ']') return p + 1;
        for (;;) {
            p = skip_json_value(base, p, len, use_avx2);
            if (p == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, p, len);
            if (p >= len) return (size_t)-1;
            if (base[p] == ',') { p++; continue; }
            if (base[p] == ']') return p + 1;
            return (size_t)-1;
        }
    }
    while (p < len && base[p] != ',' && base[p] != '}' && base[p] != ']' &&
           base[p] != ' ' && base[p] != '\n' && base[p] != '\r' && base[p] != '\t') p++;
    return p;
}

// ---------------------------------------------------------------------------
// Line Scanners (bounded `old` & open-ended `new`) + Compact 2-Tier Resync
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t s, len, f4;
    uint8_t bare;
} FastLine;

#define PEEK_MAX 36

static inline uint32_t line_f4(const uint8_t* p, uint32_t len) {
    return len >= 4 ? read_u32(p) : (len >= 2 ? (uint32_t)read_u16(p) : (len ? (uint32_t)p[0] : 0u));
}

__attribute__((target("avx2,bmi")))
static inline size_t scan_line_end_bounded_avx2(const uint8_t* base, size_t start, size_t str_e, uint8_t* bare_out) {
    __m256i v_bs = _mm256_set1_epi8('\\');
    size_t p = start;
    for (;;) {
        size_t hit;
        if (__builtin_expect(p + 32 <= str_e, 1)) {
            uint32_t m = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(base + p)), v_bs));
            if (m == 0) { p += 32; continue; }
            hit = p + (size_t)__builtin_ctz(m);
        } else {
            hit = p;
            while (hit < str_e && base[hit] != '\\') hit++;
            if (hit >= str_e) { *bare_out = 1; return str_e; }
        }
        if (__builtin_expect(hit + 1 >= str_e, 0)) return (size_t)-1;
        uint8_t c = base[hit + 1];
        if (__builtin_expect(c == 'n', 1)) { *bare_out = 0; return hit + 2; }
        if (c == 'r') {
            if (hit + 3 < str_e && base[hit + 2] == '\\' && base[hit + 3] == 'n') { *bare_out = 0; return hit + 4; }
            p = hit + 2;
        } else if (c == '"' || c == '\\' || c == 't' || c == 'b' || c == 'f') {
            p = hit + 2;
        } else return (size_t)-1;
    }
}

// Open-ended `new` line end scanner: returns end offset, sets *new_e_io if closing quote reached,
// returns start (with *new_e_io = start) if already at closing quote, or (size_t)-1 on invalid escape.
__attribute__((target("avx2,bmi")))
static inline size_t scan_line_end_open_avx2(const uint8_t* base, size_t start, size_t file_len, size_t* new_e_io, uint8_t* bare_out) {
    if (*new_e_io != (size_t)-1) return scan_line_end_bounded_avx2(base, start, *new_e_io, bare_out);
    if (__builtin_expect(start >= file_len, 0)) return (size_t)-1;
    if (__builtin_expect(base[start] == '"', 0)) { *new_e_io = start; *bare_out = 0; return start; }
    __m256i v_bs = _mm256_set1_epi8('\\');
    __m256i v_qt = _mm256_set1_epi8('"');
    size_t p = start;
    for (;;) {
        size_t hit;
        if (__builtin_expect(p + 32 <= file_len, 1)) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(base + p));
            uint32_t m = (uint32_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v, v_bs), _mm256_cmpeq_epi8(v, v_qt)));
            if (m == 0) { p += 32; continue; }
            hit = p + (size_t)__builtin_ctz(m);
        } else {
            hit = p;
            while (hit < file_len && base[hit] != '\\' && base[hit] != '"') hit++;
            if (hit >= file_len) return (size_t)-1;
        }
        if (__builtin_expect(base[hit] == '"', 0)) { *new_e_io = hit; *bare_out = 1; return hit; }
        if (__builtin_expect(hit + 1 >= file_len, 0)) return (size_t)-1;
        uint8_t c = base[hit + 1];
        if (__builtin_expect(c == 'n', 1)) { *bare_out = 0; return hit + 2; }
        if (c == 'r') {
            if (hit + 3 < file_len && base[hit + 2] == '\\' && base[hit + 3] == 'n') { *bare_out = 0; return hit + 4; }
            p = hit + 2;
        } else if (c == '"' || c == '\\' || c == 't' || c == 'b' || c == 'f') {
            p = hit + 2;
        } else return (size_t)-1;
    }
}

__attribute__((target("avx2,bmi"), noinline))
static int fill_peek_old_avx2(const uint8_t* base, size_t old_e, size_t* cur_io, FastLine* arr, int* cnt_io, int target) {
    int n = *cnt_io;
    size_t cur = *cur_io;
    while (n < target && n < PEEK_MAX && cur < old_e) {
        uint8_t bare = 0;
        size_t end = scan_line_end_bounded_avx2(base, cur, old_e, &bare);
        if (end == (size_t)-1) return -1;
        uint32_t l = (uint32_t)(end - cur);
        arr[n] = (FastLine){ (uint32_t)cur, l, line_f4(base + cur, l), bare };
        cur = end; n++;
    }
    *cur_io = cur; *cnt_io = n;
    return 0;
}

__attribute__((target("avx2,bmi"), noinline))
static int fill_peek_new_avx2(const uint8_t* base, size_t file_len, size_t* new_e_io, size_t* cur_io, FastLine* arr, int* cnt_io, int target) {
    int n = *cnt_io;
    size_t cur = *cur_io;
    while (n < target && n < PEEK_MAX && (*new_e_io == (size_t)-1 || cur < *new_e_io)) {
        uint8_t bare = 0;
        size_t end = scan_line_end_open_avx2(base, cur, file_len, new_e_io, &bare);
        if (end == (size_t)-1) return -1;
        if (end == cur) break;
        uint32_t l = (uint32_t)(end - cur);
        arr[n] = (FastLine){ (uint32_t)cur, l, line_f4(base + cur, l), bare };
        cur = end; n++;
    }
    *cur_io = cur; *cnt_io = n;
    return 0;
}

__attribute__((target("avx2")))
static inline int fl_eq_avx2(const uint8_t* base, const FastLine* a, const FastLine* b) {
    return a->len == b->len && a->f4 == b->f4 && eq_bytes_avx2(base + a->s, base + b->s, a->len);
}

__attribute__((target("avx2"), always_inline))
static inline int match16_avx2(const uint8_t* a, const uint8_t* b) {
    __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b));
    return _mm_testz_si128(x, x);
}

// Batched slow-path resync (called only on the ~40% of gaps not resolved by inline 1-line 16B fast-resync!)
__attribute__((target("avx2,bmi"), noinline))
static int resync_gap_avx2(
    WS* ws, const uint8_t* base, size_t old_e, size_t file_len,
    uint32_t o0_s, uint32_t o0_l, uint8_t o0_bare,
    uint32_t n0_s, uint32_t n0_l, uint8_t n0_bare,
    size_t* po_io, size_t* pn_io, size_t* new_e_io,
    size_t* ni_io, size_t* mj_io, int* old_bare_out, int* new_bare_out
) {
    FastLine op[PEEK_MAX], np[PEEK_MAX];
    op[0] = (FastLine){ o0_s, o0_l, line_f4(base + o0_s, o0_l), o0_bare };
    np[0] = (FastLine){ n0_s, n0_l, line_f4(base + n0_s, n0_l), n0_bare };
    int on = 1, nn = 1;
    size_t o_cur = (size_t)o0_s + o0_l;
    size_t n_cur = (size_t)n0_s + n0_l;
    size_t new_e = *new_e_io;

    int sync_di = -1, sync_dj = -1;

    // Batch-peek 4 lines upfront (covers Tier 0 fallback + sum=2 in ONE call per side!)
    if (fill_peek_old_avx2(base, old_e, &o_cur, op, &on, 4) < 0) return -1;
    if (fill_peek_new_avx2(base, file_len, &new_e, &n_cur, np, &nn, 4) < 0) return -1;

    static const int8_t pairs0[3][2] = { {1, 1}, {1, 0}, {0, 1} };
    for (int k = 0; k < 3; k++) {
        int di = pairs0[k][0], dj = pairs0[k][1];
        if (di < on && dj < nn && fl_eq_avx2(base, &op[di], &np[dj])) {
            if (op[di].len > 16 || di + 1 >= on || dj + 1 >= nn || fl_eq_avx2(base, &op[di + 1], &np[dj + 1])) {
                sync_di = di; sync_dj = dj; break;
            }
        }
    }

    // Tier 1: sum = 2 (already peeked!) then batch-peek 7 for sum = 3..5!
    if (sync_di < 0) {
        for (int sum = 2; sum <= 5 && sync_di < 0; sum++) {
            if (sum == 3) {
                if (fill_peek_old_avx2(base, old_e, &o_cur, op, &on, 7) < 0) return -1;
                if (fill_peek_new_avx2(base, file_len, &new_e, &n_cur, np, &nn, 7) < 0) return -1;
            }
            for (int di = 0; di <= sum; di++) {
                int dj = sum - di;
                if ((di == 1 && dj == 1) || di >= on || dj >= nn) continue;
                if (!fl_eq_avx2(base, &op[di], &np[dj])) continue;
                if (op[di].len <= 16 && di + 1 < on && dj + 1 < nn && !fl_eq_avx2(base, &op[di + 1], &np[dj + 1])) continue;
                sync_di = di; sync_dj = dj; break;
            }
        }
    }

    // Tier 1.5: Incremental one-sided moved-block scan (14 -> 36 lines, requires 2-line match)
    if (sync_di < 0 && on >= 2 && nn >= 2) {
        int limits[2] = { 14, PEEK_MAX };
        for (int li = 0; li < 2 && sync_di < 0; li++) {
            int lim = limits[li];
            int prev_on = on, prev_nn = nn;
            if (fill_peek_old_avx2(base, old_e, &o_cur, op, &on, lim) < 0) return -1;
            int start_i = (li == 0) ? 2 : (prev_on > 2 ? prev_on - 1 : 2);
            for (int i = start_i; i + 1 < on; i++) {
                if (fl_eq_avx2(base, &op[i], &np[0]) && fl_eq_avx2(base, &op[i + 1], &np[1])) {
                    sync_di = i; sync_dj = 0; break;
                }
            }
            if (sync_di >= 0) break;
            if (fill_peek_new_avx2(base, file_len, &new_e, &n_cur, np, &nn, lim) < 0) return -1;
            int start_j = (li == 0) ? 2 : (prev_nn > 2 ? prev_nn - 1 : 2);
            for (int j = start_j; j + 1 < nn; j++) {
                if (fl_eq_avx2(base, &op[0], &np[j]) && fl_eq_avx2(base, &op[1], &np[j + 1])) {
                    sync_di = 0; sync_dj = j; break;
                }
            }
        }
    }

    // Tier 2: General 2-sided large gap (up to 36 lines, O(1) hash pair table)
    if (sync_di < 0) {
        if (fill_peek_old_avx2(base, old_e, &o_cur, op, &on, PEEK_MAX) < 0) return -1;
        if (fill_peek_new_avx2(base, file_len, &new_e, &n_cur, np, &nn, PEEK_MAX) < 0) return -1;
        if (on >= 2 && nn >= 2) {
            uint32_t ohp[PEEK_MAX], nhp[PEEK_MAX];
            for (int i = 0; i < on; i++) ohp[i] = (op[i].f4 * 2654435761u) ^ op[i].len;
            for (int j = 0; j < nn; j++) nhp[j] = (np[j].f4 * 2654435761u) ^ np[j].len;
            uint8_t pht[256]; mem_zero_avx2(pht, 256);
            for (int i = 0; i + 1 < on; i++) {
                uint32_t s = ((ohp[i] * 2654435761u) ^ ohp[i + 1]) >> 24;
                if (pht[s] == 0) pht[s] = (uint8_t)(i + 1);
                else if (pht[(s + 1) & 255] == 0) pht[(s + 1) & 255] = (uint8_t)(i + 1);
            }
            int best_sum = 100000;
            for (int j = 0; j + 1 < nn && j < best_sum; j++) {
                uint32_t s = ((nhp[j] * 2654435761u) ^ nhp[j + 1]) >> 24;
                for (int pr = 0; pr < 2; pr++) {
                    uint8_t idx1 = pht[(s + pr) & 255];
                    if (!idx1) break;
                    int i = idx1 - 1;
                    if (i + j == 0 || i + j >= best_sum) continue;
                    if (ohp[i] == nhp[j] && ohp[i + 1] == nhp[j + 1] &&
                        fl_eq_avx2(base, &op[i], &np[j]) && fl_eq_avx2(base, &op[i + 1], &np[j + 1])) {
                        best_sum = i + j; sync_di = i; sync_dj = j;
                    }
                }
            }
        }
    }

    int emit_o = (sync_di >= 0) ? sync_di : on;
    int emit_n = (sync_dj >= 0) ? sync_dj : nn;
    if (emit_o == 0 && emit_n == 0) { emit_o = on; emit_n = nn; }

    size_t ni = *ni_io, mj = *mj_io;
    if (__builtin_expect(ni + (size_t)emit_o + 4 > ws->old_off_cap || mj + (size_t)emit_n + 4 > ws->new_off_cap || ni + (size_t)emit_o + 4 > ws->old_h_cap, 0)) {
        if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + (size_t)emit_o + 2048) < 0) return -1;
        if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + (size_t)emit_n + 2048) < 0) return -1;
        if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + (size_t)emit_o + 2048) < 0) return -1;
    }
    uint32_t* ooff = ws->old_off; uint32_t* noff = ws->new_off; oh_t* oh = ws->old_h;
    uint64_t* seen = ws->seen;
    for (int i = 0; i < emit_o; i++) {
        ooff[ni + 1] = op[i].s + op[i].len;
        oh[ni] = record_gap_old(seen, base + op[i].s, op[i].len);
        if (op[i].bare) *old_bare_out = 1;
        ni++;
    }
    for (int j = 0; j < emit_n; j++) {
        noff[mj + 1] = np[j].s + np[j].len;
        record_gap_new(seen, base + np[j].s, np[j].len);
        if (np[j].bare) *new_bare_out = 1;
        mj++;
    }
    *po_io = (emit_o > 0) ? (op[emit_o - 1].s + op[emit_o - 1].len) : *po_io;
    *pn_io = (emit_n > 0) ? (np[emit_n - 1].s + np[emit_n - 1].len) : *pn_io;
    *new_e_io = new_e;
    *ni_io = ni;
    *mj_io = mj;
    return 0;
}

// ---------------------------------------------------------------------------
// Single-Pass FUSED 64B/iter AVX2 Lockstep Block-Sync (open-ended `new`!)
// ---------------------------------------------------------------------------

__attribute__((target("avx2,bmi"), noinline))
static size_t block_sync_avx2(
    WS* ws, const uint8_t* base, size_t file_len,
    size_t old_s, size_t old_e, size_t new_s, size_t new_e_known,
    int* old_bare_out, int* new_bare_out
) {
    *old_bare_out = 0; *new_bare_out = 0;
    mem_zero_avx2(ws->seen, BF_WORDS * 8);
    ws->seen[0] = 3ULL; // Reserve bucket 0 as permanently non-unique!
    ws->runs_len = 0;
    uint64_t* seen = ws->seen;

    uint32_t* ooff = ws->old_off; size_t ocap = ws->old_off_cap;
    uint32_t* noff = ws->new_off; size_t ncap = ws->new_off_cap;
    oh_t*     oh   = ws->old_h;   size_t hcap = ws->old_h_cap;
    ooff[0] = (uint32_t)old_s; noff[0] = (uint32_t)new_s;

    size_t po = old_s, pn = new_s, new_e = new_e_known;
    size_t ni = 0, mj = 0;
    __m256i v_bs = _mm256_set1_epi8('\\');

    while (po < old_e && pn < (new_e != (size_t)-1 ? new_e : file_len)) {
        if (new_e == (size_t)-1 && base[pn] == '"') { new_e = pn; break; }
#ifdef STANDALONE_TEST
        uint64_t _tr0 = _rdtsc();
#endif
        size_t run_a0 = ni, run_b0 = mj;
        intptr_t delta = (intptr_t)pn - (intptr_t)po;
        size_t ls = po, q = po;
        size_t rem_o = old_e - po;
        size_t n_lim = (new_e != (size_t)-1) ? new_e : file_len;
        size_t rem_n = n_lim - pn;
        size_t max_lim = po + (rem_o < rem_n ? rem_o : rem_n);

        // Tight 64B/iter fused AVX2 lockstep compare + backslash scan + 1-mul hash!
        while (q + 64 <= max_lim) {
            __m256i o0 = _mm256_loadu_si256((const __m256i*)(base + q));
            __m256i n0 = _mm256_loadu_si256((const __m256i*)(base + q + delta));
            __m256i o1 = _mm256_loadu_si256((const __m256i*)(base + q + 32));
            __m256i n1 = _mm256_loadu_si256((const __m256i*)(base + q + delta + 32));
            uint32_t eq0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o0, n0));
            uint32_t eq1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o1, n1));
            if (__builtin_expect((eq0 & eq1) == 0xFFFFFFFFu, 1)) {
                uint32_t bs0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o0, v_bs));
                uint32_t bs1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o1, v_bs));
                uint64_t mask = (uint64_t)bs0 | ((uint64_t)bs1 << 32);
                if (mask == 0) { q += 64; continue; }
                if (__builtin_expect(ni + 40 > ocap || mj + 40 > ncap || ni + 40 > hcap, 0)) {
                    if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 2048) < 0) return (size_t)-1;
                    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 2048) < 0) return (size_t)-1;
                    if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 2048) < 0) return (size_t)-1;
                    ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
                }
                size_t chunk_end = q + 64;
                while (mask) {
                    uint32_t tz = (uint32_t)__builtin_ctzll(mask);
                    size_t hit = q + tz;
                    if (__builtin_expect(hit + 1 >= chunk_end, 0)) { q = hit; goto f_next; }
                    uint8_t c = base[hit + 1];
                    if (__builtin_expect(c == 'n', 1)) {
                        size_t le = hit + 2;
                        ooff[ni + 1] = (uint32_t)le;
                        noff[mj + 1] = (uint32_t)((intptr_t)le + delta);
                        oh[ni] = record_sync_line(seen, base + ls, le - ls);
                        ni++; mj++; ls = le;
                        mask &= mask - 1;
                    } else if (c == 'r') {
                        if (__builtin_expect(hit + 3 >= chunk_end, 0)) { q = hit; goto f_next; }
                        if (base[hit + 2] == '\\' && base[hit + 3] == 'n') {
                            size_t le = hit + 4;
                            ooff[ni + 1] = (uint32_t)le;
                            noff[mj + 1] = (uint32_t)((intptr_t)le + delta);
                            oh[ni] = record_sync_line(seen, base + ls, le - ls);
                            ni++; mj++; ls = le;
                            mask &= mask - 1;
                            if (tz + 2 < 64) mask &= ~(1ULL << (tz + 2)); else { q = le; goto f_next; }
                        } else mask &= mask - 1;
                    } else if (c == '\\' || c == '"') {
                        mask &= mask - 1;
                        if (tz + 1 < 64) mask &= ~(1ULL << (tz + 1)); else { q = hit + 2; goto f_next; }
                    } else if (c == 't' || c == 'b' || c == 'f') {
                        mask &= mask - 1;
                    } else return (size_t)-1;
                }
                q = chunk_end;
            f_next:;
            } else {
                size_t mism = (eq0 != 0xFFFFFFFFu) ? (size_t)__builtin_ctz(~eq0) : (32 + (size_t)__builtin_ctz(~eq1));
                size_t lim = q + mism;
                uint32_t bs0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o0, v_bs));
                uint32_t bs1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(o1, v_bs));
                uint64_t mask = (uint64_t)bs0 | ((uint64_t)bs1 << 32);
                mask &= (mism == 0) ? 0ULL : ((1ULL << mism) - 1ULL);
                if (mask) {
                    if (__builtin_expect(ni + 40 > ocap || mj + 40 > ncap || ni + 40 > hcap, 0)) {
                        if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 2048) < 0) return (size_t)-1;
                        if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 2048) < 0) return (size_t)-1;
                        if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 2048) < 0) return (size_t)-1;
                        ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
                    }
                    while (mask) {
                        uint32_t tz = (uint32_t)__builtin_ctzll(mask);
                        size_t hit = q + tz;
                        if (hit + 1 >= lim) break;
                        uint8_t c = base[hit + 1];
                        if (__builtin_expect(c == 'n', 1)) {
                            size_t le = hit + 2;
                            ooff[ni + 1] = (uint32_t)le; noff[mj + 1] = (uint32_t)((intptr_t)le + delta);
                            oh[ni] = record_sync_line(seen, base + ls, le - ls);
                            ni++; mj++; ls = le; mask &= mask - 1;
                        } else if (c == 'r') {
                            if (hit + 3 >= lim) break;
                            if (base[hit + 2] == '\\' && base[hit + 3] == 'n') {
                                size_t le = hit + 4;
                                ooff[ni + 1] = (uint32_t)le; noff[mj + 1] = (uint32_t)((intptr_t)le + delta);
                                oh[ni] = record_sync_line(seen, base + ls, le - ls);
                                ni++; mj++; ls = le; mask &= mask - 1;
                                if (tz + 2 < 64) mask &= ~(1ULL << (tz + 2)); else break;
                            } else mask &= mask - 1;
                        } else if (c == '\\' || c == '"') {
                            mask &= mask - 1;
                            if (tz + 1 < 64) mask &= ~(1ULL << (tz + 1)); else break;
                        } else if (c == 't' || c == 'b' || c == 'f') mask &= mask - 1;
                        else return (size_t)-1;
                    }
                }
                goto run_done;
            }
        }

        // Short tail (< 64 bytes left)
        {
            size_t lim = q;
            while (lim < max_lim && base[lim] == base[lim + delta]) lim++;
            size_t tq = q;
            while (tq < lim) {
                if (base[tq] == '\\') {
                    if (tq + 1 >= lim) break;
                    uint8_t c = base[tq + 1];
                    size_t le = 0;
                    if (c == 'n') le = tq + 2;
                    else if (c == 'r') {
                        if (tq + 3 >= lim) break;
                        if (base[tq + 2] == '\\' && base[tq + 3] == 'n') le = tq + 4;
                        else { tq += 2; continue; }
                    } else if (c == '\\' || c == '"' || c == 't' || c == 'b' || c == 'f') { tq += 2; continue; }
                    else return (size_t)-1;
                    if (ni + 4 > ocap || mj + 4 > ncap || ni + 4 > hcap) {
                        if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 1024) < 0) return (size_t)-1;
                        if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 1024) < 0) return (size_t)-1;
                        if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 1024) < 0) return (size_t)-1;
                        ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
                    }
                    ooff[ni + 1] = (uint32_t)le; noff[mj + 1] = (uint32_t)((intptr_t)le + delta);
                    oh[ni] = record_sync_line(seen, base + ls, le - ls);
                    ni++; mj++; ls = le; tq = le;
                } else tq++;
            }
            if (lim == old_e && tq == lim) {
                size_t n_at = (size_t)((intptr_t)lim + delta);
                int new_ended_here = (new_e != (size_t)-1) ? (n_at == new_e) : (n_at < file_len && base[n_at] == '"');
                if (new_ended_here) {
                    new_e = n_at;
                    if (lim > ls) {
                        if (ni + 4 > ocap || mj + 4 > ncap || ni + 4 > hcap) {
                            if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 64) < 0) return (size_t)-1;
                            if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 64) < 0) return (size_t)-1;
                            if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 64) < 0) return (size_t)-1;
                            ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
                        }
                        ooff[ni + 1] = (uint32_t)old_e; noff[mj + 1] = (uint32_t)new_e;
                        oh[ni] = record_sync_line(seen, base + ls, old_e - ls);
                        ni++; mj++; ls = old_e;
                        *old_bare_out = 1; *new_bare_out = 1;
                    }
                }
            }
        }

    run_done:
#ifdef STANDALONE_TEST
        cyc_run += _rdtsc() - _tr0;
#endif
        if (ni > run_a0) {
            if (__builtin_expect(ws->runs_len >= ws->runs_cap, 0) && ensure_runs(ws, ws->runs_len + 1) < 0) return (size_t)-1;
            ws->runs[ws->runs_len++] = (SyncRun){ (uint32_t)run_a0, (uint32_t)run_b0, (uint32_t)(ni - run_a0) };
        }
        po = ls; pn = (size_t)((intptr_t)ls + delta);
        if (po >= old_e) break;
        if (new_e != (size_t)-1) { if (pn >= new_e) break; }
        else if (pn < file_len && base[pn] == '"') { new_e = pn; break; }

        // INLINE 1-LINE 16B FAST-RESYNC (proven safe on open-ended `new`: old[o1s..o1s+15] has no unescaped '"')
#ifdef STANDALONE_TEST
        uint64_t _tf0 = _rdtsc();
#endif
        uint8_t ob = 0, nb = 0;
        size_t o1s = scan_line_end_bounded_avx2(base, po, old_e, &ob);
        if (__builtin_expect(o1s == (size_t)-1, 0)) return (size_t)-1;
        size_t n1s = scan_line_end_open_avx2(base, pn, file_len, &new_e, &nb);
        if (__builtin_expect(n1s == (size_t)-1, 0)) return (size_t)-1;
        if (__builtin_expect(n1s == pn, 0)) break; // hit closing quote of `new`

        size_t n_bound = (new_e != (size_t)-1) ? new_e : file_len;
        if (__builtin_expect(o1s + 16 <= old_e && n1s + 16 <= n_bound && match16_avx2(base + o1s, base + n1s), 1)) {
            if (__builtin_expect(ni + 4 > ocap || mj + 4 > ncap || ni + 4 > hcap, 0)) {
                if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 2048) < 0) return (size_t)-1;
                if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 2048) < 0) return (size_t)-1;
                if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 2048) < 0) return (size_t)-1;
                ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
            }
            ooff[ni + 1] = (uint32_t)o1s; oh[ni] = record_gap_old(seen, base + po, o1s - po); ni++;
            noff[mj + 1] = (uint32_t)n1s; record_gap_new(seen, base + pn, n1s - pn); mj++;
            po = o1s; pn = n1s;
#ifdef STANDALONE_TEST
            cyc_fastgap += _rdtsc() - _tf0; cnt_fg++;
#endif
            continue;
        }
        if (o1s + 16 <= old_e && pn + 16 <= n_bound && match16_avx2(base + o1s, base + pn)) {
            if (__builtin_expect(ni + 4 > ocap || ni + 4 > hcap, 0)) {
                if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 2048) < 0) return (size_t)-1;
                if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 2048) < 0) return (size_t)-1;
                ooff = ws->old_off; ocap = ws->old_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
            }
            ooff[ni + 1] = (uint32_t)o1s; oh[ni] = record_gap_old(seen, base + po, o1s - po); ni++;
            po = o1s;
#ifdef STANDALONE_TEST
            cyc_fastgap += _rdtsc() - _tf0; cnt_fg++;
#endif
            continue;
        }
        if (po + 16 <= old_e && n1s + 16 <= n_bound && match16_avx2(base + po, base + n1s)) {
            if (__builtin_expect(mj + 4 > ncap, 0)) {
                if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 2048) < 0) return (size_t)-1;
                noff = ws->new_off; ncap = ws->new_off_cap;
            }
            noff[mj + 1] = (uint32_t)n1s; record_gap_new(seen, base + pn, n1s - pn); mj++;
            pn = n1s;
#ifdef STANDALONE_TEST
            cyc_fastgap += _rdtsc() - _tf0; cnt_fg++;
#endif
            continue;
        }
#ifdef STANDALONE_TEST
        cyc_fastgap += _rdtsc() - _tf0;
        uint64_t _tg0 = _rdtsc();
#endif
        if (resync_gap_avx2(ws, base, old_e, file_len,
                            (uint32_t)po, (uint32_t)(o1s - po), ob,
                            (uint32_t)pn, (uint32_t)(n1s - pn), nb,
                            &po, &pn, &new_e, &ni, &mj, old_bare_out, new_bare_out) < 0) return (size_t)-1;
        ooff = ws->old_off; ocap = ws->old_off_cap; noff = ws->new_off; ncap = ws->new_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
#ifdef STANDALONE_TEST
        cyc_gap += _rdtsc() - _tg0; cnt_sg++;
#endif
    }

    while (po < old_e) {
        uint8_t bare = 0;
        size_t end = scan_line_end_bounded_avx2(base, po, old_e, &bare);
        if (end == (size_t)-1) return (size_t)-1;
        if (__builtin_expect(ni + 4 > ocap || ni + 4 > hcap, 0)) {
            if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 1024) < 0) return (size_t)-1;
            if (ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 1024) < 0) return (size_t)-1;
            ooff = ws->old_off; ocap = ws->old_off_cap; oh = ws->old_h; hcap = ws->old_h_cap;
        }
        ooff[ni + 1] = (uint32_t)end;
        oh[ni] = record_gap_old(seen, base + po, end - po);
        if (bare) *old_bare_out = 1;
        ni++; po = end;
    }
    while (pn < (new_e != (size_t)-1 ? new_e : file_len)) {
        uint8_t bare = 0;
        size_t end = scan_line_end_open_avx2(base, pn, file_len, &new_e, &bare);
        if (end == (size_t)-1) return (size_t)-1;
        if (end == pn) break;
        if (__builtin_expect(mj + 4 > ncap, 0)) {
            if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 1024) < 0) return (size_t)-1;
            noff = ws->new_off; ncap = ws->new_off_cap;
        }
        noff[mj + 1] = (uint32_t)end;
        record_gap_new(seen, base + pn, end - pn);
        if (bare) *new_bare_out = 1;
        mj++; pn = end;
    }
    if (new_e == (size_t)-1) {
        if (pn < file_len && base[pn] == '"') new_e = pn;
        else return (size_t)-1;
    }

    ws->a_len = ni; ws->b_len = mj;
    return new_e;
}

// ---------------------------------------------------------------------------
// Ultra-compact -Os SSE2 Fallback Block-Sync
// ---------------------------------------------------------------------------

SSE2_OPT
static int scan_line_sse2(const uint8_t* base, size_t start, size_t limit, int open_qt, FastLine* out, size_t* end_qt_out) {
    if (open_qt) {
        if (start >= limit) return -1;
        if (base[start] == '"') { *end_qt_out = start; return 0; }
    }
    size_t p = start;
    while (p < limit) {
        uint8_t b = base[p];
        if (open_qt && b == '"') {
            uint32_t l = (uint32_t)(p - start);
            *out = (FastLine){ (uint32_t)start, l, line_f4(base + start, l), 1 };
            *end_qt_out = p; return 1;
        }
        if (b == '\\') {
            if (p + 1 >= limit) return -1;
            uint8_t c = base[p + 1];
            if (c == 'n') { uint32_t l = (uint32_t)(p + 2 - start); *out = (FastLine){ (uint32_t)start, l, line_f4(base + start, l), 0 }; return 1; }
            if (c == 'r') {
                if (p + 3 < limit && base[p + 2] == '\\' && base[p + 3] == 'n') {
                    uint32_t l = (uint32_t)(p + 4 - start); *out = (FastLine){ (uint32_t)start, l, line_f4(base + start, l), 0 }; return 1;
                }
                p += 2; continue;
            }
            if (c == '"' || c == '\\' || c == 't' || c == 'b' || c == 'f') { p += 2; continue; }
            return -1;
        }
        p++;
    }
    if (open_qt) return -1;
    uint32_t l = (uint32_t)(limit - start);
    *out = (FastLine){ (uint32_t)start, l, line_f4(base + start, l), 1 };
    return 1;
}

SSE2_OPT
static int fl_eq_sse2(const uint8_t* base, const FastLine* a, const FastLine* b) {
    return a->len == b->len && a->f4 == b->f4 && eq_bytes_sse2(base + a->s, base + b->s, a->len);
}

SSE2_OPT
static size_t block_sync_sse2(
    WS* ws, const uint8_t* base, size_t file_len,
    size_t old_s, size_t old_e, size_t new_s, size_t new_e_known,
    int* old_bare_out, int* new_bare_out
) {
    *old_bare_out = 0; *new_bare_out = 0;
    mem_zero_sse2(ws->seen, BF_WORDS * 8);
    ws->seen[0] = 3ULL;
    ws->runs_len = 0;
    uint64_t* seen = ws->seen;
    ws->old_off[0] = (uint32_t)old_s; ws->new_off[0] = (uint32_t)new_s;

    size_t po = old_s, pn = new_s, new_e = new_e_known, ni = 0, mj = 0;
    while (po < old_e && pn < (new_e != (size_t)-1 ? new_e : file_len)) {
        if (new_e == (size_t)-1 && base[pn] == '"') { new_e = pn; break; }
        size_t run_a0 = ni, run_b0 = mj;
        intptr_t delta = (intptr_t)pn - (intptr_t)po;
        size_t n_lim = (new_e != (size_t)-1) ? new_e : file_len;
        size_t max_lim = po + ((old_e - po) < (n_lim - pn) ? (old_e - po) : (n_lim - pn));
        size_t lim = po;
        while (lim + 16 <= max_lim) {
            uint32_t eq = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128((const __m128i*)(base + lim)),
                _mm_loadu_si128((const __m128i*)(base + lim + delta))));
            if (eq == 0xFFFFu) lim += 16; else { lim += (size_t)__builtin_ctz((~eq) & 0xFFFFu); break; }
        }
        while (lim < max_lim && base[lim] == base[lim + delta]) lim++;
        size_t ls = po, tq = po;
        while (tq < lim) {
            if (base[tq] == '\\') {
                if (tq + 1 >= lim) break;
                uint8_t c = base[tq + 1];
                size_t le = 0;
                if (c == 'n') le = tq + 2;
                else if (c == 'r') {
                    if (tq + 3 >= lim) break;
                    if (base[tq + 2] == '\\' && base[tq + 3] == 'n') le = tq + 4;
                    else { tq += 2; continue; }
                } else if (c == '\\' || c == '"' || c == 't' || c == 'b' || c == 'f') { tq += 2; continue; }
                else return (size_t)-1;
                if (ni + 4 > ws->old_off_cap && ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 1024) < 0) return (size_t)-1;
                if (mj + 4 > ws->new_off_cap && ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 1024) < 0) return (size_t)-1;
                if (ni + 4 > ws->old_h_cap && ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 1024) < 0) return (size_t)-1;
                ws->old_off[ni + 1] = (uint32_t)le; ws->new_off[mj + 1] = (uint32_t)((intptr_t)le + delta);
                ws->old_h[ni] = record_sync_line(seen, base + ls, le - ls);
                ni++; mj++; ls = le; tq = le;
            } else tq++;
        }
        if (lim == old_e && tq == lim) {
            size_t n_at = (size_t)((intptr_t)lim + delta);
            int ne_hit = (new_e != (size_t)-1) ? (n_at == new_e) : (n_at < file_len && base[n_at] == '"');
            if (ne_hit) {
                new_e = n_at;
                if (lim > ls) {
                    if (ni + 4 > ws->old_off_cap && ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 64) < 0) return (size_t)-1;
                    if (mj + 4 > ws->new_off_cap && ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 64) < 0) return (size_t)-1;
                    if (ni + 4 > ws->old_h_cap && ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 64) < 0) return (size_t)-1;
                    ws->old_off[ni + 1] = (uint32_t)old_e; ws->new_off[mj + 1] = (uint32_t)new_e;
                    ws->old_h[ni] = record_sync_line(seen, base + ls, old_e - ls);
                    ni++; mj++; ls = old_e; *old_bare_out = 1; *new_bare_out = 1;
                }
            }
        }
        if (ni > run_a0) {
            if (ws->runs_len >= ws->runs_cap && ensure_runs(ws, ws->runs_len + 1) < 0) return (size_t)-1;
            ws->runs[ws->runs_len++] = (SyncRun){ (uint32_t)run_a0, (uint32_t)run_b0, (uint32_t)(ni - run_a0) };
        }
        po = ls; pn = (size_t)((intptr_t)ls + delta);
        if (po >= old_e) break;
        if (new_e != (size_t)-1) { if (pn >= new_e) break; }
        else if (pn < file_len && base[pn] == '"') { new_e = pn; break; }

        FastLine op[PEEK_MAX], np[PEEK_MAX];
        int on = 0, nn = 0, sync_di = -1, sync_dj = -1;
        size_t oc = po, nc = pn;
        #define SO(w) do { while (on < (w) && on < PEEK_MAX && oc < old_e) { if (scan_line_sse2(base, oc, old_e, 0, &op[on], NULL) < 0) return (size_t)-1; oc = op[on].s + op[on].len; on++; } } while (0)
        #define SN(w) do { while (nn < (w) && nn < PEEK_MAX && nc < new_e) { \
            size_t qp = (size_t)-1; int rc = scan_line_sse2(base, nc, new_e != (size_t)-1 ? new_e : file_len, new_e == (size_t)-1, &np[nn], &qp); \
            if (rc < 0) return (size_t)-1; if (qp != (size_t)-1) new_e = qp; if (rc == 0) break; nc = np[nn].s + np[nn].len; nn++; \
        } } while (0)
        for (int sum = 1; sum <= 5 && sync_di < 0; sum++) {
            SO(sum + 2); SN(sum + 2);
            for (int di = 0; di <= sum; di++) {
                int dj = sum - di;
                if (di >= on || dj >= nn || !fl_eq_sse2(base, &op[di], &np[dj])) continue;
                if (op[di].len <= 16 && di + 1 < on && dj + 1 < nn && !fl_eq_sse2(base, &op[di + 1], &np[dj + 1])) continue;
                sync_di = di; sync_dj = dj; break;
            }
        }
        if (sync_di < 0) {
            SO(PEEK_MAX); SN(PEEK_MAX);
            for (int i = 0; i + 1 < on && sync_di < 0; i++) {
                for (int j = 0; j + 1 < nn; j++) {
                    if (i + j > 0 && fl_eq_sse2(base, &op[i], &np[j]) && fl_eq_sse2(base, &op[i + 1], &np[j + 1])) {
                        sync_di = i; sync_dj = j; break;
                    }
                }
            }
        }
        int eo = sync_di >= 0 ? sync_di : on, en = sync_dj >= 0 ? sync_dj : nn;
        if (eo == 0 && en == 0) { eo = on; en = nn; }
        if (ni + (size_t)eo + 4 > ws->old_off_cap && ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + eo + 1024) < 0) return (size_t)-1;
        if (mj + (size_t)en + 4 > ws->new_off_cap && ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + en + 1024) < 0) return (size_t)-1;
        if (ni + (size_t)eo + 4 > ws->old_h_cap && ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + eo + 1024) < 0) return (size_t)-1;
        for (int i = 0; i < eo; i++) { ws->old_off[ni + 1] = op[i].s + op[i].len; ws->old_h[ni] = record_gap_old(seen, base + op[i].s, op[i].len); if (op[i].bare) *old_bare_out = 1; ni++; }
        for (int j = 0; j < en; j++) { ws->new_off[mj + 1] = np[j].s + np[j].len; record_gap_new(seen, base + np[j].s, np[j].len); if (np[j].bare) *new_bare_out = 1; mj++; }
        po = eo > 0 ? (op[eo - 1].s + op[eo - 1].len) : po;
        pn = en > 0 ? (np[en - 1].s + np[en - 1].len) : pn;
        #undef SO
        #undef SN
    }
    while (po < old_e) {
        FastLine pl;
        if (scan_line_sse2(base, po, old_e, 0, &pl, NULL) < 0) return (size_t)-1;
        if (ni + 4 > ws->old_off_cap && ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, ni + 1, ni + 1024) < 0) return (size_t)-1;
        if (ni + 4 > ws->old_h_cap && ensure_oh(&ws->arena, &ws->old_h, &ws->old_h_cap, ni, ni + 1024) < 0) return (size_t)-1;
        ws->old_off[ni + 1] = pl.s + pl.len; ws->old_h[ni] = record_gap_old(seen, base + pl.s, pl.len);
        if (pl.bare) *old_bare_out = 1;
        ni++; po = pl.s + pl.len;
    }
    while (pn < (new_e != (size_t)-1 ? new_e : file_len)) {
        FastLine pl; size_t qp = (size_t)-1;
        int rc = scan_line_sse2(base, pn, new_e != (size_t)-1 ? new_e : file_len, new_e == (size_t)-1, &pl, &qp);
        if (rc < 0) return (size_t)-1;
        if (qp != (size_t)-1) new_e = qp;
        if (rc == 0) break;
        if (mj + 4 > ws->new_off_cap && ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, mj + 1, mj + 1024) < 0) return (size_t)-1;
        ws->new_off[mj + 1] = pl.s + pl.len; record_gap_new(seen, base + pl.s, pl.len);
        if (pl.bare) *new_bare_out = 1;
        mj++; pn = pl.s + pl.len;
    }
    if (new_e == (size_t)-1) { if (pn < file_len && base[pn] == '"') new_e = pn; else return (size_t)-1; }
    ws->a_len = ni; ws->b_len = mj;
    return new_e;
}

// ---------------------------------------------------------------------------
// Exact Local Subproblem Solver (Exact 1xM/Nx1 + Compact Tiny + Old-only HT Big)
// ---------------------------------------------------------------------------

static inline int emit_chg_block(WS* ws, uint32_t o, uint32_t p, uint32_t dels, uint32_t ins, uint32_t* total_del, uint32_t* total_ins) {
    size_t n = ws->chg_len;
    if (n > 0 && ws->chg[n - 1].o + ws->chg[n - 1].dels == o && ws->chg[n - 1].p + ws->chg[n - 1].ins == p) {
        ws->chg[n - 1].dels += dels;
        ws->chg[n - 1].ins  += ins;
    } else {
        if (__builtin_expect(n >= ws->chg_cap, 0) && ensure_chg(ws, n + 1) < 0) return -1;
        ws->chg[n] = (Chg){ o, p, dels, ins };
        ws->chg_len = n + 1;
    }
    *total_del += dels;
    *total_ins += ins;
    return 0;
}

static inline int lines_eq(const uint8_t* base, const uint32_t* ooff, uint32_t i, const uint32_t* noff, uint32_t j) {
    uint32_t os = ooff[i], oe = ooff[i + 1], ns = noff[j], ne = noff[j + 1];
    uint32_t ol = oe - os;
    if (ol != (ne - ns)) return 0;
    return eq_bytes_sse2(base + os, base + ns, ol);
}

COLD_OPT
static int flush_segment(
    WS* ws, size_t rel_d, int32_t end_x, int32_t end_k, int32_t seg_x,
    uint32_t* cur_o, uint32_t* cur_p, uint32_t* total_del, uint32_t* total_ins
) {
    const uint32_t* rbase = ws->row_base;
    const int32_t* rlo = ws->row_lo;
    const int32_t* trx = ws->tr_x;
    const uint8_t* trc = ws->tr_c;

    if (rel_d == 1) {
        size_t idx = (size_t)rbase[1] + (size_t)((end_k - rlo[1]) >> 1);
        uint8_t is_del = trc[idx];
        int32_t start_x = is_del ? (seg_x + 1) : seg_x;
        if (emit_chg_block(ws, *cur_o, *cur_p, is_del ? 1 : 0, is_del ? 0 : 1, total_del, total_ins) < 0) return -1;
        uint32_t snake = (uint32_t)(end_x - start_x);
        *cur_o += (is_del ? 1 : 0) + snake;
        *cur_p += (is_del ? 0 : 1) + snake;
        return 0;
    }

    size_t need_ops = rel_d * 2 + 4;
    if (need_ops > ws->seg_cap) {
        size_t nc = ws->seg_cap ? ws->seg_cap * 2 : 1024;
        while (nc < need_ops) nc *= 2;
        uint32_t* np = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        if (!np) return -1;
        ws->seg_ops = np; ws->seg_cap = nc;
    }
    uint32_t* sops = ws->seg_ops;
    size_t nops = 0;
    int32_t cx = end_x, ck = end_k;
    for (size_t rd = rel_d; rd >= 1; rd--) {
        size_t idx = (size_t)rbase[rd] + (size_t)((ck - rlo[rd]) >> 1);
        uint8_t is_del = trc[idx];
        int32_t prev_k = is_del ? (ck - 1) : (ck + 1);
        int32_t prev_x = (rd == 1) ? seg_x : trx[(size_t)rbase[rd - 1] + (size_t)((prev_k - rlo[rd - 1]) >> 1)];
        int32_t start_x = is_del ? (prev_x + 1) : prev_x;
        uint32_t snake = (uint32_t)(cx - start_x);
        if (snake > 0) sops[nops++] = snake;
        sops[nops++] = (is_del ? (1u << 30) : (2u << 30)) | 1u;
        cx = prev_x; ck = prev_k;
    }
    for (size_t si = nops; si > 0; si--) {
        uint32_t op = sops[si - 1];
        uint32_t kind = op >> 30, cnt = op & 0x3FFFFFFFu;
        if (kind == 0) { *cur_o += cnt; *cur_p += cnt; }
        else if (kind == 1) { if (emit_chg_block(ws, *cur_o, *cur_p, cnt, 0, total_del, total_ins) < 0) return -1; *cur_o += cnt; }
        else { if (emit_chg_block(ws, *cur_o, *cur_p, 0, cnt, total_del, total_ins) < 0) return -1; *cur_p += cnt; }
    }
    return 0;
}

// Fast Zero-HT Stack Tiny-Myers (n, m <= 12, n >= 2, m >= 2; incremental v sentinel init!)
__attribute__((noinline))
static int solve_tiny_myers(
    WS* ws, const uint8_t* base,
    uint32_t aL, uint32_t n, uint32_t bL, uint32_t m,
    uint32_t* total_del, uint32_t* total_ins
) {
    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    uint8_t aid[13], bid[13];
    uint32_t rep_s[12], rep_l[12];
    uint8_t nid = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t s = ooff[aL + i], l = ooff[aL + i + 1] - s;
        uint8_t id = nid;
        for (uint8_t r = 0; r < nid; r++) {
            if (rep_l[r] == l && eq_bytes_sse2(base + rep_s[r], base + s, l)) { id = r; break; }
        }
        if (id == nid) { rep_s[nid] = s; rep_l[nid] = l; nid++; }
        aid[i] = id;
    }
    int any_match = 0;
    for (uint32_t j = 0; j < m; j++) {
        uint32_t s = noff[bL + j], l = noff[bL + j + 1] - s;
        uint8_t id = 200;
        for (uint8_t r = 0; r < nid; r++) {
            if (rep_l[r] == l && eq_bytes_sse2(base + rep_s[r], base + s, l)) { id = r; any_match = 1; break; }
        }
        bid[j] = id;
    }
    if (!any_match) return emit_chg_block(ws, aL, bL, n, m, total_del, total_ins);

    aid[n] = 254; bid[m] = 255;

    int8_t v[56];
    int8_t tr_x[25][52];
    uint8_t tr_c[25][52];
    const int OFF = 26;
    int32_t ni = (int32_t)n, mi = (int32_t)m, max_d = ni + mi;
    v[OFF - 1] = -2; v[OFF] = 0; v[OFF + 1] = -2;
    int32_t found_d = max_d;

    for (int32_t d = 1; d <= max_d; d++) {
        v[OFF - d - 1] = -2;
        v[OFF + d + 1] = -2;
        for (int32_t k = -d; k <= d; k += 2) {
            int8_t rem_x = v[OFF + k - 1], add_x = v[OFF + k + 1];
            int can_rem = (rem_x != -2) & (rem_x < ni);
            int can_add = (add_x != -2) & ((add_x - k) <= mi);
            if (!(can_rem | can_add)) { v[OFF + k] = -2; tr_x[d][OFF + k] = -2; continue; }
            int32_t by_del = can_rem ? (rem_x + 1) : -2;
            int32_t by_ins = can_add ? add_x : -2;
            int32_t x; uint8_t is_del;
            if (by_del >= by_ins) { x = by_del; is_del = 1; } else { x = by_ins; is_del = 0; }
            int32_t y = x - k;
            while (aid[x] == bid[y]) { x++; y++; }
            v[OFF + k] = (int8_t)x; tr_x[d][OFF + k] = (int8_t)x; tr_c[d][OFF + k] = is_del;
            if (x >= ni && y >= mi) { found_d = d; goto tiny_done; }
        }
    }
tiny_done:;
    uint32_t ops[52]; int nops = 0;
    int32_t cx = ni, ck = ni - mi;
    for (int32_t d = found_d; d >= 1; d--) {
        uint8_t is_del = tr_c[d][OFF + ck];
        int32_t prev_k = is_del ? (ck - 1) : (ck + 1);
        int32_t prev_x = (d == 1) ? 0 : tr_x[d - 1][OFF + prev_k];
        int32_t start_x = is_del ? (prev_x + 1) : prev_x;
        uint32_t snk = (uint32_t)(cx - start_x);
        if (snk > 0) ops[nops++] = snk;
        ops[nops++] = (is_del ? (1u << 30) : (2u << 30)) | 1u;
        cx = prev_x; ck = prev_k;
    }
    uint32_t co = aL, cp = bL;
    for (int i = nops - 1; i >= 0; i--) {
        uint32_t op = ops[i], kind = op >> 30, cnt = op & 0x3FFFFFFFu;
        if (kind == 0) { co += cnt; cp += cnt; }
        else if (kind == 1) { if (emit_chg_block(ws, co, cp, cnt, 0, total_del, total_ins) < 0) return -1; co += cnt; }
        else { if (emit_chg_block(ws, co, cp, 0, cnt, total_del, total_ins) < 0) return -1; cp += cnt; }
    }
    return 0;
}

__attribute__((target("avx2,bmi"), noinline))
static int run_big_myers_avx2(
    WS* ws, uint32_t aL, uint32_t n, uint32_t bL, uint32_t m,
    const uint16_t* a_ids, const uint16_t* b_ids,
    const uint16_t* last_a, const uint16_t* last_b,
    uint32_t* total_del, uint32_t* total_ins
) {
    uint32_t cur_o = aL, cur_p = bL;
    int32_t n_i = (int32_t)n, m_i = (int32_t)m;
    int32_t k_base = 0, k_lo = 0, k_hi = 0, seg_x = 0;
    size_t rel_d = 0, tr_ptr = 0;
    ws->v[ws->v_half] = 0;

    size_t max_edit = (size_t)n + m;
    for (size_t d = 1; d <= max_edit; d++) {
        rel_d++;
        int32_t next_lo = k_lo - 1, next_hi = k_hi + 1;
        int32_t nl_d = k_base - (next_lo - 1), nr_d = (next_hi + 1) - k_base;
        size_t need_half = (size_t)(nl_d > nr_d ? nl_d : nr_d) + 2;
        if (__builtin_expect(need_half > ws->v_half, 0) && ensure_v(ws, need_half) < 0) return -1;
        size_t row_len = (size_t)((next_hi - next_lo) >> 1) + 1;
        if (__builtin_expect(tr_ptr + row_len > ws->tr_cap || rel_d + 1 > ws->row_cap, 0)) {
            if (ensure_trace(ws, tr_ptr, tr_ptr + row_len, rel_d, rel_d + 1) < 0) return -1;
        }
        ws->row_base[rel_d] = (uint32_t)tr_ptr;
        ws->row_lo[rel_d] = next_lo;

        int32_t* v_zero = ws->v + (intptr_t)ws->v_half - k_base;
        int32_t* tr_xp = ws->tr_x + tr_ptr;
        uint8_t* tr_cp = ws->tr_c + tr_ptr;
        int32_t new_lo = next_hi + 1, new_hi = next_lo - 1;
        int32_t best_sum = -1, best_k = 0, best_x = 0, best_y = 0, best_snk = 0;
        int done = 0;

        int32_t* vp = v_zero + next_lo;
        for (int32_t k = next_lo; k <= next_hi; k += 2, vp += 2, tr_xp++, tr_cp++) {
            int32_t rem_x = vp[-1], add_x = vp[1];
            int can_rem = (rem_x != DEAD) & (rem_x < n_i);
            int can_add = (add_x != DEAD) & ((add_x - k) <= m_i);
            if (!(can_rem | can_add)) { *vp = DEAD; *tr_xp = DEAD; continue; }
            int32_t by_del = can_rem ? (rem_x + 1) : -2;
            int32_t by_ins = can_add ? add_x : -2;
            int32_t x; uint8_t is_del;
            if (by_del >= by_ins) { x = by_del; is_del = 1; } else { x = by_ins; is_del = 0; }
            int32_t sx = x, y = x - k;
            if (a_ids[x] == b_ids[y]) {
                x++; y++;
                for (;;) {
                    uint32_t eqm = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(
                        _mm256_loadu_si256((const __m256i*)(a_ids + x)),
                        _mm256_loadu_si256((const __m256i*)(b_ids + y))));
                    if (eqm == 0xFFFFFFFFu) { x += 16; y += 16; }
                    else { int32_t adv = (int32_t)(__builtin_ctz(~eqm) >> 1); x += adv; y += adv; break; }
                }
            }
            *vp = x; *tr_xp = x; *tr_cp = is_del;
            if (k < new_lo) new_lo = k;
            if (k > new_hi) new_hi = k;
            int32_t sum = x + y;
            if (sum > best_sum) { best_sum = sum; best_k = k; best_x = x; best_y = y; best_snk = x - sx; }
            if (x >= n_i && y >= m_i) { done = 1; break; }
        }
        tr_ptr += row_len;

        for (int32_t ck = k_lo; ck <= k_hi; ck += 2) v_zero[ck] = DEAD;
        if (done) {
            for (int32_t ck = new_lo; ck <= new_hi; ck += 2) v_zero[ck] = DEAD;
            return flush_segment(ws, rel_d, n_i, n_i - m_i, seg_x, &cur_o, &cur_p, total_del, total_ins);
        }

        if (best_snk > 0 || (rel_d & 7) == 0) {
            int32_t p_lo = best_k, p_hi = best_k;
            uint16_t by1 = (uint16_t)(best_y + 1);
            int32_t b_bad = 0, s = 0;
            for (int32_t k = best_k - 2; k >= new_lo; k -= 2) {
                int32_t i2 = best_x - s - 2;
                if (i2 < 0) {
                    for (int32_t q = new_lo; q <= k; q += 2) if (v_zero[q] != DEAD) { p_lo = q; break; }
                    break;
                }
                b_bad += (last_b[a_ids[i2 + 1]] >= by1) + (last_b[a_ids[i2]] >= by1);
                s += 2;
                int32_t x = v_zero[k];
                if (x != DEAD) {
                    if (x <= best_x && (x - k) <= best_y && (b_bad << 1) < s) v_zero[k] = DEAD;
                    else p_lo = k;
                }
            }
            uint16_t bx1 = (uint16_t)(best_x + 1);
            int32_t a_bad = 0; s = 0;
            for (int32_t k = best_k + 2; k <= new_hi; k += 2) {
                int32_t j2 = best_y - s - 2;
                if (j2 < 0) {
                    for (int32_t q = new_hi; q >= k; q -= 2) if (v_zero[q] != DEAD) { p_hi = q; break; }
                    break;
                }
                a_bad += (last_a[b_ids[j2 + 1]] >= bx1) + (last_a[b_ids[j2]] >= bx1);
                s += 2;
                int32_t x = v_zero[k];
                if (x != DEAD) {
                    if (x <= best_x && (x - k) <= best_y && (a_bad << 1) < s) v_zero[k] = DEAD;
                    else p_hi = k;
                }
            }
            k_lo = p_lo; k_hi = p_hi;
        } else {
            k_lo = new_lo; k_hi = new_hi;
        }

        if (k_lo == k_hi) {
            int32_t cx = v_zero[k_lo];
            v_zero[k_lo] = DEAD;
            if (flush_segment(ws, rel_d, cx, k_lo, seg_x, &cur_o, &cur_p, total_del, total_ins) < 0) return -1;
            seg_x = cx; k_base = k_lo;
            ws->v[ws->v_half] = cx;
            rel_d = 0; tr_ptr = 0;
        }
    }
    return 0;
}

SSE2_OPT
static int run_big_myers_sse2(
    WS* ws, uint32_t aL, uint32_t n, uint32_t bL, uint32_t m,
    const uint16_t* a_ids, const uint16_t* b_ids,
    const uint16_t* last_a, const uint16_t* last_b,
    uint32_t* total_del, uint32_t* total_ins
) {
    uint32_t cur_o = aL, cur_p = bL;
    int32_t n_i = (int32_t)n, m_i = (int32_t)m;
    int32_t k_base = 0, k_lo = 0, k_hi = 0, seg_x = 0;
    size_t rel_d = 0, tr_ptr = 0;
    ws->v[ws->v_half] = 0;

    size_t max_edit = (size_t)n + m;
    for (size_t d = 1; d <= max_edit; d++) {
        rel_d++;
        int32_t next_lo = k_lo - 1, next_hi = k_hi + 1;
        int32_t nl_d = k_base - (next_lo - 1), nr_d = (next_hi + 1) - k_base;
        size_t need_half = (size_t)(nl_d > nr_d ? nl_d : nr_d) + 2;
        if (need_half > ws->v_half && ensure_v(ws, need_half) < 0) return -1;
        size_t row_len = (size_t)((next_hi - next_lo) >> 1) + 1;
        if (tr_ptr + row_len > ws->tr_cap || rel_d + 1 > ws->row_cap) {
            if (ensure_trace(ws, tr_ptr, tr_ptr + row_len, rel_d, rel_d + 1) < 0) return -1;
        }
        ws->row_base[rel_d] = (uint32_t)tr_ptr;
        ws->row_lo[rel_d] = next_lo;

        int32_t* v_zero = ws->v + (intptr_t)ws->v_half - k_base;
        int32_t* tr_xp = ws->tr_x + tr_ptr;
        uint8_t* tr_cp = ws->tr_c + tr_ptr;
        int32_t new_lo = next_hi + 1, new_hi = next_lo - 1;
        int32_t best_sum = -1, best_k = 0, best_x = 0, best_y = 0, best_snk = 0;
        int done = 0;

        int32_t* vp = v_zero + next_lo;
        for (int32_t k = next_lo; k <= next_hi; k += 2, vp += 2, tr_xp++, tr_cp++) {
            int32_t rem_x = vp[-1], add_x = vp[1];
            int can_rem = (rem_x != DEAD) & (rem_x < n_i);
            int can_add = (add_x != DEAD) & ((add_x - k) <= m_i);
            if (!(can_rem | can_add)) { *vp = DEAD; *tr_xp = DEAD; continue; }
            int32_t by_del = can_rem ? (rem_x + 1) : -2;
            int32_t by_ins = can_add ? add_x : -2;
            int32_t x; uint8_t is_del;
            if (by_del >= by_ins) { x = by_del; is_del = 1; } else { x = by_ins; is_del = 0; }
            int32_t sx = x, y = x - k;
            while (a_ids[x] == b_ids[y]) { x++; y++; }
            *vp = x; *tr_xp = x; *tr_cp = is_del;
            if (k < new_lo) new_lo = k;
            if (k > new_hi) new_hi = k;
            int32_t sum = x + y;
            if (sum > best_sum) { best_sum = sum; best_k = k; best_x = x; best_y = y; best_snk = x - sx; }
            if (x >= n_i && y >= m_i) { done = 1; break; }
        }
        tr_ptr += row_len;

        for (int32_t ck = k_lo; ck <= k_hi; ck += 2) v_zero[ck] = DEAD;
        if (done) {
            for (int32_t ck = new_lo; ck <= new_hi; ck += 2) v_zero[ck] = DEAD;
            return flush_segment(ws, rel_d, n_i, n_i - m_i, seg_x, &cur_o, &cur_p, total_del, total_ins);
        }

        if (best_snk > 0 || (rel_d & 7) == 0) {
            int32_t p_lo = best_k, p_hi = best_k;
            uint16_t by1 = (uint16_t)(best_y + 1);
            int32_t b_bad = 0, s = 0;
            for (int32_t k = best_k - 2; k >= new_lo; k -= 2) {
                int32_t i2 = best_x - s - 2;
                if (i2 < 0) { for (int32_t q = new_lo; q <= k; q += 2) if (v_zero[q] != DEAD) { p_lo = q; break; } break; }
                b_bad += (last_b[a_ids[i2 + 1]] >= by1) + (last_b[a_ids[i2]] >= by1);
                s += 2;
                int32_t x = v_zero[k];
                if (x != DEAD) { if (x <= best_x && (x - k) <= best_y && (b_bad << 1) < s) v_zero[k] = DEAD; else p_lo = k; }
            }
            uint16_t bx1 = (uint16_t)(best_x + 1);
            int32_t a_bad = 0; s = 0;
            for (int32_t k = best_k + 2; k <= new_hi; k += 2) {
                int32_t j2 = best_y - s - 2;
                if (j2 < 0) { for (int32_t q = new_hi; q >= k; q -= 2) if (v_zero[q] != DEAD) { p_hi = q; break; } break; }
                a_bad += (last_a[b_ids[j2 + 1]] >= bx1) + (last_a[b_ids[j2]] >= bx1);
                s += 2;
                int32_t x = v_zero[k];
                if (x != DEAD) { if (x <= best_x && (x - k) <= best_y && (a_bad << 1) < s) v_zero[k] = DEAD; else p_hi = k; }
            }
            k_lo = p_lo; k_hi = p_hi;
        } else { k_lo = new_lo; k_hi = new_hi; }

        if (k_lo == k_hi) {
            int32_t cx = v_zero[k_lo]; v_zero[k_lo] = DEAD;
            if (flush_segment(ws, rel_d, cx, k_lo, seg_x, &cur_o, &cur_p, total_del, total_ins) < 0) return -1;
            seg_x = cx; k_base = k_lo; ws->v[ws->v_half] = cx; rel_d = 0; tr_ptr = 0;
        }
    }
    return 0;
}

__attribute__((noinline))
static int solve_subproblem(
    WS* ws, const uint8_t* base,
    uint32_t aL, uint32_t aR, uint32_t bL, uint32_t bR,
    uint32_t known_eq_suffix, int use_avx2,
    uint32_t* total_del, uint32_t* total_ins
#ifdef STANDALONE_TEST
    , uint64_t* stat_subs, uint64_t* stat_fast, uint64_t* stat_tiny, uint64_t* stat_myers_lines
#endif
) {
    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    const oh_t* oh = ws->old_h;

    while (aL < aR && bL < bR && lines_eq(base, ooff, aL, noff, bL)) { aL++; bL++; }

    uint32_t n = aR - aL;
    uint32_t m = bR - bL;
    if (n == 0 && m == 0) return 0;

    // Provably-safe unique-suffix trim (up to n,m <= 24 when known_eq_suffix > 0!)
    if (known_eq_suffix > 0 && n <= 24 && m <= 24) {
        while (known_eq_suffix > 0 && aL < aR && bL < bR) {
            uint32_t lh = oh[aR - 1];
            uint32_t ls = ooff[aR - 1], ll = ooff[aR] - ls;
            int dup = 0;
            for (uint32_t i = aL; i + 1 < aR; i++) {
                if (oh[i] == lh && (ooff[i + 1] - ooff[i]) == ll && eq_bytes_sse2(base + ooff[i], base + ls, ll)) { dup = 1; break; }
            }
            if (!dup) {
                for (uint32_t j = bL; j + 1 < bR; j++) {
                    if ((noff[j + 1] - noff[j]) == ll && eq_bytes_sse2(base + noff[j], base + ls, ll)) { dup = 1; break; }
                }
            }
            if (dup) break;
            aR--; bR--; known_eq_suffix--;
        }
        n = aR - aL; m = bR - bL;
        if (n == 0 && m == 0) return 0;
    }

#ifdef STANDALONE_TEST
    (*stat_subs)++;
#endif

    if (n == 0) {
#ifdef STANDALONE_TEST
        (*stat_fast)++;
#endif
        return emit_chg_block(ws, aL, bL, 0, m, total_del, total_ins);
    }
    if (m == 0) {
#ifdef STANDALONE_TEST
        (*stat_fast)++;
#endif
        return emit_chg_block(ws, aL, bL, n, 0, total_del, total_ins);
    }
    if (n == 1 && m == 1) {
#ifdef STANDALONE_TEST
        (*stat_fast)++;
#endif
        return emit_chg_block(ws, aL, bL, 1, 1, total_del, total_ins);
    }
    // EXACT O(m) solver for ALL n == 1 (disjoint OR matched: Myers always matches FIRST occurrence j >= 1!)
    if (n == 1) {
#ifdef STANDALONE_TEST
        (*stat_fast)++;
#endif
        uint32_t s = ooff[aL], l = ooff[aL + 1] - s;
        for (uint32_t j = 1; j < m; j++) {
            if ((noff[bL + j + 1] - noff[bL + j]) == l && eq_bytes_sse2(base + s, base + noff[bL + j], l)) {
#ifdef STANDALONE_TEST
                cnt_1x_match++;
#endif
                if (emit_chg_block(ws, aL, bL, 0, j, total_del, total_ins) < 0) return -1;
                if (j + 1 < m) return emit_chg_block(ws, aL + 1, bL + j + 1, 0, m - 1 - j, total_del, total_ins);
                return 0;
            }
        }
        return emit_chg_block(ws, aL, bL, 1, m, total_del, total_ins);
    }
    // EXACT O(n) solver for ALL m == 1 (disjoint OR matched: Myers always matches FIRST occurrence i >= 1!)
    if (m == 1) {
#ifdef STANDALONE_TEST
        (*stat_fast)++;
#endif
        uint32_t s = noff[bL], l = noff[bL + 1] - s;
        for (uint32_t i = 1; i < n; i++) {
            if ((ooff[aL + i + 1] - ooff[aL + i]) == l && eq_bytes_sse2(base + ooff[aL + i], base + s, l)) {
#ifdef STANDALONE_TEST
                cnt_1x_match++;
#endif
                if (emit_chg_block(ws, aL, bL, i, 0, total_del, total_ins) < 0) return -1;
                if (i + 1 < n) return emit_chg_block(ws, aL + i + 1, bL + 1, n - 1 - i, 0, total_del, total_ins);
                return 0;
            }
        }
        return emit_chg_block(ws, aL, bL, n, 1, total_del, total_ins);
    }

    if (n <= 12 && m <= 12) {
#ifdef STANDALONE_TEST
        (*stat_tiny)++;
        uint64_t _ct0 = _rdtsc();
        int r = solve_tiny_myers(ws, base, aL, n, bL, m, total_del, total_ins);
        cyc_sub_tiny += _rdtsc() - _ct0;
        return r;
#else
        return solve_tiny_myers(ws, base, aL, n, bL, m, total_del, total_ins);
#endif
    }

#ifdef STANDALONE_TEST
    *stat_myers_lines += (uint64_t)n + m;
    cnt_big++;
    uint64_t _cb0 = _rdtsc();
#endif

    if (n >= MAX_SUB_LINES || m >= MAX_SUB_LINES) return -1;
    size_t ht_sz = 64;
    while (ht_sz < (size_t)n * 2) ht_sz <<= 1;
    if (ht_sz > ws->ht_cap) {
        uint32_t* np = (uint32_t*)arena_alloc(&ws->arena, ht_sz * 4, 32);
        if (!np) return -1;
        ws->ht = np; ws->ht_cap = ht_sz;
    }
    if ((size_t)n + 4 > ws->ent_cap) {
        size_t nc = ws->ent_cap ? ws->ent_cap * 2 : 1024;
        while (nc < (size_t)n + 4) nc *= 2;
        uint32_t* no = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        uint32_t* nl = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        if (!no || !nl) return -1;
        ws->ent_off = no; ws->ent_len = nl; ws->ent_cap = nc;
    }
    if (n + 24 > ws->a_ids_cap && ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, 0, n + 24) < 0) return -1;
    if (m + 24 > ws->b_ids_cap && ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, 0, m + 24) < 0) return -1;

    uint32_t* ht = ws->ht;
    mem_zero_sse2(ht, ht_sz * 4);
    size_t mask = ht_sz - 1;
    uint32_t* eoff = ws->ent_off; uint32_t* elen = ws->ent_len;
    uint16_t* a_ids = ws->a_ids; uint16_t* b_ids = ws->b_ids;
    size_t next_id = 1;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t s = ooff[aL + i], l = ooff[aL + i + 1] - s;
        uint64_t h = hash_64(base + s, l);
        uint32_t tag = (uint32_t)(h >> 32) & 0xFFFF0000u;
        size_t slot = (size_t)h & mask;
        for (;;) {
            uint32_t ent = ht[slot];
            if (ent == 0) {
                uint32_t id = (uint32_t)next_id++;
                ht[slot] = tag | id; eoff[id] = s; elen[id] = l;
                a_ids[i] = (uint16_t)id; break;
            }
            if ((ent & 0xFFFF0000u) == tag) {
                uint32_t id = ent & 0xFFFFu;
                if (elen[id] == l && eq_bytes_sse2(base + eoff[id], base + s, l)) { a_ids[i] = (uint16_t)id; break; }
            }
            slot = (slot + 1) & mask;
        }
    }
    int any_match = 0;
    for (uint32_t j = 0; j < m; j++) {
        uint32_t s = noff[bL + j], l = noff[bL + j + 1] - s;
        uint64_t h = hash_64(base + s, l);
        uint32_t tag = (uint32_t)(h >> 32) & 0xFFFF0000u;
        size_t slot = (size_t)h & mask;
        uint16_t matched_id = 0;
        for (;;) {
            uint32_t ent = ht[slot];
            if (ent == 0) break;
            if ((ent & 0xFFFF0000u) == tag) {
                uint32_t id = ent & 0xFFFFu;
                if (elen[id] == l && eq_bytes_sse2(base + eoff[id], base + s, l)) { matched_id = (uint16_t)id; any_match = 1; break; }
            }
            slot = (slot + 1) & mask;
        }
        b_ids[j] = matched_id;
    }
    if (!any_match) {
#ifdef STANDALONE_TEST
        cyc_sub_big += _rdtsc() - _cb0;
#endif
        return emit_chg_block(ws, aL, bL, n, m, total_del, total_ins);
    }
    a_ids[n] = 0xFFFEu; b_ids[m] = 0xFFFFu;

    if (next_id > ws->last_cap) {
        size_t nc = ws->last_cap ? ws->last_cap * 2 : 1024;
        while (nc < next_id) nc *= 2;
        uint16_t* la = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        uint16_t* lb = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        if (!la || !lb) return -1;
        ws->last_a = la; ws->last_b = lb; ws->last_cap = nc;
    }
    uint16_t* last_a = ws->last_a; uint16_t* last_b = ws->last_b;
    mem_zero_sse2(last_a, next_id * 2);
    mem_zero_sse2(last_b, next_id * 2);
    for (uint32_t i = 0; i < n; i++) last_a[a_ids[i]] = (uint16_t)(i + 1);
    for (uint32_t j = 0; j < m; j++) if (b_ids[j]) last_b[b_ids[j]] = (uint16_t)(j + 1);

    if (ensure_trace(ws, 0, 512, 0, 128) < 0) return -1;
    int r = use_avx2 ? run_big_myers_avx2(ws, aL, n, bL, m, a_ids, b_ids, last_a, last_b, total_del, total_ins)
                     : run_big_myers_sse2(ws, aL, n, bL, m, a_ids, b_ids, last_a, last_b, total_del, total_ins);
#ifdef STANDALONE_TEST
    cyc_sub_big += _rdtsc() - _cb0;
#endif
    return r;
}

// ---------------------------------------------------------------------------
// Exact O(N) Kadane-Phi Bridge Decomposition over Synced Runs
// ---------------------------------------------------------------------------

__attribute__((noinline))
static int run_kadane_phi_diff(
    WS* ws, const uint8_t* base, int use_avx2,
    uint32_t* out_del, uint32_t* out_ins
#ifdef STANDALONE_TEST
    , uint64_t* stat_subs, uint64_t* stat_fast, uint64_t* stat_tiny, uint64_t* stat_myers_lines, uint64_t* stat_cuts,
    double* pt_cnt, double* pt_sub
#endif
) {
    ws->chg_len = 0;
    *out_del = 0;
    *out_ins = 0;

    uint32_t N = (uint32_t)ws->a_len;
    uint32_t M = (uint32_t)ws->b_len;

    uint32_t min_nm = N < M ? N : M;
    uint32_t prefix = 0;
    if (ws->runs_len > 0 && ws->runs[0].a0 == 0 && ws->runs[0].b0 == 0) prefix = ws->runs[0].len;
    while (prefix < min_nm && lines_eq(base, ws->old_off, prefix, ws->new_off, prefix)) prefix++;

    uint32_t rem_a = N - prefix, rem_b = M - prefix;
    uint32_t min_rem = rem_a < rem_b ? rem_a : rem_b;
    uint32_t suffix = 0;
    if (ws->runs_len > 0) {
        const SyncRun* lr = &ws->runs[ws->runs_len - 1];
        if (lr->a0 + lr->len == N && lr->b0 + lr->len == M) suffix = lr->len < min_rem ? lr->len : min_rem;
    }
    while (suffix < min_rem && lines_eq(base, ws->old_off, N - 1 - suffix, ws->new_off, M - 1 - suffix)) suffix++;

    uint32_t a_lo = prefix, a_hi = N - suffix;
    uint32_t b_lo = prefix, b_hi = M - suffix;
    if (a_lo == a_hi && b_lo == b_hi) return 0;
    if (a_lo == a_hi) return emit_chg_block(ws, a_lo, b_lo, 0, b_hi - b_lo, out_del, out_ins);
    if (b_lo == b_hi) return emit_chg_block(ws, a_lo, b_lo, a_hi - a_lo, 0, out_del, out_ins);

    size_t R = 0;
    for (size_t i = 0; i < ws->runs_len; i++) {
        uint32_t a0 = ws->runs[i].a0, b0 = ws->runs[i].b0, len = ws->runs[i].len;
        if (a0 >= a_hi || b0 >= b_hi) continue;
        uint32_t d_lo = 0;
        if (a0 < a_lo) { uint32_t d = a_lo - a0; if (d > d_lo) d_lo = d; }
        if (b0 < b_lo) { uint32_t d = b_lo - b0; if (d > d_lo) d_lo = d; }
        uint32_t d_hi = len;
        if (a0 + d_hi > a_hi) d_hi = a_hi - a0;
        if (b0 + d_hi > b_hi) d_hi = b_hi - b0;
        if (d_lo < d_hi) ws->runs[R++] = (SyncRun){ a0 + d_lo, b0 + d_lo, d_hi - d_lo };
    }
    ws->runs_len = R;

    if (R == 0) {
        return solve_subproblem(ws, base, a_lo, a_hi, b_lo, b_hi, 0, use_avx2, out_del, out_ins
#ifdef STANDALONE_TEST
            , stat_subs, stat_fast, stat_tiny, stat_myers_lines
#endif
        );
    }

    if (ensure_rphi(ws, R) < 0) return -1;
    const SyncRun* runs = ws->runs;
    const oh_t* oh = ws->old_h;
    const uint64_t* seen = ws->seen;
    RunPhi* phi = ws->rphi;

#ifdef STANDALONE_TEST
    double _c0 = _t_ms();
#endif
    uint32_t pA = 0, pB = 0, pUV = 0;
    uint32_t prev_a = a_lo, prev_b = b_lo;
    for (size_t r = 0; r < R; r++) {
        uint32_t a0 = runs[r].a0, b0 = runs[r].b0, len = runs[r].len;
        uint32_t u = a0 - prev_a, v = b0 - prev_b;
        pA += u; pB += v; pUV += u + v;
        phi[r].preA = pA; phi[r].preB = pB; phi[r].preUV = pUV;

        uint32_t U = 0;
        const oh_t* hp = oh + a0;
        for (uint32_t i = 0; i < len; i++) U += is_uniq_2b(seen, hp[i]);
        phi[r].uniq = U;

        pA  = (pA  > U)     ? (pA  - U)     : 0;
        pB  = (pB  > U)     ? (pB  - U)     : 0;
        pUV = (pUV > 2 * U) ? (pUV - 2 * U) : 0;
        prev_a = a0 + len; prev_b = b0 + len;
    }

    uint32_t sA = a_hi - prev_a, sB = b_hi - prev_b, sUV = sA + sB;
    for (size_t ri = R; ri > 0; ri--) {
        size_t r = ri - 1;
        phi[r].sufA = sA; phi[r].sufB = sB; phi[r].sufUV = sUV;
        uint32_t U = phi[r].uniq;
        sA  = (sA  > U)     ? (sA  - U)     : 0;
        sB  = (sB  > U)     ? (sB  - U)     : 0;
        sUV = (sUV > 2 * U) ? (sUV - 2 * U) : 0;
        uint32_t pa = (r == 0) ? a_lo : (runs[r - 1].a0 + runs[r - 1].len);
        uint32_t pb = (r == 0) ? b_lo : (runs[r - 1].b0 + runs[r - 1].len);
        uint32_t u = runs[r].a0 - pa, v = runs[r].b0 - pb;
        sA += u; sB += v; sUV += u + v;
    }
#ifdef STANDALONE_TEST
    *pt_cnt += _t_ms() - _c0;
    double _sb0 = _t_ms();
#endif

    uint32_t cur_a = a_lo, cur_b = b_lo;
    for (size_t r = 0; r < R; r++) {
        uint32_t U = phi[r].uniq;
        if (U == 0) continue;

        int32_t k_min = (int32_t)U, k_max = -1;
        if (phi[r].preA + phi[r].sufA < U) {
            int32_t lo = (int32_t)phi[r].preA, hi = (int32_t)(U - 1 - phi[r].sufA);
            if (lo < k_min) k_min = lo;
            if (hi > k_max) k_max = hi;
        }
        if (phi[r].preB + phi[r].sufB < U) {
            int32_t lo = (int32_t)phi[r].preB, hi = (int32_t)(U - 1 - phi[r].sufB);
            if (lo < k_min) k_min = lo;
            if (hi > k_max) k_max = hi;
        }
        if (phi[r].preUV + phi[r].sufUV < 2 * U) {
            int32_t lo = (int32_t)(phi[r].preUV >> 1), hi = (int32_t)(U - 1 - (phi[r].sufUV >> 1));
            if (lo < k_min) k_min = lo;
            if (hi > k_max) k_max = hi;
        }
        if (k_min > k_max) continue;

#ifdef STANDALONE_TEST
        (*stat_cuts)++;
#endif
        uint32_t a0 = runs[r].a0, b0 = runs[r].b0, len = runs[r].len;
        const oh_t* hp = oh + a0;

        uint32_t off1 = 0;
        {
            int32_t cnt = 0;
            for (uint32_t i = 0; i < len; i++) {
                if (is_uniq_2b(seen, hp[i])) {
                    if (cnt == k_min) { off1 = i; break; }
                    cnt++;
                }
            }
        }

        uint32_t cut_a1 = a0 + off1, cut_b1 = b0 + off1;
        if (cur_a < cut_a1 || cur_b < cut_b1) {
            if (solve_subproblem(ws, base, cur_a, cut_a1, cur_b, cut_b1, off1, use_avx2, out_del, out_ins
#ifdef STANDALONE_TEST
                , stat_subs, stat_fast, stat_tiny, stat_myers_lines
#endif
            ) < 0) return -1;
        }
        cur_a = a0 + len;
        cur_b = b0 + len;
    }

    if (cur_a < a_hi || cur_b < b_hi) {
        if (solve_subproblem(ws, base, cur_a, a_hi, cur_b, b_hi, 0, use_avx2, out_del, out_ins
#ifdef STANDALONE_TEST
            , stat_subs, stat_fast, stat_tiny, stat_myers_lines
#endif
        ) < 0) return -1;
    }
#ifdef STANDALONE_TEST
    *pt_sub += _t_ms() - _sb0;
#endif
    return 0;
}

// ---------------------------------------------------------------------------
// Branchless-Overcopy JSON Hunk Formatter (AVX2 & compact -Os SSE2)
// ---------------------------------------------------------------------------

static const char DIGITS2[200] =
    "0001020304050607080910111213141516171819"
    "2021222324252627282930313233343536373839"
    "4041424344454647484950515253545556575859"
    "6061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

static inline uint8_t* write_u32_dec(uint8_t* dst, uint32_t x) {
    if (x < 10) { *dst = (uint8_t)('0' + x); return dst + 1; }
    if (x < 100) { dst[0] = (uint8_t)DIGITS2[x * 2]; dst[1] = (uint8_t)DIGITS2[x * 2 + 1]; return dst + 2; }
    char tmp[10]; int p = 10;
    while (x >= 100) { uint32_t r = x % 100; x /= 100; p -= 2; tmp[p] = DIGITS2[r * 2]; tmp[p + 1] = DIGITS2[r * 2 + 1]; }
    if (x >= 10) { p -= 2; tmp[p] = DIGITS2[x * 2]; tmp[p + 1] = DIGITS2[x * 2 + 1]; }
    else { tmp[--p] = (char)('0' + x); }
    for (int i = p; i < 10; i++) *dst++ = (uint8_t)tmp[i];
    return dst;
}

__attribute__((target("avx2"), always_inline))
static inline uint8_t* emit_range_avx2(uint8_t* dst, const uint8_t* base, const uint32_t* offs, uint32_t from, uint32_t to, uint8_t tag, int fast_ok, uint32_t bare_idx) {
    if (__builtin_expect(fast_ok, 1)) {
        for (uint32_t i = from; i < to; i++) {
            uint32_t s = offs[i], e = offs[i + 1];
            *dst = tag;
            dst = copy_line_over_avx2(dst + 1, base + s, e - s);
        }
    } else {
        for (uint32_t i = from; i < to; i++) {
            uint32_t s = offs[i], e = offs[i + 1];
            *dst++ = tag;
            dst = copy_short_avx2(dst, base + s, e - s);
            if (__builtin_expect(i == bare_idx, 0)) {
                dst[0] = '\\'; dst[1] = 'n'; dst[2] = '%'; dst[3] = '\\'; dst[4] = 'n'; dst += 5;
            }
        }
    }
    return dst;
}

static int format_header(WS* ws, uint32_t total_del, uint32_t total_ins) {
    uint32_t n0 = (uint32_t)ws->a_len, m0 = (uint32_t)ws->b_len;
    size_t n_chg = ws->chg_len;
    const Chg* chg = ws->chg;
    uint32_t num_hunks = 0;
    if (n_chg > 0) {
        num_hunks = 1;
        for (size_t ci = 1; ci < n_chg; ci++) {
            if (chg[ci].o - (chg[ci - 1].o + chg[ci - 1].dels) > 6) num_hunks++;
        }
    }
    if (ensure_out(ws, ws->out_len + 128) < 0) return -1;
    uint8_t* dst = ws->out + ws->out_len;
    dst = copy_short_sse2(dst, (const uint8_t*)"makefun-diff/1 old=", 19);
    dst = write_u32_dec(dst, n0);
    dst = copy_short_sse2(dst, (const uint8_t*)" new=", 5);
    dst = write_u32_dec(dst, m0);
    dst = copy_short_sse2(dst, (const uint8_t*)" hunks=", 7);
    dst = write_u32_dec(dst, num_hunks);
    dst = copy_short_sse2(dst, (const uint8_t*)" deleted=", 9);
    dst = write_u32_dec(dst, total_del);
    dst = copy_short_sse2(dst, (const uint8_t*)" inserted=", 10);
    dst = write_u32_dec(dst, total_ins);
    *dst++ = '\\'; *dst++ = 'n';
    ws->out_len = (size_t)(dst - ws->out);
    return 0;
}

__attribute__((target("avx2"), noinline))
static int format_hunks_json_avx2(WS* ws, const uint8_t* base, size_t file_len, int old_bare, int new_bare, uint32_t total_del, uint32_t total_ins) {
    if (format_header(ws, total_del, total_ins) < 0) return -1;
    size_t n_chg = ws->chg_len;
    if (n_chg == 0) return 0;
    uint32_t n0 = (uint32_t)ws->a_len, m0 = (uint32_t)ws->b_len;
    const Chg* chg = ws->chg;
    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    size_t safe_lim = file_len > 64 ? file_len - 64 : 0;
    uint32_t old_bare_idx = (old_bare && n0 > 0) ? (n0 - 1) : 0xFFFFFFFFu;
    uint32_t new_bare_idx = (new_bare && m0 > 0) ? (m0 - 1) : 0xFFFFFFFFu;

    size_t h_start = 0;
    while (h_start < n_chg) {
        size_t h_end = h_start + 1;
        while (h_end < n_chg && chg[h_end].o - (chg[h_end - 1].o + chg[h_end - 1].dels) <= 6) h_end++;
        uint32_t old_start = chg[h_start].o > 3 ? chg[h_start].o - 3 : 0;
        uint32_t new_start = chg[h_start].p - (chg[h_start].o - old_start);
        uint32_t last_o_end = chg[h_end - 1].o + chg[h_end - 1].dels;
        uint32_t last_p_end = chg[h_end - 1].p + chg[h_end - 1].ins;
        uint32_t rem = n0 - last_o_end;
        uint32_t after = rem < 3 ? rem : 3;
        uint32_t old_end = last_o_end + after;
        uint32_t new_end = last_p_end + after;

        size_t need = ws->out_len + 128 + (size_t)(ooff[old_end] - ooff[old_start]) + (size_t)(noff[new_end] - noff[new_start]) + (size_t)((old_end - old_start) + (new_end - new_start));
        if (__builtin_expect(need > ws->out_cap, 0) && ensure_out_grow(ws, need) < 0) return -1;
        uint8_t* dst = ws->out + ws->out_len;

        *dst++ = '@'; *dst++ = ' ';
        dst = write_u32_dec(dst, old_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, old_end - old_start); *dst++ = ' ';
        dst = write_u32_dec(dst, new_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, new_end - new_start);
        *dst++ = '\\'; *dst++ = 'n';

        int fast_ok = !(old_bare && old_end == n0) && !(new_bare && new_end == m0)
                   && (size_t)ooff[old_end] <= safe_lim && (size_t)noff[new_end] <= safe_lim;

        uint32_t at = old_start;
        for (size_t ci = h_start; ci < h_end; ci++) {
            const Chg* c = &chg[ci];
            if (at < c->o) dst = emit_range_avx2(dst, base, ooff, at, c->o, '=', fast_ok, old_bare_idx);
            if (c->dels > 0) dst = emit_range_avx2(dst, base, ooff, c->o, c->o + c->dels, '-', fast_ok, old_bare_idx);
            if (c->ins > 0) dst = emit_range_avx2(dst, base, noff, c->p, c->p + c->ins, '+', fast_ok, new_bare_idx);
            at = c->o + c->dels;
        }
        if (at < old_end) dst = emit_range_avx2(dst, base, ooff, at, old_end, '=', fast_ok, old_bare_idx);
        ws->out_len = (size_t)(dst - ws->out);
        h_start = h_end;
    }
    return 0;
}

SSE2_OPT
static uint8_t* emit_range_sse2(uint8_t* dst, const uint8_t* base, const uint32_t* offs, uint32_t from, uint32_t to, uint8_t tag, uint32_t bare_idx) {
    for (uint32_t i = from; i < to; i++) {
        uint32_t s = offs[i], e = offs[i + 1];
        *dst++ = tag;
        dst = copy_short_sse2(dst, base + s, e - s);
        if (i == bare_idx) { dst[0] = '\\'; dst[1] = 'n'; dst[2] = '%'; dst[3] = '\\'; dst[4] = 'n'; dst += 5; }
    }
    return dst;
}

SSE2_OPT
static int format_hunks_json_sse2(WS* ws, const uint8_t* base, size_t file_len, int old_bare, int new_bare, uint32_t total_del, uint32_t total_ins) {
    (void)file_len;
    if (format_header(ws, total_del, total_ins) < 0) return -1;
    size_t n_chg = ws->chg_len;
    if (n_chg == 0) return 0;
    uint32_t n0 = (uint32_t)ws->a_len, m0 = (uint32_t)ws->b_len;
    const Chg* chg = ws->chg;
    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    uint32_t old_bare_idx = (old_bare && n0 > 0) ? (n0 - 1) : 0xFFFFFFFFu;
    uint32_t new_bare_idx = (new_bare && m0 > 0) ? (m0 - 1) : 0xFFFFFFFFu;

    size_t h_start = 0;
    while (h_start < n_chg) {
        size_t h_end = h_start + 1;
        while (h_end < n_chg && chg[h_end].o - (chg[h_end - 1].o + chg[h_end - 1].dels) <= 6) h_end++;
        uint32_t old_start = chg[h_start].o > 3 ? chg[h_start].o - 3 : 0;
        uint32_t new_start = chg[h_start].p - (chg[h_start].o - old_start);
        uint32_t last_o_end = chg[h_end - 1].o + chg[h_end - 1].dels;
        uint32_t last_p_end = chg[h_end - 1].p + chg[h_end - 1].ins;
        uint32_t rem = n0 - last_o_end;
        uint32_t after = rem < 3 ? rem : 3;
        uint32_t old_end = last_o_end + after;
        uint32_t new_end = last_p_end + after;

        size_t need = ws->out_len + 128 + (size_t)(ooff[old_end] - ooff[old_start]) + (size_t)(noff[new_end] - noff[new_start]) + (size_t)((old_end - old_start) + (new_end - new_start));
        if (need > ws->out_cap && ensure_out_grow(ws, need) < 0) return -1;
        uint8_t* dst = ws->out + ws->out_len;

        *dst++ = '@'; *dst++ = ' ';
        dst = write_u32_dec(dst, old_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, old_end - old_start); *dst++ = ' ';
        dst = write_u32_dec(dst, new_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, new_end - new_start);
        *dst++ = '\\'; *dst++ = 'n';

        uint32_t at = old_start;
        for (size_t ci = h_start; ci < h_end; ci++) {
            const Chg* c = &chg[ci];
            if (at < c->o) dst = emit_range_sse2(dst, base, ooff, at, c->o, '=', old_bare_idx);
            if (c->dels > 0) dst = emit_range_sse2(dst, base, ooff, c->o, c->o + c->dels, '-', old_bare_idx);
            if (c->ins > 0) dst = emit_range_sse2(dst, base, noff, c->p, c->p + c->ins, '+', new_bare_idx);
            at = c->o + c->dels;
        }
        if (at < old_end) dst = emit_range_sse2(dst, base, ooff, at, old_end, '=', old_bare_idx);
        ws->out_len = (size_t)(dst - ws->out);
        h_start = h_end;
    }
    return 0;
}

static int flush_out_fd(long fd, WS* ws) {
    size_t sent = 0;
    while (sent < ws->out_len) {
        long w = sys3(SYS_write, fd, (long)(ws->out + sent), (long)(ws->out_len - sent));
        if (w <= 0) return -1;
        sent += (size_t)w;
    }
    ws->out_len = 0;
    return 0;
}

// ---------------------------------------------------------------------------
// Top-level File Runner
// ---------------------------------------------------------------------------

static int run_cases_file_impl(const char* in_path, const char* out_path, int force_sse2, int exit_on_ok) {
#ifdef STANDALONE_TEST
    double pt_find = 0, pt_sync = 0, pt_cnt = 0, pt_sub = 0, pt_fmt = 0, pt_flush = 0;
    uint64_t stat_subs = 0, stat_fast = 0, stat_tiny = 0, stat_myers_lines = 0, stat_cuts = 0, stat_runs = 0;
    cyc_run = cyc_fastgap = cyc_gap = cyc_sub_tiny = cyc_sub_big = 0;
    cnt_fg = cnt_sg = cnt_big = cnt_1x_match = 0;
#endif
    long in_fd = sys4(SYS_openat, AT_FDCWD, (long)in_path, O_RDONLY, 0);
    if (in_fd < 0) return -1;
    long sz_l = sys3(SYS_lseek, in_fd, 0, SEEK_END);
    if (sz_l <= 0) { sys1(SYS_close, in_fd); return -1; }
    size_t in_len = (size_t)sz_l;

    uint8_t* mmap_base = NULL;
    size_t mmap_len = 0;
    uint8_t* base = NULL;
    {
        long mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_SHARED | MAP_POPULATE, in_fd, 0);
        if (mp < 0 && mp > -4096) mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, in_fd, 0);
        if (mp < 0 && mp > -4096) mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_PRIVATE, in_fd, 0);
        if (!(mp < 0 && mp > -4096)) { mmap_base = (uint8_t*)mp; mmap_len = in_len; base = mmap_base; }
    }

    size_t est_lines = in_len / 240;
    if (est_lines < 2048) est_lines = 2048;
    if (est_lines > 44000) est_lines = 44000;

    size_t arena_sz = est_lines * (8 + sizeof(oh_t)) + (BF_WORDS * 8) + (500u << 10);
    if (arena_sz < (640u << 10)) arena_sz = (640u << 10);

    WS ws;
    mem_zero_sse2(&ws, sizeof(ws));
    if (!base) arena_sz += in_len + 64;
    if (arena_init(&ws.arena, arena_sz) < 0) {
        if (mmap_base) os_munmap(mmap_base, mmap_len);
        sys1(SYS_close, in_fd);
        return -1;
    }

    if (!base) {
        base = (uint8_t*)arena_alloc(&ws.arena, in_len + 64, 32);
        if (!base) { arena_free(&ws.arena); sys1(SYS_close, in_fd); return -1; }
        size_t got = 0;
        while (got < in_len) {
            long r = sys4(SYS_pread64, in_fd, (long)(base + got), (long)(in_len - got), (long)got);
            if (r <= 0) break;
            got += (size_t)r;
        }
        if (got != in_len) { arena_free(&ws.arena); sys1(SYS_close, in_fd); return -1; }
        mem_zero_sse2(base + in_len, 64);
    }
    sys1(SYS_close, in_fd);

    int use_avx2 = force_sse2 ? 0 : cpu_has_avx2();

    long out_fd = sys4(SYS_openat, AT_FDCWD, (long)out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        if (mmap_base) os_munmap(mmap_base, mmap_len);
        arena_free(&ws.arena);
        return -1;
    }
    sys2(SYS_fchmod, out_fd, 0644);

    if (ensure_u32(&ws.arena, &ws.old_off, &ws.old_off_cap, 0, est_lines + 64) < 0 ||
        ensure_u32(&ws.arena, &ws.new_off, &ws.new_off_cap, 0, est_lines + 64) < 0 ||
        ensure_chg(&ws, 4096) < 0 ||
        ensure_v(&ws, 1024) < 0) {
        goto fail;
    }

    size_t scratch_off = (ws.arena.used + 63) & ~(size_t)63;
    ws.arena.used = scratch_off;
    ws.scratch_base = (uint8_t*)ws.arena.blocks[0] + scratch_off;
    ws.seen = (uint64_t*)arena_alloc(&ws.arena, BF_WORDS * 8, 64);
    if (!ws.seen) goto fail;
    if (ensure_oh(&ws.arena, &ws.old_h, &ws.old_h_cap, 0, est_lines + 64) < 0 ||
        ensure_runs(&ws, 4096) < 0 ||
        ensure_rphi(&ws, 4096) < 0) {
        goto fail;
    }
    ws.scratch_cap = (ws.arena.n_blocks == 1) ? (ws.arena.cap - scratch_off) : 0;

    size_t p = skip_ws(base, 0, in_len);
    if (p >= in_len || base[p] != '[') goto fail;
    p = skip_ws(base, p + 1, in_len);
    if (p < in_len && base[p] == ']') {
        if (skip_ws(base, p + 1, in_len) != in_len) goto fail;
        uint8_t empty_arr[2] = { '[', ']' };
        if (sys3(SYS_write, out_fd, (long)empty_arr, 2) != 2) goto fail;
        sys1(SYS_close, out_fd);
        if (exit_on_ok) sys1(SYS_exit_group, 0);
        if (mmap_base) os_munmap(mmap_base, mmap_len);
        arena_free(&ws.arena);
        return 0;
    }

    size_t case_idx = 0;
    for (;;) {
        p = skip_ws(base, p, in_len);
        if (p >= in_len || base[p] != '{') goto fail;
        p = skip_ws(base, p + 1, in_len);
        int solved = 0;
        int old_bare = 0, new_bare = 0;

        while (p < in_len && base[p] != '}') {
            if (base[p] != '"') goto fail;
            size_t ks = p + 1;
            size_t ke = use_avx2 ? find_quote_end_avx2(base, ks, in_len) : find_quote_end_sse2(base, ks, in_len);
            if (ke == (size_t)-1) goto fail;
            size_t klen = ke - ks;
            p = skip_ws(base, ke + 1, in_len);
            if (p >= in_len || base[p] != ':') goto fail;
            p = skip_ws(base, p + 1, in_len);

            if (klen == 5 && eq_bytes_sse2(base + ks, (const uint8_t*)"input", 5)) {
                if (solved || p >= in_len || base[p] != '{') goto fail;
                p = skip_ws(base, p + 1, in_len);
                size_t old_s = 0, old_e = 0, new_s = 0, new_e = (size_t)-1;
                int has_old = 0, has_new = 0, synced = 0;

                while (p < in_len && base[p] != '}') {
                    if (base[p] != '"') goto fail;
                    size_t iks = p + 1;
                    size_t ike = use_avx2 ? find_quote_end_avx2(base, iks, in_len) : find_quote_end_sse2(base, iks, in_len);
                    if (ike == (size_t)-1) goto fail;
                    size_t iklen = ike - iks;
                    p = skip_ws(base, ike + 1, in_len);
                    if (p >= in_len || base[p] != ':') goto fail;
                    p = skip_ws(base, p + 1, in_len);

                    if (iklen == 3 && base[iks] == 'o' && base[iks + 1] == 'l' && base[iks + 2] == 'd') {
                        if (has_old || p >= in_len || base[p] != '"') goto fail;
#ifdef STANDALONE_TEST
                        double _f0 = _t_ms();
#endif
                        old_s = p + 1;
                        old_e = use_avx2 ? find_quote_end_avx2(base, old_s, in_len) : find_quote_end_sse2(base, old_s, in_len);
#ifdef STANDALONE_TEST
                        pt_find += _t_ms() - _f0;
#endif
                        if (old_e == (size_t)-1) goto fail;
                        p = old_e + 1; has_old = 1;
                        if (has_new) {
#ifdef STANDALONE_TEST
                            double _s0 = _t_ms();
#endif
                            size_t rne = use_avx2 ? block_sync_avx2(&ws, base, in_len, old_s, old_e, new_s, new_e, &old_bare, &new_bare)
                                                  : block_sync_sse2(&ws, base, in_len, old_s, old_e, new_s, new_e, &old_bare, &new_bare);
#ifdef STANDALONE_TEST
                            pt_sync += _t_ms() - _s0;
#endif
                            if (rne == (size_t)-1) goto fail;
                            synced = 1;
                        }
                    } else if (iklen == 3 && base[iks] == 'n' && base[iks + 1] == 'e' && base[iks + 2] == 'w') {
                        if (has_new || p >= in_len || base[p] != '"') goto fail;
                        new_s = p + 1; has_new = 1;
                        if (has_old) {
#ifdef STANDALONE_TEST
                            double _s0 = _t_ms();
#endif
                            size_t rne = use_avx2 ? block_sync_avx2(&ws, base, in_len, old_s, old_e, new_s, (size_t)-1, &old_bare, &new_bare)
                                                  : block_sync_sse2(&ws, base, in_len, old_s, old_e, new_s, (size_t)-1, &old_bare, &new_bare);
#ifdef STANDALONE_TEST
                            pt_sync += _t_ms() - _s0;
#endif
                            if (rne == (size_t)-1) goto fail;
                            new_e = rne; p = new_e + 1; synced = 1;
                        } else {
                            new_e = use_avx2 ? find_quote_end_avx2(base, new_s, in_len) : find_quote_end_sse2(base, new_s, in_len);
                            if (new_e == (size_t)-1) goto fail;
                            p = new_e + 1;
                        }
                    } else {
                        p = skip_json_value(base, p, in_len, use_avx2);
                        if (p == (size_t)-1) goto fail;
                    }
                    p = skip_ws(base, p, in_len);
                    if (p < in_len && base[p] == ',') p = skip_ws(base, p + 1, in_len);
                }
                if (p >= in_len || base[p] != '}' || !synced) goto fail;
                p++; solved = 1;
            } else {
                p = skip_json_value(base, p, in_len, use_avx2);
                if (p == (size_t)-1) goto fail;
            }
            p = skip_ws(base, p, in_len);
            if (p < in_len && base[p] == ',') p = skip_ws(base, p + 1, in_len);
        }
        if (!solved || p >= in_len || base[p] != '}') goto fail;
        p = skip_ws(base, p + 1, in_len);
#ifdef STANDALONE_TEST
        stat_runs += ws.runs_len;
#endif

        uint32_t td = 0, ti = 0;
        int krc = run_kadane_phi_diff(&ws, base, use_avx2, &td, &ti
#ifdef STANDALONE_TEST
            , &stat_subs, &stat_fast, &stat_tiny, &stat_myers_lines, &stat_cuts, &pt_cnt, &pt_sub
#endif
        );
        if (krc < 0) goto fail;

        ws.out = ws.scratch_base;
        ws.out_cap = ws.scratch_cap;
        ws.out_len = 0;
        if (ensure_out(&ws, 4) < 0) goto fail;
        ws.out[ws.out_len++] = (case_idx == 0) ? '[' : ',';
        ws.out[ws.out_len++] = '"';
#ifdef STANDALONE_TEST
        double _fm0 = _t_ms();
#endif
        int frc = use_avx2 ? format_hunks_json_avx2(&ws, base, in_len, old_bare, new_bare, td, ti)
                           : format_hunks_json_sse2(&ws, base, in_len, old_bare, new_bare, td, ti);
#ifdef STANDALONE_TEST
        pt_fmt += _t_ms() - _fm0;
#endif
        if (frc < 0) goto fail;
        if (ensure_out(&ws, ws.out_len + 3) < 0) goto fail;
        ws.out[ws.out_len++] = '"';

        case_idx++;
        if (p < in_len && base[p] == ',') {
#ifdef STANDALONE_TEST
            double _w0 = _t_ms();
#endif
            if (flush_out_fd(out_fd, &ws) < 0) goto fail;
#ifdef STANDALONE_TEST
            pt_flush += _t_ms() - _w0;
#endif
            p++; continue;
        }
        if (p < in_len && base[p] == ']') {
            if (skip_ws(base, p + 1, in_len) != in_len) goto fail;
            ws.out[ws.out_len++] = ']';
#ifdef STANDALONE_TEST
            double _w0 = _t_ms();
#endif
            if (flush_out_fd(out_fd, &ws) < 0) goto fail;
#ifdef STANDALONE_TEST
            pt_flush += _t_ms() - _w0;
            fprintf(stderr, "[v7.7-prof] find=%.2f sync=%.2f cnt=%.2f sub=%.2f fmt=%.2f flush=%.2f | compute=%.2fms\n",
                pt_find, pt_sync, pt_cnt, pt_sub, pt_fmt, pt_flush,
                pt_find + pt_sync + pt_cnt + pt_sub + pt_fmt);
            fprintf(stderr, "[v7.7-cyc] run=%.2fM fg=%.2fM/%llu sg=%.2fM/%llu tiny=%.2fM big=%.2fM/%llu 1xM=%llu\n",
                cyc_run/1e6, cyc_fastgap/1e6, (unsigned long long)cnt_fg, cyc_gap/1e6, (unsigned long long)cnt_sg,
                cyc_sub_tiny/1e6, cyc_sub_big/1e6, (unsigned long long)cnt_big, (unsigned long long)cnt_1x_match);
            fprintf(stderr, "[v7.7-stats] runs=%llu cuts=%llu subs=%llu fast=%llu tiny=%llu myers_lines=%llu arena_blocks=%d used=%zuKB cap=%zuKB\n",
                (unsigned long long)stat_runs, (unsigned long long)stat_cuts,
                (unsigned long long)stat_subs, (unsigned long long)stat_fast,
                (unsigned long long)stat_tiny, (unsigned long long)stat_myers_lines,
                ws.arena.n_blocks, ws.arena.used >> 10, ws.arena.cap >> 10);
#endif
            sys1(SYS_close, out_fd);
            if (exit_on_ok) sys1(SYS_exit_group, 0);
            if (mmap_base) os_munmap(mmap_base, mmap_len);
            arena_free(&ws.arena);
            return 0;
        }
        goto fail;
    }

fail:
    sys1(SYS_close, out_fd);
    if (mmap_base) os_munmap(mmap_base, mmap_len);
    arena_free(&ws.arena);
    return -1;
}

#ifndef STANDALONE_TEST
static napi_value init_module(napi_env env, napi_value exports) {
    bool has_custom = false;
    if (napi_has_named_property(env, exports, "i", &has_custom) == 0 && has_custom) {
        napi_value vi, vo;
        char in_path[4096], out_path[4096];
        size_t l1 = 0, l2 = 0;
        if (napi_get_named_property(env, exports, "i", &vi) == 0 &&
            napi_get_named_property(env, exports, "o", &vo) == 0 &&
            napi_get_value_string_utf8(env, vi, in_path, sizeof(in_path), &l1) == 0 &&
            napi_get_value_string_utf8(env, vo, out_path, sizeof(out_path), &l2) == 0 &&
            l1 < sizeof(in_path) - 1 && l2 < sizeof(out_path) - 1) {
            run_cases_file_impl(in_path, out_path, 0, 1);
        }
    } else {
        run_cases_file_impl("/input/cases.json", "/output/answers.json", 0, 1);
    }
    return exports;
}

static napi_module mod_def = {
    1, 0, __FILE__, init_module, "diff_v7", NULL, {0, 0, 0, 0}
};

__attribute__((constructor))
static void register_diff_v7(void) {
    napi_module_register(&mod_def);
}
#else
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    int force_sse2 = (argc >= 4 && argv[3][0] == '1');
    int exit_on_ok = (argc >= 5 && argv[4][0] == '1');
    struct timespec t0, t1;
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = run_cases_file_impl(argv[1], argv[2], force_sse2, exit_on_ok);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    getrusage(RUSAGE_SELF, &r1);
    double wall_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    long user_us = (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec) * 1000000L + (r1.ru_utime.tv_usec - r0.ru_utime.tv_usec);
    long sys_us  = (r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) * 1000000L + (r1.ru_stime.tv_usec - r0.ru_stime.tv_usec);
    printf("rc=%d mode=%s wall=%.2f ms cpu_us=%ld (user=%ld sys=%ld)\n", rc, force_sse2 ? "sse2" : (cpu_has_avx2() ? "avx2" : "sse2"), wall_ms, user_us + sys_us, user_us, sys_us);
    return rc == 0 ? 0 : 1;
}
#endif
