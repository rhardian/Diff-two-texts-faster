// Zero-libc Linux x86_64 N-API addon for exact makefun-diff 1.0.0 (v5).
// Key architectural features:
// - Zero libc dependencies (direct Linux x86_64 syscalls, runtime CPUID+XGETBV AVX2/SSE2 dispatch)
// - Zero-copy MAP_SHARED|MAP_POPULATE file mapping + compact ~1 MB single-mmap L2-packed arena
// - Scratch-aliased output buffer (reuses dead ht+ent_meta+ids+last slab as 650+ KB output buffer)
// - L1-tiled 64B/iter AVX2 JSON line scanner + zero-loop sampled hash + 8-ahead ht prefetch
// - 4-byte ent_meta + 16-bit tag ht + L1 short_val[256] duplicate bypass + empty-line fast-path
// - uint16_t line IDs (16-wide AVX2 _mm256_cmpeq_epi16 Myers snake) + smart exact dominance pruning
// - Branchless 32B/64B AVX2 hunk formatter + fast single-write per case + clean _exit(0)
//
// Automatically returns -1 on non-canonical JSON or >=65533 lines so main.ts falls back to engine.cjs.
//
// Build (Linux x86_64):
//   gcc -O3 -shared -fPIC -nostdlib -fno-builtin -fno-stack-protector
//       -fno-asynchronous-unwind-tables -fvisibility=hidden
//       -Wl,--gc-sections,-s,-z,max-page-size=4096,--allow-shlib-undefined -o diff.node diff_napi.c

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#ifdef STANDALONE_TEST
#include <stdio.h>
#endif

// ---------------------------------------------------------------------------
// Minimal N-API ABI definitions (stable since Node.js 8.0, resolved from host `node`)
// ---------------------------------------------------------------------------

typedef struct napi_env__* napi_env;
typedef struct napi_value__* napi_value;
typedef struct napi_callback_info__* napi_callback_info;
typedef int napi_status;
typedef napi_value (*napi_callback)(napi_env env, napi_callback_info info);

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
extern napi_status napi_create_function(napi_env env, const char* utf8name, size_t length, napi_callback cb, void* data, napi_value* result);
extern napi_status napi_set_named_property(napi_env env, napi_value object, const char* utf8name, napi_value value);
extern napi_status napi_get_cb_info(napi_env env, napi_callback_info cbinfo, size_t* argc, napi_value* argv, napi_value* this_arg, void** data);
extern napi_status napi_get_value_string_utf8(napi_env env, napi_value value, char* buf, size_t bufsize, size_t* result);
extern napi_status napi_create_int32(napi_env env, int32_t value, napi_value* result);
#endif

// ---------------------------------------------------------------------------
// Direct Linux x86_64 syscalls (zero libc dependency!)
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
    long p = sys6(SYS_mmap, 0, (long)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_POPULATE, -1, 0);
    if (p < 0 && p > -4096) {
        p = sys6(SYS_mmap, 0, (long)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p < 0 && p > -4096) return NULL;
    }
    return (void*)p;
}

static inline void os_munmap(void* p, size_t len) {
    if (p) sys2(SYS_munmap, (long)p, (long)len);
}

// ---------------------------------------------------------------------------
// Runtime CPUID check for AVX2 + BMI1
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
// Low-level memory ops (zero libc!)
// ---------------------------------------------------------------------------

static inline uint64_t read_u64(const uint8_t* p) { uint64_t v; __builtin_memcpy(&v, p, 8); return v; }
static inline uint32_t read_u32(const uint8_t* p) { uint32_t v; __builtin_memcpy(&v, p, 4); return v; }
static inline uint16_t read_u16(const uint8_t* p) { uint16_t v; __builtin_memcpy(&v, p, 2); return v; }

static inline void mem_zero_bytes(void* dst, size_t n) {
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

__attribute__((target("avx2"), always_inline))
static inline void mem_copy_bytes_avx2(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n >= 128) {
        _mm256_storeu_si256((__m256i*)(d),      _mm256_loadu_si256((const __m256i*)(s)));
        _mm256_storeu_si256((__m256i*)(d + 32), _mm256_loadu_si256((const __m256i*)(s + 32)));
        _mm256_storeu_si256((__m256i*)(d + 64), _mm256_loadu_si256((const __m256i*)(s + 64)));
        _mm256_storeu_si256((__m256i*)(d + 96), _mm256_loadu_si256((const __m256i*)(s + 96)));
        d += 128; s += 128; n -= 128;
    }
    while (n >= 32) { _mm256_storeu_si256((__m256i*)d, _mm256_loadu_si256((const __m256i*)s)); d += 32; s += 32; n -= 32; }
    while (n >= 8) { uint64_t v = read_u64(s); __builtin_memcpy(d, &v, 8); d += 8; s += 8; n -= 8; }
    while (n--) *d++ = *s++;
}

static inline void mem_copy_bytes(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n >= 64) {
        _mm_storeu_si128((__m128i*)(d),      _mm_loadu_si128((const __m128i*)(s)));
        _mm_storeu_si128((__m128i*)(d + 16), _mm_loadu_si128((const __m128i*)(s + 16)));
        _mm_storeu_si128((__m128i*)(d + 32), _mm_loadu_si128((const __m128i*)(s + 32)));
        _mm_storeu_si128((__m128i*)(d + 48), _mm_loadu_si128((const __m128i*)(s + 48)));
        d += 64; s += 64; n -= 64;
    }
    while (n >= 16) { _mm_storeu_si128((__m128i*)d, _mm_loadu_si128((const __m128i*)s)); d += 16; s += 16; n -= 16; }
    while (n--) *d++ = *s++;
}

static inline uint8_t* copy_short_sse2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    if (len >= 16) {
        while (len >= 16) {
            _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
            dst += 16; src += 16; len -= 16;
        }
        if (len > 0) _mm_storeu_si128((__m128i*)(end - 16), _mm_loadu_si128((const __m128i*)(src + len - 16)));
        return end;
    }
    if (len >= 8) {
        uint64_t a = read_u64(src), b = read_u64(src + len - 8);
        __builtin_memcpy(dst, &a, 8); __builtin_memcpy(end - 8, &b, 8);
        return end;
    }
    if (len >= 4) {
        uint32_t a = read_u32(src), b = read_u32(src + len - 4);
        __builtin_memcpy(dst, &a, 4); __builtin_memcpy(end - 4, &b, 4);
        return end;
    }
    for (size_t i = 0; i < len; i++) dst[i] = src[i];
    return end;
}

__attribute__((target("avx2"), always_inline))
static inline uint8_t* copy_short_avx2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    if (len >= 32) {
        while (len >= 32) {
            _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
            dst += 32; src += 32; len -= 32;
        }
        if (len > 0) _mm256_storeu_si256((__m256i*)(end - 32), _mm256_loadu_si256((const __m256i*)(src + len - 32)));
        return end;
    }
    return copy_short_sse2(dst, src, len);
}

// Branchless over-copy for lines safely away from file end (dst has >=64B slack, src has >=64B slack)
__attribute__((target("avx2"), always_inline))
static inline uint8_t* copy_line_fast_avx2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
    if (__builtin_expect(len <= 32, 1)) return end;
    _mm256_storeu_si256((__m256i*)(dst + 32), _mm256_loadu_si256((const __m256i*)(src + 32)));
    if (__builtin_expect(len <= 64, 1)) return end;
    dst += 64; src += 64; len -= 64;
    while (len >= 32) {
        _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
        dst += 32; src += 32; len -= 32;
    }
    if (len > 0) _mm256_storeu_si256((__m256i*)(end - 32), _mm256_loadu_si256((const __m256i*)(src + len - 32)));
    return end;
}

static inline uint8_t* copy_line_fast_sse2(uint8_t* dst, const uint8_t* src, size_t len) {
    uint8_t* end = dst + len;
    _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
    if (len <= 16) return end;
    _mm_storeu_si128((__m128i*)(dst + 16), _mm_loadu_si128((const __m128i*)(src + 16)));
    if (len <= 32) return end;
    _mm_storeu_si128((__m128i*)(dst + 32), _mm_loadu_si128((const __m128i*)(src + 32)));
    _mm_storeu_si128((__m128i*)(dst + 48), _mm_loadu_si128((const __m128i*)(src + 48)));
    if (len <= 64) return end;
    dst += 64; src += 64; len -= 64;
    while (len >= 16) {
        _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
        dst += 16; src += 16; len -= 16;
    }
    if (len > 0) _mm_storeu_si128((__m128i*)(end - 16), _mm_loadu_si128((const __m128i*)(src + len - 16)));
    return end;
}

static inline int eq_bytes_sse2(const uint8_t* a, const uint8_t* b, size_t len) {
    if (len <= 16) {
        if (len >= 8) return ((read_u64(a) ^ read_u64(b)) | (read_u64(a + len - 8) ^ read_u64(b + len - 8))) == 0;
        if (len >= 4) return ((read_u32(a) ^ read_u32(b)) | (read_u32(a + len - 4) ^ read_u32(b + len - 4))) == 0;
        for (size_t i = 0; i < len; i++) if (a[i] != b[i]) return 0;
        return 1;
    }
    while (len >= 16) {
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b))) != 0xFFFF) return 0;
        a += 16; b += 16; len -= 16;
    }
    if (len > 0) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_loadu_si128((const __m128i*)(a + len - 16)),
            _mm_loadu_si128((const __m128i*)(b + len - 16)))) == 0xFFFF;
    }
    return 1;
}

__attribute__((target("avx2"), noinline))
static int eq_bytes_avx2_long(const uint8_t* a, const uint8_t* b, size_t len) {
    while (len >= 32) {
        if ((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_loadu_si256((const __m256i*)a), _mm256_loadu_si256((const __m256i*)b))) != 0xFFFFFFFFu) return 0;
        a += 32; b += 32; len -= 32;
    }
    if (len > 0) {
        return (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_loadu_si256((const __m256i*)(a + len - 32)),
            _mm256_loadu_si256((const __m256i*)(b + len - 32)))) == 0xFFFFFFFFu;
    }
    return 1;
}

__attribute__((target("avx2"), always_inline))
static inline int eq_bytes_avx2(const uint8_t* a, const uint8_t* b, size_t len) {
    if (__builtin_expect(len <= 32, 1)) {
        if (len >= 16) {
            __m128i e0 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b));
            __m128i e1 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)(a + len - 16)), _mm_loadu_si128((const __m128i*)(b + len - 16)));
            return _mm_movemask_epi8(_mm_and_si128(e0, e1)) == 0xFFFF;
        }
        if (len >= 8) return ((read_u64(a) ^ read_u64(b)) | (read_u64(a + len - 8) ^ read_u64(b + len - 8))) == 0;
        if (len >= 4) return ((read_u32(a) ^ read_u32(b)) | (read_u32(a + len - 4) ^ read_u32(b + len - 4))) == 0;
        for (size_t i = 0; i < len; i++) if (a[i] != b[i]) return 0;
        return 1;
    }
    return eq_bytes_avx2_long(a, b, len);
}

// ---------------------------------------------------------------------------
// Fast full-coverage 64-bit hash (zero loop for <=48B lines, linear for >48B -> zero DoS/O(N^2))
// ---------------------------------------------------------------------------

#define K1 0x9E3779B97F4A7C15ULL
#define K2 0xBF58476D1CE4E5B9ULL

static inline uint64_t hash_sample(const uint8_t* p, size_t len) {
    if (__builtin_expect(len >= 16, 1)) {
        uint64_t a = read_u64(p);
        uint64_t b = read_u64(p + (len >> 2));
        uint64_t c = read_u64(p + (len >> 1) - 4);
        uint64_t d = read_u64(p + len - 8);
        uint64_t h = ((a ^ (len * K2)) * K1 ^ b) * K2 ^ (c * K1) ^ d;
        if (__builtin_expect(len > 128, 0)) {
            const uint8_t* q = p + 16;
            const uint8_t* e = p + len - 16;
            while (q + 16 <= e) {
                h = (h ^ read_u64(q)) * K1 ^ read_u64(q + 8);
                q += 16;
            }
        }
        return h ^ (h >> 32);
    }
    if (len >= 4) {
        uint64_t a = read_u32(p);
        uint64_t b = read_u32(p + len - 4);
        uint64_t c = (len >= 8) ? read_u32(p + 4) : 0;
        uint64_t h = ((a | (b << 32)) ^ (c * K2) ^ len) * K1;
        return h ^ (h >> 32);
    }
    uint64_t v = len ? ((uint64_t)p[0] | ((uint64_t)p[len - 1] << 8)) : 0;
    uint64_t h = (v ^ (len << 16)) * K1;
    return h ^ (h >> 32);
}

// Pack len (1..8) and exact bytes into a unique 64-bit key for O(1) L1 duplicate check!
// Format: for len <= 7, top byte = len (1..7), low 7 bytes = raw bytes (zero padded);
// for len == 8, only used when we check len <= 7! (Wait: len <= 7 covers "", "}", "  }", "    }", "};", "end", "---", "#", "  ]" WITH \n!
// Actually "    }\n" is 7 bytes, "}\r\n" is 5 bytes, "    }\r\n" is 9 bytes. Better: store len in separate byte or use pack_le7!)
static inline uint64_t pack_le7(const uint8_t* p, size_t len) {
    uint64_t v = (uint64_t)len << 56;
    if (len >= 4) {
        v |= (uint64_t)read_u32(p);
        v |= ((uint64_t)read_u32(p + len - 4) << ((len - 4) * 8));
    } else if (len >= 2) {
        v |= (uint64_t)read_u16(p);
        if (len == 3) v |= ((uint64_t)p[2] << 16);
    } else if (len == 1) {
        v |= p[0];
    }
    return v;
}

// ---------------------------------------------------------------------------
// Arena allocator (single compact mmap chunk, grows only on extreme inputs)
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t* base;
    size_t used;
    size_t cap;
    void* blocks[16];
    size_t block_lens[16];
    int n_blocks;
} Arena;

static int arena_init(Arena* a, size_t cap) {
    cap = (cap + 4095) & ~(size_t)4095;
    uint8_t* p = (uint8_t*)os_mmap(cap);
    if (!p) return -1;
    a->base = p;
    a->used = 0;
    a->cap = cap;
    a->blocks[0] = p;
    a->block_lens[0] = cap;
    a->n_blocks = 1;
    return 0;
}

static void arena_free(Arena* a) {
    for (int i = 0; i < a->n_blocks; i++) os_munmap(a->blocks[i], a->block_lens[i]);
    a->n_blocks = 0;
}

static void* arena_alloc(Arena* a, size_t sz, size_t align) {
    size_t p = (a->used + align - 1) & ~(align - 1);
    if (p + sz <= a->cap) {
        a->used = p + sz;
        return a->base + p;
    }
    size_t ncap = a->cap * 2;
    if (ncap < sz + 65536) ncap = sz + 65536;
    ncap = (ncap + 4095) & ~(size_t)4095;
    if (a->n_blocks >= 16) return NULL;
    uint8_t* nb = (uint8_t*)os_mmap(ncap);
    if (!nb) return NULL;
    a->blocks[a->n_blocks] = nb;
    a->block_lens[a->n_blocks] = ncap;
    a->n_blocks++;
    a->base = nb;
    a->cap = ncap;
    a->used = sz;
    return nb;
}

// ---------------------------------------------------------------------------
// Workspace structs (L2-packed + scratch-aliased output buffer!)
// ---------------------------------------------------------------------------

// 4-byte line metadata:
//   bit 31 (NEW_ONLY_BIT): 0 = first seen in old (index into old_off), 1 = first seen in new (index into new_off)
//   bit 30 (MULTI_BIT):    for old lines, 0 = unique in old, 1 = appeared >=2 times in old
//   bits 0..29:            first line index
#define NEW_ONLY_BIT 0x80000000u
#define MULTI_BIT    0x40000000u
#define IDX_MASK     0x3FFFFFFFu
#define MAX_NATIVE_LINES 65533u
#define DEAD (-2)

typedef struct {
    uint32_t o, p, dels, ins;
} Chg;

typedef struct {
    Arena arena;

    // Permanent arrays (survive across formatting!)
    uint32_t* old_off; size_t old_off_cap;
    uint32_t* new_off; size_t new_off_cap;
    Chg* chg;          size_t chg_len; size_t chg_cap;
    int32_t* v;        size_t v_half;

    // Scratch slab (dead during format_hunks_json -> aliased as ws.out!)
    uint8_t* scratch_base;
    size_t scratch_cap;

    uint32_t* ht;      size_t ht_mask; size_t ht_cap;
    uint32_t* ent_meta; size_t entries_cap;
    size_t next_id;

    uint16_t* a_ids;   size_t a_ids_cap; size_t a_len;
    uint16_t* b_ids;   size_t b_ids_cap; size_t b_len;

    uint16_t* last_a16; size_t last16_cap;
    uint16_t* last_b16;

    int32_t* tr_x;     size_t tr_cap;
    uint8_t* tr_c;
    uint32_t* row_base; size_t row_cap;
    int32_t* row_lo;
    uint32_t* seg_ops; size_t seg_cap;

    // Output buffer (defaults to scratch_base/scratch_cap; grows cleanly on huge diff outputs)
    uint8_t* out;      size_t out_len; size_t out_cap;
    uint8_t* out_overflow; size_t out_overflow_cap;

    // Tiny 2 KB L1 short-line cache for first 256 unique lines with len <= 7 (0 = unused / len > 7)
    uint64_t short_key[256];
} WS;

__attribute__((noinline, cold))
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

__attribute__((noinline, cold))
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

__attribute__((noinline, cold))
static int ensure_entries(WS* ws, size_t need) {
    if (need <= ws->entries_cap) return 0;
    size_t nc = ws->entries_cap ? ws->entries_cap * 2 : ((need + 63) & ~(size_t)63);
    while (nc < need) nc *= 2;
    uint32_t* np = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
    if (!np) return -1;
    if (ws->next_id > 0) mem_copy_bytes(np, ws->ent_meta, ws->next_id * 4);
    ws->ent_meta = np; ws->entries_cap = nc;
    return 0;
}

__attribute__((noinline, cold))
static int ensure_out_grow(WS* ws, size_t need) {
    if (ws->out_overflow && need <= ws->out_overflow_cap) {
        if (ws->out != ws->out_overflow) {
            if (ws->out_len > 0) mem_copy_bytes(ws->out_overflow, ws->out, ws->out_len);
            ws->out = ws->out_overflow;
            ws->out_cap = ws->out_overflow_cap;
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

static int ensure_trace(WS* ws, size_t tr_used, size_t tr_need, size_t row_used, size_t row_need) {
    if (tr_need > ws->tr_cap) {
        size_t nc = ws->tr_cap ? ws->tr_cap * 2 : 4096;
        while (nc < tr_need) nc *= 2;
        int32_t* nx = (int32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        uint8_t* nch = (uint8_t*)arena_alloc(&ws->arena, nc, 32);
        if (!nx || !nch) return -1;
        if (tr_used > 0) {
            mem_copy_bytes(nx, ws->tr_x, tr_used * 4);
            mem_copy_bytes(nch, ws->tr_c, tr_used);
        }
        ws->tr_x = nx; ws->tr_c = nch; ws->tr_cap = nc;
    }
    if (row_need > ws->row_cap) {
        size_t nc = ws->row_cap ? ws->row_cap * 2 : 1024;
        while (nc < row_need) nc *= 2;
        uint32_t* nb = (uint32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        int32_t* nl = (int32_t*)arena_alloc(&ws->arena, nc * 4, 32);
        if (!nb || !nl) return -1;
        if (row_used > 0) {
            mem_copy_bytes(nb, ws->row_base, row_used * 4);
            mem_copy_bytes(nl, ws->row_lo, row_used * 4);
        }
        ws->row_base = nb; ws->row_lo = nl; ws->row_cap = nc;
    }
    return 0;
}

__attribute__((noinline, cold))
static int grow_ht(WS* ws, const uint8_t* base) {
    size_t ncap = (ws->ht_mask + 1) << 1;
    if (ncap > ws->ht_cap) {
        uint32_t* nht = (uint32_t*)arena_alloc(&ws->arena, ncap * 4, 32);
        if (!nht) return -1;
        ws->ht = nht; ws->ht_cap = ncap;
    }
    mem_zero_bytes(ws->ht, ncap * 4);
    size_t mask = ncap - 1;
    ws->ht_mask = mask;
    uint32_t* ht = ws->ht;
    const uint32_t* em = ws->ent_meta;
    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    for (size_t id = 0; id < ws->next_id; id++) {
        uint32_t m = em[id];
        uint32_t idx = m & IDX_MASK;
        uint32_t off = (m & NEW_ONLY_BIT) ? noff[idx] : ooff[idx];
        uint32_t end = (m & NEW_ONLY_BIT) ? noff[idx + 1] : ooff[idx + 1];
        uint64_t h = hash_sample(base + off, end - off);
        uint32_t tag16 = ((uint32_t)(h >> 32)) & 0xFFFF0000u;
        size_t slot = (size_t)h & mask;
        while (ht[slot] != 0) slot = (slot + 1) & mask;
        ht[slot] = tag16 | (uint32_t)(id + 1);
    }
    return 0;
}

static int intern_reset(WS* ws, size_t est_lines, int use_avx2, int first_case) {
    size_t cap = 1024;
    size_t target = est_lines * 2;
    while (cap < target) cap <<= 1;
    if (cap > 65536 && est_lines <= 48000) cap = 65536;
    if (cap > ws->ht_cap) {
        uint32_t* nht = (uint32_t*)arena_alloc(&ws->arena, cap * 4, 32);
        if (!nht) return -1;
        ws->ht = nht; ws->ht_cap = cap;
        first_case = 1; // freshly zeroed mmap page
    }
    if (!first_case) {
        if (use_avx2) mem_zero_avx2(ws->ht, cap * 4);
        else mem_zero_bytes(ws->ht, cap * 4);
    }
    ws->ht_mask = cap - 1;
    if (ensure_entries(ws, est_lines) < 0) return -1;
    ws->next_id = 0;
    return 0;
}

// ---------------------------------------------------------------------------
// Fast 16-bit-tag + 4-byte ent_meta + L1 short_key[256] interning
// ---------------------------------------------------------------------------

__attribute__((target("avx2"), always_inline))
static inline uint32_t intern_old_avx2(WS* __restrict ws, const uint8_t* __restrict base, uint32_t off, uint32_t len, uint64_t h, uint32_t line_idx) {
    if (__builtin_expect((ws->next_id * 4) >= (ws->ht_mask * 3), 0)) {
        if (grow_ht(ws, base) < 0) return 0xFFFFFFFFu;
    }
    size_t mask = ws->ht_mask;
    uint32_t tag16 = ((uint32_t)(h >> 32)) & 0xFFFF0000u;
    size_t slot = (size_t)h & mask;
    uint32_t* __restrict ht = ws->ht;
    uint32_t* __restrict em = ws->ent_meta;
    const uint32_t* __restrict ooff = ws->old_off;
    for (;;) {
        uint32_t ent = ht[slot];
        if (ent == 0) {
            size_t id = ws->next_id;
            if (__builtin_expect(id >= MAX_NATIVE_LINES, 0)) return 0xFFFFFFFFu;
            if (__builtin_expect(id >= ws->entries_cap, 0)) {
                if (ensure_entries(ws, id + 1) < 0) return 0xFFFFFFFFu;
                em = ws->ent_meta;
            }
            ws->next_id = id + 1;
            ht[slot] = tag16 | (uint32_t)(id + 1);
            em[id] = line_idx;
            if (id < 256) ws->short_key[id] = (len <= 7) ? pack_le7(base + off, len) : 0;
            return (uint32_t)id;
        }
        if ((ent & 0xFFFF0000u) == tag16) {
            size_t id = (ent & 0xFFFFu) - 1;
            if (id < 256 && len <= 7) {
                if (ws->short_key[id] == pack_le7(base + off, len)) {
                    uint32_t m = em[id];
                    if (!(m & MULTI_BIT)) em[id] = m | MULTI_BIT;
                    return (uint32_t)id;
                }
            } else {
                uint32_t m = em[id];
                uint32_t idx = m & IDX_MASK;
                uint32_t eoff = ooff[idx], elen = ooff[idx + 1] - eoff;
                if (elen == len && eq_bytes_avx2(base + eoff, base + off, len)) {
                    if (!(m & MULTI_BIT)) em[id] = m | MULTI_BIT;
                    return (uint32_t)id;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}

__attribute__((target("avx2"), always_inline))
static inline uint32_t intern_new_avx2(WS* __restrict ws, const uint8_t* __restrict base, uint32_t off, uint32_t len, uint64_t h, uint32_t line_idx) {
    if (__builtin_expect((ws->next_id * 4) >= (ws->ht_mask * 3), 0)) {
        if (grow_ht(ws, base) < 0) return 0xFFFFFFFFu;
    }
    size_t mask = ws->ht_mask;
    uint32_t tag16 = ((uint32_t)(h >> 32)) & 0xFFFF0000u;
    size_t slot = (size_t)h & mask;
    uint32_t* __restrict ht = ws->ht;
    uint32_t* __restrict em = ws->ent_meta;
    const uint32_t* __restrict ooff = ws->old_off;
    const uint32_t* __restrict noff = ws->new_off;
    for (;;) {
        uint32_t ent = ht[slot];
        if (ent == 0) {
            size_t id = ws->next_id;
            if (__builtin_expect(id >= MAX_NATIVE_LINES, 0)) return 0xFFFFFFFFu;
            if (__builtin_expect(id >= ws->entries_cap, 0)) {
                if (ensure_entries(ws, id + 1) < 0) return 0xFFFFFFFFu;
                em = ws->ent_meta;
            }
            ws->next_id = id + 1;
            ht[slot] = tag16 | (uint32_t)(id + 1);
            em[id] = NEW_ONLY_BIT | line_idx;
            if (id < 256) ws->short_key[id] = (len <= 7) ? pack_le7(base + off, len) : 0;
            return (uint32_t)id;
        }
        if ((ent & 0xFFFF0000u) == tag16) {
            size_t id = (ent & 0xFFFFu) - 1;
            if (id < 256 && len <= 7) {
                if (ws->short_key[id] == pack_le7(base + off, len)) return (uint32_t)id;
            } else {
                uint32_t m = em[id];
                uint32_t idx = m & IDX_MASK;
                uint32_t eoff = (m & NEW_ONLY_BIT) ? noff[idx] : ooff[idx];
                uint32_t eend = (m & NEW_ONLY_BIT) ? noff[idx + 1] : ooff[idx + 1];
                if ((eend - eoff) == len && eq_bytes_avx2(base + eoff, base + off, len)) {
                    return (uint32_t)id;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}

static inline uint32_t intern_old_sse2(WS* __restrict ws, const uint8_t* __restrict base, uint32_t off, uint32_t len, uint64_t h, uint32_t line_idx) {
    if (__builtin_expect((ws->next_id * 4) >= (ws->ht_mask * 3), 0)) {
        if (grow_ht(ws, base) < 0) return 0xFFFFFFFFu;
    }
    size_t mask = ws->ht_mask;
    uint32_t tag16 = ((uint32_t)(h >> 32)) & 0xFFFF0000u;
    size_t slot = (size_t)h & mask;
    uint32_t* __restrict ht = ws->ht;
    uint32_t* __restrict em = ws->ent_meta;
    const uint32_t* __restrict ooff = ws->old_off;
    for (;;) {
        uint32_t ent = ht[slot];
        if (ent == 0) {
            size_t id = ws->next_id;
            if (__builtin_expect(id >= MAX_NATIVE_LINES, 0)) return 0xFFFFFFFFu;
            if (__builtin_expect(id >= ws->entries_cap, 0)) {
                if (ensure_entries(ws, id + 1) < 0) return 0xFFFFFFFFu;
                em = ws->ent_meta;
            }
            ws->next_id = id + 1;
            ht[slot] = tag16 | (uint32_t)(id + 1);
            em[id] = line_idx;
            if (id < 256) ws->short_key[id] = (len <= 7) ? pack_le7(base + off, len) : 0;
            return (uint32_t)id;
        }
        if ((ent & 0xFFFF0000u) == tag16) {
            size_t id = (ent & 0xFFFFu) - 1;
            if (id < 256 && len <= 7) {
                if (ws->short_key[id] == pack_le7(base + off, len)) {
                    uint32_t m = em[id];
                    if (!(m & MULTI_BIT)) em[id] = m | MULTI_BIT;
                    return (uint32_t)id;
                }
            } else {
                uint32_t m = em[id];
                uint32_t idx = m & IDX_MASK;
                uint32_t eoff = ooff[idx], elen = ooff[idx + 1] - eoff;
                if (elen == len && eq_bytes_sse2(base + eoff, base + off, len)) {
                    if (!(m & MULTI_BIT)) em[id] = m | MULTI_BIT;
                    return (uint32_t)id;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}

static inline uint32_t intern_new_sse2(WS* __restrict ws, const uint8_t* __restrict base, uint32_t off, uint32_t len, uint64_t h, uint32_t line_idx) {
    if (__builtin_expect((ws->next_id * 4) >= (ws->ht_mask * 3), 0)) {
        if (grow_ht(ws, base) < 0) return 0xFFFFFFFFu;
    }
    size_t mask = ws->ht_mask;
    uint32_t tag16 = ((uint32_t)(h >> 32)) & 0xFFFF0000u;
    size_t slot = (size_t)h & mask;
    uint32_t* __restrict ht = ws->ht;
    uint32_t* __restrict em = ws->ent_meta;
    const uint32_t* __restrict ooff = ws->old_off;
    const uint32_t* __restrict noff = ws->new_off;
    for (;;) {
        uint32_t ent = ht[slot];
        if (ent == 0) {
            size_t id = ws->next_id;
            if (__builtin_expect(id >= MAX_NATIVE_LINES, 0)) return 0xFFFFFFFFu;
            if (__builtin_expect(id >= ws->entries_cap, 0)) {
                if (ensure_entries(ws, id + 1) < 0) return 0xFFFFFFFFu;
                em = ws->ent_meta;
            }
            ws->next_id = id + 1;
            ht[slot] = tag16 | (uint32_t)(id + 1);
            em[id] = NEW_ONLY_BIT | line_idx;
            if (id < 256) ws->short_key[id] = (len <= 7) ? pack_le7(base + off, len) : 0;
            return (uint32_t)id;
        }
        if ((ent & 0xFFFF0000u) == tag16) {
            size_t id = (ent & 0xFFFFu) - 1;
            if (id < 256 && len <= 7) {
                if (ws->short_key[id] == pack_le7(base + off, len)) return (uint32_t)id;
            } else {
                uint32_t m = em[id];
                uint32_t idx = m & IDX_MASK;
                uint32_t eoff = (m & NEW_ONLY_BIT) ? noff[idx] : ooff[idx];
                uint32_t eend = (m & NEW_ONLY_BIT) ? noff[idx + 1] : ooff[idx + 1];
                if ((eend - eoff) == len && eq_bytes_sse2(base + eoff, base + off, len)) {
                    return (uint32_t)id;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}

// ---------------------------------------------------------------------------
// L1-tiled 64B/iter AVX2 & 32B/iter SSE2 canonical JSON `old` scanners
// ---------------------------------------------------------------------------

__attribute__((target("avx2,bmi")))
static size_t scan_json_old_avx2(WS* ws, const uint8_t* base, size_t start, size_t file_len, int* bare_out) {
    ws->a_len = 0;
    *bare_out = 0;
    if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, 0, 1024) < 0) return (size_t)-1;
    if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, 0, 1024) < 0) return (size_t)-1;
    uint16_t* a_ids = ws->a_ids;
    uint32_t* old_off = ws->old_off;
    size_t a_cap = ws->a_ids_cap;
    size_t o_cap = ws->old_off_cap;
    old_off[0] = (uint32_t)start;

    __m256i v_bs = _mm256_set1_epi8('\\');
    __m256i v_qt = _mm256_set1_epi8('"');

    size_t p = start, n_lines = 0, end_pos = 0;
    int eof = 0;
    uint32_t empty_lf_id = 0xFFFFFFFFu;
    uint32_t empty_crlf_id = 0xFFFFFFFFu;
    uint64_t th[176]; // holds up to 128 + 32 lines per L1 tile

    while (!eof) {
        size_t b0 = n_lines;
        size_t blim = b0 + 128;
        while (n_lines < blim) {
            if (__builtin_expect(n_lines + 40 > a_cap || n_lines + 41 > o_cap, 0)) {
                if (n_lines >= MAX_NATIVE_LINES) return (size_t)-1;
                if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, n_lines, n_lines + 256) < 0) return (size_t)-1;
                if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, n_lines + 1, n_lines + 257) < 0) return (size_t)-1;
                a_ids = ws->a_ids; a_cap = ws->a_ids_cap;
                old_off = ws->old_off; o_cap = ws->old_off_cap;
            }
            if (__builtin_expect(p + 64 <= file_len, 1)) {
                __m256i v0 = _mm256_loadu_si256((const __m256i*)(base + p));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(base + p + 32));
                uint32_t m0 = (uint32_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v0, v_bs), _mm256_cmpeq_epi8(v0, v_qt)));
                uint32_t m1 = (uint32_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v1, v_bs), _mm256_cmpeq_epi8(v1, v_qt)));
                uint64_t m = (uint64_t)m0 | ((uint64_t)m1 << 32);
                while (m) {
                    uint32_t tz = (uint32_t)__builtin_ctzll(m);
                    size_t hit = p + tz;
                    if (__builtin_expect(base[hit] == '\\', 1)) {
                        if (__builtin_expect(hit + 1 >= file_len, 0)) return (size_t)-1;
                        uint8_t c = base[hit + 1];
                        if (__builtin_expect(c == 'n', 1)) {
                            old_off[++n_lines] = (uint32_t)(hit + 2);
                            m &= m - 1;
                        } else if (c == 'r') {
                            if (hit + 3 < file_len && base[hit + 2] == '\\' && base[hit + 3] == 'n') {
                                old_off[++n_lines] = (uint32_t)(hit + 4);
                                m &= m - 1;
                                if (tz + 2 < 64) { m &= ~(1ULL << (tz + 2)); }
                                else { p = hit + 4; goto next_vec; }
                            } else {
                                m &= m - 1;
                            }
                        } else if (c == '\\' || c == '"') {
                            m &= m - 1;
                            if (tz + 1 < 64) { m &= ~(1ULL << (tz + 1)); }
                            else { p = hit + 2; goto next_vec; }
                        } else if (c == 't' || c == 'b' || c == 'f') {
                            m &= m - 1;
                        } else {
                            return (size_t)-1;
                        }
                    } else {
                        if (hit > (size_t)old_off[n_lines]) {
                            old_off[++n_lines] = (uint32_t)hit;
                            *bare_out = 1;
                        }
                        end_pos = hit + 1;
                        eof = 1;
                        goto flush_tile;
                    }
                }
                p += 64;
            next_vec:;
            } else {
                // Safe scalar tail
                while (p < file_len && n_lines < blim) {
                    uint8_t b = base[p];
                    if (b == '"') {
                        if (p > (size_t)old_off[n_lines]) {
                            old_off[++n_lines] = (uint32_t)p;
                            *bare_out = 1;
                        }
                        end_pos = p + 1;
                        eof = 1;
                        goto flush_tile;
                    }
                    if (b == '\\') {
                        if (p + 1 >= file_len) return (size_t)-1;
                        uint8_t c = base[p + 1];
                        if (c == 'n') { old_off[++n_lines] = (uint32_t)(p + 2); p += 2; }
                        else if (c == 'r') {
                            if (p + 3 < file_len && base[p + 2] == '\\' && base[p + 3] == 'n') {
                                old_off[++n_lines] = (uint32_t)(p + 4); p += 4;
                            } else p += 2;
                        } else if (c == '\\' || c == '"' || c == 't' || c == 'b' || c == 'f') {
                            p += 2;
                        } else return (size_t)-1;
                    } else p++;
                }
                if (!eof && p >= file_len) return (size_t)-1;
            }
        }
    flush_tile:;
        if (n_lines >= MAX_NATIVE_LINES) return (size_t)-1;
        size_t cnt = n_lines - b0;
        if (cnt == 0) break;

        // Pass A (L1 hot): hash all lines in tile
        for (size_t k = 0; k < cnt; k++) {
            uint32_t s = old_off[b0 + k], l = old_off[b0 + k + 1] - s;
            th[k] = hash_sample(base + s, l);
        }
        // Pass B (8-ahead prefetch): intern all lines in tile
        size_t mask = ws->ht_mask;
        uint32_t* ht = ws->ht;
        size_t non_bare_cnt = (eof && *bare_out) ? (cnt - 1) : cnt;
        for (size_t k = 0; k < cnt; k++) {
            if (k + 8 < cnt) __builtin_prefetch(&ht[th[k + 8] & mask], 1, 1);
            uint32_t li = (uint32_t)(b0 + k);
            uint32_t s = old_off[li], l = old_off[li + 1] - s;
            // Fast empty-line bypass (only for non-bare lines which are guaranteed to end in \n!)
            if (k < non_bare_cnt) {
                if (l == 2 && empty_lf_id != 0xFFFFFFFFu) {
                    if (!(ws->ent_meta[empty_lf_id] & MULTI_BIT)) ws->ent_meta[empty_lf_id] |= MULTI_BIT;
                    a_ids[li] = (uint16_t)empty_lf_id;
                    continue;
                }
                if (l == 4 && empty_crlf_id != 0xFFFFFFFFu && read_u32(base + s) == 0x6E5C725Cu) {
                    if (!(ws->ent_meta[empty_crlf_id] & MULTI_BIT)) ws->ent_meta[empty_crlf_id] |= MULTI_BIT;
                    a_ids[li] = (uint16_t)empty_crlf_id;
                    continue;
                }
            }
            uint32_t id = intern_old_avx2(ws, base, s, l, th[k], li);
            if (__builtin_expect(id == 0xFFFFFFFFu, 0)) return (size_t)-1;
            if (k < non_bare_cnt) {
                if (l == 2 && empty_lf_id == 0xFFFFFFFFu) empty_lf_id = id;
                else if (l == 4 && empty_crlf_id == 0xFFFFFFFFu && read_u32(base + s) == 0x6E5C725Cu) empty_crlf_id = id;
            }
            a_ids[li] = (uint16_t)id;
            mask = ws->ht_mask;
            ht = ws->ht;
        }
    }
    ws->a_len = n_lines;
    return end_pos;
}

static size_t scan_json_old_sse2(WS* ws, const uint8_t* base, size_t start, size_t file_len, int* bare_out) {
    ws->a_len = 0;
    *bare_out = 0;
    if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, 0, 1024) < 0) return (size_t)-1;
    if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, 0, 1024) < 0) return (size_t)-1;
    uint16_t* a_ids = ws->a_ids;
    uint32_t* old_off = ws->old_off;
    size_t a_cap = ws->a_ids_cap;
    size_t o_cap = ws->old_off_cap;
    old_off[0] = (uint32_t)start;

    __m128i v_bs = _mm_set1_epi8('\\');
    __m128i v_qt = _mm_set1_epi8('"');

    size_t p = start, n_lines = 0, end_pos = 0;
    int eof = 0;
    uint32_t empty_lf_id = 0xFFFFFFFFu;
    uint32_t empty_crlf_id = 0xFFFFFFFFu;
    uint64_t th[176];

    while (!eof) {
        size_t b0 = n_lines;
        size_t blim = b0 + 128;
        while (n_lines < blim) {
            if (__builtin_expect(n_lines + 40 > a_cap || n_lines + 41 > o_cap, 0)) {
                if (n_lines >= MAX_NATIVE_LINES) return (size_t)-1;
                if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, n_lines, n_lines + 256) < 0) return (size_t)-1;
                if (ensure_u32(&ws->arena, &ws->old_off, &ws->old_off_cap, n_lines + 1, n_lines + 257) < 0) return (size_t)-1;
                a_ids = ws->a_ids; a_cap = ws->a_ids_cap;
                old_off = ws->old_off; o_cap = ws->old_off_cap;
            }
            if (__builtin_expect(p + 32 <= file_len, 1)) {
                __m128i v0 = _mm_loadu_si128((const __m128i*)(base + p));
                __m128i v1 = _mm_loadu_si128((const __m128i*)(base + p + 16));
                uint32_t m0 = (uint32_t)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v0, v_bs), _mm_cmpeq_epi8(v0, v_qt)));
                uint32_t m1 = (uint32_t)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v1, v_bs), _mm_cmpeq_epi8(v1, v_qt)));
                uint32_t m = m0 | (m1 << 16);
                while (m) {
                    uint32_t tz = (uint32_t)__builtin_ctz(m);
                    size_t hit = p + tz;
                    if (__builtin_expect(base[hit] == '\\', 1)) {
                        if (__builtin_expect(hit + 1 >= file_len, 0)) return (size_t)-1;
                        uint8_t c = base[hit + 1];
                        if (__builtin_expect(c == 'n', 1)) {
                            old_off[++n_lines] = (uint32_t)(hit + 2);
                            m &= m - 1;
                        } else if (c == 'r') {
                            if (hit + 3 < file_len && base[hit + 2] == '\\' && base[hit + 3] == 'n') {
                                old_off[++n_lines] = (uint32_t)(hit + 4);
                                m &= m - 1;
                                if (tz + 2 < 32) { m &= ~(1u << (tz + 2)); }
                                else { p = hit + 4; goto next_vec_s; }
                            } else {
                                m &= m - 1;
                            }
                        } else if (c == '\\' || c == '"') {
                            m &= m - 1;
                            if (tz + 1 < 32) { m &= ~(1u << (tz + 1)); }
                            else { p = hit + 2; goto next_vec_s; }
                        } else if (c == 't' || c == 'b' || c == 'f') {
                            m &= m - 1;
                        } else {
                            return (size_t)-1;
                        }
                    } else {
                        if (hit > (size_t)old_off[n_lines]) {
                            old_off[++n_lines] = (uint32_t)hit;
                            *bare_out = 1;
                        }
                        end_pos = hit + 1;
                        eof = 1;
                        goto flush_tile_s;
                    }
                }
                p += 32;
            next_vec_s:;
            } else {
                while (p < file_len && n_lines < blim) {
                    uint8_t b = base[p];
                    if (b == '"') {
                        if (p > (size_t)old_off[n_lines]) {
                            old_off[++n_lines] = (uint32_t)p;
                            *bare_out = 1;
                        }
                        end_pos = p + 1;
                        eof = 1;
                        goto flush_tile_s;
                    }
                    if (b == '\\') {
                        if (p + 1 >= file_len) return (size_t)-1;
                        uint8_t c = base[p + 1];
                        if (c == 'n') { old_off[++n_lines] = (uint32_t)(p + 2); p += 2; }
                        else if (c == 'r') {
                            if (p + 3 < file_len && base[p + 2] == '\\' && base[p + 3] == 'n') {
                                old_off[++n_lines] = (uint32_t)(p + 4); p += 4;
                            } else p += 2;
                        } else if (c == '\\' || c == '"' || c == 't' || c == 'b' || c == 'f') {
                            p += 2;
                        } else return (size_t)-1;
                    } else p++;
                }
                if (!eof && p >= file_len) return (size_t)-1;
            }
        }
    flush_tile_s:;
        if (n_lines >= MAX_NATIVE_LINES) return (size_t)-1;
        size_t cnt = n_lines - b0;
        if (cnt == 0) break;
        for (size_t k = 0; k < cnt; k++) {
            uint32_t s = old_off[b0 + k], l = old_off[b0 + k + 1] - s;
            th[k] = hash_sample(base + s, l);
        }
        size_t mask = ws->ht_mask;
        uint32_t* ht = ws->ht;
        size_t non_bare_cnt = (eof && *bare_out) ? (cnt - 1) : cnt;
        for (size_t k = 0; k < cnt; k++) {
            if (k + 8 < cnt) __builtin_prefetch(&ht[th[k + 8] & mask], 1, 1);
            uint32_t li = (uint32_t)(b0 + k);
            uint32_t s = old_off[li], l = old_off[li + 1] - s;
            if (k < non_bare_cnt) {
                if (l == 2 && empty_lf_id != 0xFFFFFFFFu) {
                    if (!(ws->ent_meta[empty_lf_id] & MULTI_BIT)) ws->ent_meta[empty_lf_id] |= MULTI_BIT;
                    a_ids[li] = (uint16_t)empty_lf_id;
                    continue;
                }
                if (l == 4 && empty_crlf_id != 0xFFFFFFFFu && read_u32(base + s) == 0x6E5C725Cu) {
                    if (!(ws->ent_meta[empty_crlf_id] & MULTI_BIT)) ws->ent_meta[empty_crlf_id] |= MULTI_BIT;
                    a_ids[li] = (uint16_t)empty_crlf_id;
                    continue;
                }
            }
            uint32_t id = intern_old_sse2(ws, base, s, l, th[k], li);
            if (__builtin_expect(id == 0xFFFFFFFFu, 0)) return (size_t)-1;
            if (k < non_bare_cnt) {
                if (l == 2 && empty_lf_id == 0xFFFFFFFFu) empty_lf_id = id;
                else if (l == 4 && empty_crlf_id == 0xFFFFFFFFu && read_u32(base + s) == 0x6E5C725Cu) empty_crlf_id = id;
            }
            a_ids[li] = (uint16_t)id;
            mask = ws->ht_mask;
            ht = ws->ht;
        }
    }
    ws->a_len = n_lines;
    return end_pos;
}

// ---------------------------------------------------------------------------
// Predictive bulk SIMD `new` scanners (64B AVX2 / 32B SSE2 memcmp + hash_sample)
// ---------------------------------------------------------------------------

__attribute__((target("avx2,bmi")))
static size_t scan_json_new_avx2(WS* ws, const uint8_t* base, size_t start, size_t file_len, int old_bare, int* bare_out) {
    ws->b_len = 0;
    *bare_out = 0;
    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, 0, 1024) < 0) return (size_t)-1;
    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, 0, 1024) < 0) return (size_t)-1;
    uint16_t* b_ids = ws->b_ids;
    uint32_t* new_off = ws->new_off;
    size_t b_cap = ws->b_ids_cap;
    size_t n_cap = ws->new_off_cap;
    new_off[0] = (uint32_t)start;

    size_t n_old = ws->a_len;
    size_t n_old_nl = (old_bare && n_old > 0) ? (n_old - 1) : n_old;
    const uint32_t* ooff = ws->old_off;
    const uint16_t* a_ids = ws->a_ids;
    size_t old_limit = n_old_nl > 0 ? (size_t)ooff[n_old_nl] : 0;

    __m256i v_bs = _mm256_set1_epi8('\\');
    __m256i v_qt = _mm256_set1_epi8('"');

    size_t p = start, pred = 0, m_lines = 0;

    for (;;) {
        if (pred < n_old_nl && p < file_len) {
            size_t old_start = (size_t)ooff[pred];
            size_t avail_old = old_limit - old_start;
            size_t avail_new = file_len - p;
            size_t max_cmp = avail_old < avail_new ? avail_old : avail_new;
            const uint8_t* ab = base + old_start;
            const uint8_t* bb = base + p;
            size_t matched = 0;
            while (matched + 64 <= max_cmp) {
                uint32_t e0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                    _mm256_loadu_si256((const __m256i*)(ab + matched)),
                    _mm256_loadu_si256((const __m256i*)(bb + matched))));
                if (e0 != 0xFFFFFFFFu) { matched += (size_t)__builtin_ctz(~e0); goto done_cmp; }
                uint32_t e1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                    _mm256_loadu_si256((const __m256i*)(ab + matched + 32)),
                    _mm256_loadu_si256((const __m256i*)(bb + matched + 32))));
                if (e1 != 0xFFFFFFFFu) { matched += 32 + (size_t)__builtin_ctz(~e1); goto done_cmp; }
                matched += 64;
            }
            while (matched + 32 <= max_cmp) {
                uint32_t eqm = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                    _mm256_loadu_si256((const __m256i*)(ab + matched)),
                    _mm256_loadu_si256((const __m256i*)(bb + matched))));
                if (eqm == 0xFFFFFFFFu) matched += 32;
                else { matched += (size_t)__builtin_ctz(~eqm); goto done_cmp; }
            }
            while (matched < max_cmp && ab[matched] == bb[matched]) matched++;
        done_cmp:;
            size_t old_reached = old_start + matched;
            if (old_reached >= (size_t)ooff[pred + 1]) {
                intptr_t delta = (intptr_t)p - (intptr_t)old_start;
                size_t k = pred + 1;
                while (k < n_old_nl && (size_t)ooff[k + 1] <= old_reached) k++;
                size_t count = k - pred;
                if (__builtin_expect(m_lines + count >= MAX_NATIVE_LINES, 0)) return (size_t)-1;
                if (__builtin_expect(m_lines + count + 16 > b_cap || m_lines + count + 3 > n_cap, 0)) {
                    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + count + 16) < 0) return (size_t)-1;
                    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + count + 3) < 0) return (size_t)-1;
                    b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
                    new_off = ws->new_off; n_cap = ws->new_off_cap;
                }
                mem_copy_bytes_avx2(b_ids + m_lines, a_ids + pred, count * 2);
                const uint32_t* src_off = ooff + pred + 1;
                uint32_t* dst_off = new_off + m_lines + 1;
                __m256i vd = _mm256_set1_epi32((int32_t)delta);
                size_t i = 0;
                for (; i + 8 <= count; i += 8) {
                    _mm256_storeu_si256((__m256i*)(dst_off + i), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(src_off + i)), vd));
                }
                for (; i < count; i++) dst_off[i] = (uint32_t)((intptr_t)src_off[i] + delta);
                m_lines += count;
                p = (size_t)((intptr_t)ooff[k] + delta);
                pred = k;
            }
        }

        size_t line_start = p;
        int is_eof = 0;
        size_t line_end;
        for (;;) {
            size_t hit;
            if (__builtin_expect(p + 32 <= file_len, 1)) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(base + p));
                uint32_t mask = (uint32_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v, v_bs), _mm256_cmpeq_epi8(v, v_qt)));
                if (mask == 0) { p += 32; continue; }
                hit = p + (size_t)__builtin_ctz(mask);
            } else {
                hit = p;
                while (hit < file_len && base[hit] != '\\' && base[hit] != '"') hit++;
                if (hit >= file_len) return (size_t)-1;
            }
            if (base[hit] == '\\') {
                if (__builtin_expect(hit + 1 >= file_len, 0)) return (size_t)-1;
                uint8_t nxt = base[hit + 1];
                if (nxt == 'n') { line_end = hit + 2; break; }
                else if (nxt == 'r') {
                    if (hit + 3 < file_len && base[hit + 2] == '\\' && base[hit + 3] == 'n') { line_end = hit + 4; break; }
                    p = hit + 2; continue;
                } else if (nxt == '"' || nxt == '\\' || nxt == 't' || nxt == 'b' || nxt == 'f') {
                    p = hit + 2; continue;
                } else return (size_t)-1;
            } else {
                line_end = hit; is_eof = 1; break;
            }
        }

        if (is_eof) {
            if (line_end > line_start) {
                if (m_lines >= MAX_NATIVE_LINES) return (size_t)-1;
                size_t line_len = line_end - line_start;
                uint64_t fh = hash_sample(base + line_start, line_len);
                uint32_t id = intern_new_avx2(ws, base, (uint32_t)line_start, (uint32_t)line_len, fh, (uint32_t)m_lines);
                if (id == 0xFFFFFFFFu) return (size_t)-1;
                if (m_lines + 16 > b_cap || m_lines + 3 > n_cap) {
                    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + 16) < 0) return (size_t)-1;
                    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + 3) < 0) return (size_t)-1;
                    b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
                    new_off = ws->new_off; n_cap = ws->new_off_cap;
                }
                b_ids[m_lines] = (uint16_t)id;
                new_off[m_lines + 1] = (uint32_t)line_end;
                m_lines++;
                *bare_out = 1;
            }
            ws->b_len = m_lines;
            return line_end + 1;
        }

        if (__builtin_expect(m_lines >= MAX_NATIVE_LINES, 0)) return (size_t)-1;
        size_t line_len = line_end - line_start;
        uint64_t fh = hash_sample(base + line_start, line_len);
        uint32_t id = intern_new_avx2(ws, base, (uint32_t)line_start, (uint32_t)line_len, fh, (uint32_t)m_lines);
        if (__builtin_expect(id == 0xFFFFFFFFu, 0)) return (size_t)-1;
        if (__builtin_expect(m_lines + 16 > b_cap || m_lines + 3 > n_cap, 0)) {
            if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + 16) < 0) return (size_t)-1;
            if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + 3) < 0) return (size_t)-1;
            b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
            new_off = ws->new_off; n_cap = ws->new_off_cap;
        }
        b_ids[m_lines] = (uint16_t)id;
        new_off[m_lines + 1] = (uint32_t)line_end;
        m_lines++;
        p = line_end;

        uint32_t op = ws->ent_meta[id];
        if ((op & (NEW_ONLY_BIT | MULTI_BIT)) == 0) {
            pred = (size_t)op + 1;
        } else if (pred < n_old_nl) {
            if (a_ids[pred] == (uint16_t)id) pred += 1;
            else if (pred + 1 < n_old_nl && a_ids[pred + 1] == (uint16_t)id) pred += 2;
        }
    }
}

static size_t scan_json_new_sse2(WS* ws, const uint8_t* base, size_t start, size_t file_len, int old_bare, int* bare_out) {
    ws->b_len = 0;
    *bare_out = 0;
    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, 0, 1024) < 0) return (size_t)-1;
    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, 0, 1024) < 0) return (size_t)-1;
    uint16_t* b_ids = ws->b_ids;
    uint32_t* new_off = ws->new_off;
    size_t b_cap = ws->b_ids_cap;
    size_t n_cap = ws->new_off_cap;
    new_off[0] = (uint32_t)start;

    size_t n_old = ws->a_len;
    size_t n_old_nl = (old_bare && n_old > 0) ? (n_old - 1) : n_old;
    const uint32_t* ooff = ws->old_off;
    const uint16_t* a_ids = ws->a_ids;
    size_t old_limit = n_old_nl > 0 ? (size_t)ooff[n_old_nl] : 0;

    __m128i v_bs = _mm_set1_epi8('\\');
    __m128i v_qt = _mm_set1_epi8('"');

    size_t p = start, pred = 0, m_lines = 0;

    for (;;) {
        if (pred < n_old_nl && p < file_len) {
            size_t old_start = (size_t)ooff[pred];
            size_t avail_old = old_limit - old_start;
            size_t avail_new = file_len - p;
            size_t max_cmp = avail_old < avail_new ? avail_old : avail_new;
            const uint8_t* ab = base + old_start;
            const uint8_t* bb = base + p;
            size_t matched = 0;
            while (matched + 16 <= max_cmp) {
                uint32_t eqm = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(
                    _mm_loadu_si128((const __m128i*)(ab + matched)),
                    _mm_loadu_si128((const __m128i*)(bb + matched))));
                if (eqm == 0xFFFFu) matched += 16;
                else { matched += (size_t)__builtin_ctz((~eqm) & 0xFFFFu); break; }
            }
            if (matched + 16 > max_cmp) {
                while (matched < max_cmp && ab[matched] == bb[matched]) matched++;
            }
            size_t old_reached = old_start + matched;
            if (old_reached >= (size_t)ooff[pred + 1]) {
                intptr_t delta = (intptr_t)p - (intptr_t)old_start;
                size_t k = pred + 1;
                while (k < n_old_nl && (size_t)ooff[k + 1] <= old_reached) k++;
                size_t count = k - pred;
                if (__builtin_expect(m_lines + count >= MAX_NATIVE_LINES, 0)) return (size_t)-1;
                if (__builtin_expect(m_lines + count + 16 > b_cap || m_lines + count + 3 > n_cap, 0)) {
                    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + count + 16) < 0) return (size_t)-1;
                    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + count + 3) < 0) return (size_t)-1;
                    b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
                    new_off = ws->new_off; n_cap = ws->new_off_cap;
                }
                mem_copy_bytes(b_ids + m_lines, a_ids + pred, count * 2);
                const uint32_t* src_off = ooff + pred + 1;
                uint32_t* dst_off = new_off + m_lines + 1;
                __m128i vd = _mm_set1_epi32((int32_t)delta);
                size_t i = 0;
                for (; i + 4 <= count; i += 4) {
                    _mm_storeu_si128((__m128i*)(dst_off + i), _mm_add_epi32(_mm_loadu_si128((const __m128i*)(src_off + i)), vd));
                }
                for (; i < count; i++) dst_off[i] = (uint32_t)((intptr_t)src_off[i] + delta);
                m_lines += count;
                p = (size_t)((intptr_t)ooff[k] + delta);
                pred = k;
            }
        }

        size_t line_start = p;
        int is_eof = 0;
        size_t line_end;
        for (;;) {
            size_t hit;
            if (__builtin_expect(p + 16 <= file_len, 1)) {
                __m128i v = _mm_loadu_si128((const __m128i*)(base + p));
                uint32_t mask = (uint32_t)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v, v_bs), _mm_cmpeq_epi8(v, v_qt)));
                if (mask == 0) { p += 16; continue; }
                hit = p + (size_t)__builtin_ctz(mask);
            } else {
                hit = p;
                while (hit < file_len && base[hit] != '\\' && base[hit] != '"') hit++;
                if (hit >= file_len) return (size_t)-1;
            }
            if (base[hit] == '\\') {
                if (__builtin_expect(hit + 1 >= file_len, 0)) return (size_t)-1;
                uint8_t nxt = base[hit + 1];
                if (nxt == 'n') { line_end = hit + 2; break; }
                else if (nxt == 'r') {
                    if (hit + 3 < file_len && base[hit + 2] == '\\' && base[hit + 3] == 'n') { line_end = hit + 4; break; }
                    p = hit + 2; continue;
                } else if (nxt == '"' || nxt == '\\' || nxt == 't' || nxt == 'b' || nxt == 'f') {
                    p = hit + 2; continue;
                } else return (size_t)-1;
            } else {
                line_end = hit; is_eof = 1; break;
            }
        }

        if (is_eof) {
            if (line_end > line_start) {
                if (m_lines >= MAX_NATIVE_LINES) return (size_t)-1;
                size_t line_len = line_end - line_start;
                uint64_t fh = hash_sample(base + line_start, line_len);
                uint32_t id = intern_new_sse2(ws, base, (uint32_t)line_start, (uint32_t)line_len, fh, (uint32_t)m_lines);
                if (id == 0xFFFFFFFFu) return (size_t)-1;
                if (m_lines + 16 > b_cap || m_lines + 3 > n_cap) {
                    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + 16) < 0) return (size_t)-1;
                    if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + 3) < 0) return (size_t)-1;
                    b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
                    new_off = ws->new_off; n_cap = ws->new_off_cap;
                }
                b_ids[m_lines] = (uint16_t)id;
                new_off[m_lines + 1] = (uint32_t)line_end;
                m_lines++;
                *bare_out = 1;
            }
            ws->b_len = m_lines;
            return line_end + 1;
        }

        if (__builtin_expect(m_lines >= MAX_NATIVE_LINES, 0)) return (size_t)-1;
        size_t line_len = line_end - line_start;
        uint64_t fh = hash_sample(base + line_start, line_len);
        uint32_t id = intern_new_sse2(ws, base, (uint32_t)line_start, (uint32_t)line_len, fh, (uint32_t)m_lines);
        if (__builtin_expect(id == 0xFFFFFFFFu, 0)) return (size_t)-1;
        if (__builtin_expect(m_lines + 16 > b_cap || m_lines + 3 > n_cap, 0)) {
            if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m_lines, m_lines + 16) < 0) return (size_t)-1;
            if (ensure_u32(&ws->arena, &ws->new_off, &ws->new_off_cap, m_lines + 1, m_lines + 3) < 0) return (size_t)-1;
            b_ids = ws->b_ids; b_cap = ws->b_ids_cap;
            new_off = ws->new_off; n_cap = ws->new_off_cap;
        }
        b_ids[m_lines] = (uint16_t)id;
        new_off[m_lines + 1] = (uint32_t)line_end;
        m_lines++;
        p = line_end;

        uint32_t op = ws->ent_meta[id];
        if ((op & (NEW_ONLY_BIT | MULTI_BIT)) == 0) {
            pred = (size_t)op + 1;
        } else if (pred < n_old_nl) {
            if (a_ids[pred] == (uint16_t)id) pred += 1;
            else if (pred + 1 < n_old_nl && a_ids[pred + 1] == (uint16_t)id) pred += 2;
        }
    }
}

// ---------------------------------------------------------------------------
// Exact 16-bit-ID Dominance-Pruned Myers with Fast Width-1 Trace Flushing
// ---------------------------------------------------------------------------

static inline int append_edit(WS* ws, uint32_t* cur_o, uint32_t* cur_p, uint32_t* total_del, uint32_t* total_ins, int is_del) {
    uint32_t co = *cur_o, cp = *cur_p;
    size_t n = ws->chg_len;
    int merge = (n > 0) && (ws->chg[n - 1].o + ws->chg[n - 1].dels == co) && (ws->chg[n - 1].p + ws->chg[n - 1].ins == cp);
    if (!merge) {
        if (__builtin_expect(n >= ws->chg_cap, 0) && ensure_chg(ws, n + 1) < 0) return -1;
        ws->chg[n] = (Chg){ co, cp, 0, 0 };
        ws->chg_len = n + 1;
    }
    Chg* lc = &ws->chg[ws->chg_len - 1];
    if (is_del) { lc->dels += 1; *cur_o = co + 1; *total_del += 1; }
    else        { lc->ins  += 1; *cur_p = cp + 1; *total_ins += 1; }
    return 0;
}

static int flush_segment(
    WS* ws, size_t rel_d, int32_t end_x, int32_t end_k, int32_t seg_x,
    uint32_t* cur_o, uint32_t* cur_p, uint32_t* total_del, uint32_t* total_ins
) {
    const uint32_t* rbase = ws->row_base;
    const int32_t* rlo = ws->row_lo;
    const int32_t* trx = ws->tr_x;
    const uint8_t* trc = ws->tr_c;

    // Fast path for rel_d == 1 (very common single del/ins hunk!)
    if (rel_d == 1) {
        size_t idx = (size_t)rbase[1] + (size_t)((end_k - rlo[1]) >> 1);
        uint8_t is_del = trc[idx];
        int32_t start_x = is_del ? (seg_x + 1) : seg_x;
        if (append_edit(ws, cur_o, cur_p, total_del, total_ins, is_del) < 0) return -1;
        uint32_t snake = (uint32_t)(end_x - start_x);
        *cur_o += snake; *cur_p += snake;
        return 0;
    }

    if (rel_d * 2 + 4 > ws->seg_cap) {
        if (ensure_u32(&ws->arena, &ws->seg_ops, &ws->seg_cap, 0, rel_d * 2 + 4) < 0) return -1;
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
        cx = prev_x;
        ck = prev_k;
    }

    uint32_t co = *cur_o, cp = *cur_p, td = *total_del, ti = *total_ins;
    while (nops > 0) {
        uint32_t op = sops[--nops];
        uint32_t kind = op >> 30;
        uint32_t count = op & 0x3FFFFFFFu;
        if (kind == 0) { co += count; cp += count; continue; }
        size_t n = ws->chg_len;
        int merge = (n > 0) && (ws->chg[n - 1].o + ws->chg[n - 1].dels == co) && (ws->chg[n - 1].p + ws->chg[n - 1].ins == cp);
        if (!merge) {
            if (n >= ws->chg_cap && ensure_chg(ws, n + 1) < 0) return -1;
            ws->chg[n] = (Chg){ co, cp, 0, 0 };
            ws->chg_len = n + 1;
        }
        Chg* lc = &ws->chg[ws->chg_len - 1];
        if (kind == 1) { lc->dels += count; co += count; td += count; }
        else           { lc->ins  += count; cp += count; ti += count; }
    }
    *cur_o = co; *cur_p = cp; *total_del = td; *total_ins = ti;
    return 0;
}

__attribute__((target("avx2,bmi")))
static int run_myers_avx2(WS* ws, uint32_t* out_del, uint32_t* out_ins, int first_case) {
    size_t n0 = ws->a_len, m0 = ws->b_len;
    uint16_t* a_all = ws->a_ids;
    uint16_t* b_all = ws->b_ids;

    size_t prefix = 0, min_nm = n0 < m0 ? n0 : m0;
    while (prefix + 16 <= min_nm && (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(
        _mm256_loadu_si256((const __m256i*)(a_all + prefix)),
        _mm256_loadu_si256((const __m256i*)(b_all + prefix)))) == 0xFFFFFFFFu) prefix += 16;
    while (prefix < min_nm && a_all[prefix] == b_all[prefix]) prefix++;
    size_t suffix = 0, rem_a = n0 - prefix, rem_b = m0 - prefix, min_rem = rem_a < rem_b ? rem_a : rem_b;
    while (suffix < min_rem && a_all[n0 - 1 - suffix] == b_all[m0 - 1 - suffix]) suffix++;

    size_t n = n0 - prefix - suffix;
    size_t m = m0 - prefix - suffix;
    ws->chg_len = 0;

    if (n == 0 && m == 0) { *out_del = 0; *out_ins = 0; return 0; }
    if (ensure_chg(ws, 4096) < 0) return -1;
    if (n == 0) {
        ws->chg[0] = (Chg){ (uint32_t)prefix, (uint32_t)prefix, 0, (uint32_t)m };
        ws->chg_len = 1; *out_del = 0; *out_ins = (uint32_t)m; return 0;
    }
    if (m == 0) {
        ws->chg[0] = (Chg){ (uint32_t)prefix, (uint32_t)prefix, (uint32_t)n, 0 };
        ws->chg_len = 1; *out_del = (uint32_t)n; *out_ins = 0; return 0;
    }

    if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, n0, n0 + 18) < 0) return -1;
    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m0, m0 + 18) < 0) return -1;
    uint16_t* a_ptr = ws->a_ids + prefix;
    uint16_t* b_ptr = ws->b_ids + prefix;
    a_ptr[n] = 0xFFFEu;
    b_ptr[m] = 0xFFFFu;
    int32_t n_i = (int32_t)n, m_i = (int32_t)m;

    size_t num_ids = ws->next_id;
    if (num_ids > ws->last16_cap) {
        size_t nc = ws->last16_cap ? ws->last16_cap * 2 : ((num_ids + 63) & ~(size_t)63);
        while (nc < num_ids) nc *= 2;
        ws->last_a16 = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        ws->last_b16 = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        if (!ws->last_a16 || !ws->last_b16) return -1;
        ws->last16_cap = nc;
    }
    uint16_t* __restrict last_a16 = ws->last_a16;
    uint16_t* __restrict last_b16 = ws->last_b16;
    if (!first_case) {
        mem_zero_avx2(last_a16, num_ids * 2);
        mem_zero_avx2(last_b16, num_ids * 2);
    }
    for (size_t i = 0; i < n; i++) last_a16[a_ptr[i]] = (uint16_t)(i + 1);
    for (size_t j = 0; j < m; j++) last_b16[b_ptr[j]] = (uint16_t)(j + 1);

    if (ws->v_half == 0 && ensure_v(ws, 1024) < 0) return -1;

    uint32_t cur_o = (uint32_t)prefix, cur_p = (uint32_t)prefix, total_del = 0, total_ins = 0;
    int32_t k_base = 0, k_lo = 0, k_hi = 0, seg_x = 0;
    size_t rel_d = 0, tr_ptr = 0;
    ws->v[ws->v_half] = 0;

    size_t max_edit = n + m;
    for (size_t d = 1; d <= max_edit; d++) {
        rel_d++;
        int32_t next_lo = k_lo - 1;
        int32_t next_hi = k_hi + 1;

        int32_t nl_d = k_base - (next_lo - 1);
        int32_t nr_d = (next_hi + 1) - k_base;
        size_t need_half = (size_t)(nl_d > nr_d ? nl_d : nr_d) + 2;
        if (__builtin_expect(need_half > ws->v_half, 0)) {
            if (ensure_v(ws, need_half) < 0) return -1;
        }
        size_t row_len = (size_t)(((next_hi - next_lo) >> 1) + 1);
        if (__builtin_expect(tr_ptr + row_len > ws->tr_cap || rel_d + 1 > ws->row_cap, 0)) {
            if (ensure_trace(ws, tr_ptr, tr_ptr + row_len, rel_d, rel_d + 1) < 0) return -1;
        }

        ws->row_base[rel_d] = (uint32_t)tr_ptr;
        ws->row_lo[rel_d] = next_lo;

        int32_t* v_zero = ws->v + ((ptrdiff_t)ws->v_half - (ptrdiff_t)k_base);
        int32_t* tr_xp = ws->tr_x + tr_ptr;
        uint8_t* tr_cp = ws->tr_c + tr_ptr;

        int32_t new_lo = next_hi + 1, new_hi = next_lo - 1;
        int32_t best_sum = -1, best_k = 0, best_x = 0, best_y = 0, best_snk = 0;
        int done = 0;

        int32_t* vp = v_zero + next_lo;
        for (int32_t k = next_lo; k <= next_hi; k += 2, vp += 2, tr_xp++, tr_cp++) {
            int32_t rem_x = vp[-1];
            int32_t add_x = vp[1];
            int can_rem = (rem_x != DEAD) & (rem_x < n_i);
            int can_add = (add_x != DEAD) & ((add_x - k) <= m_i);
            if (!(can_rem | can_add)) {
                *vp = DEAD;
                *tr_xp = DEAD;
                continue;
            }
            int32_t by_del = can_rem ? (rem_x + 1) : -2;
            int32_t by_ins = can_add ? add_x : -2;
            int32_t x; uint8_t is_del;
            if (by_del >= by_ins) { x = by_del; is_del = 1; }
            else                  { x = by_ins; is_del = 0; }
            int32_t sx = x;
            int32_t y = x - k;
            if (a_ptr[x] == b_ptr[y]) {
                x++; y++;
                for (;;) {
                    uint32_t eqm = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(
                        _mm256_loadu_si256((const __m256i*)(a_ptr + x)),
                        _mm256_loadu_si256((const __m256i*)(b_ptr + y))));
                    if (eqm == 0xFFFFFFFFu) { x += 16; y += 16; }
                    else { int32_t adv = (int32_t)(__builtin_ctz(~eqm) >> 1); x += adv; y += adv; break; }
                }
            }
            *vp = x;
            *tr_xp = x;
            *tr_cp = is_del;
            if (k < new_lo) new_lo = k;
            if (k > new_hi) new_hi = k;
            int32_t sum = x + y;
            if (sum > best_sum) { best_sum = sum; best_k = k; best_x = x; best_y = y; best_snk = x - sx; }
            if (__builtin_expect(x >= n_i && y >= m_i, 0)) { done = 1; break; }
        }
        tr_ptr += row_len;

        for (int32_t ck = k_lo; ck <= k_hi; ck += 2) v_zero[ck] = DEAD;

        if (done) {
            for (int32_t ck2 = new_lo; ck2 <= new_hi; ck2 += 2) v_zero[ck2] = DEAD;
            if (flush_segment(ws, rel_d, n_i, n_i - m_i, seg_x, &cur_o, &cur_p, &total_del, &total_ins) < 0) return -1;
            break;
        }

        // Exact dominance pruning: run whenever best_k landed a snake or every 8 steps (skips wasted random last_a/b reads during non-snake stretches!)
        if (best_snk > 0 || (rel_d & 7) == 0) {
            int32_t p_lo = best_k, p_hi = best_k;
            uint16_t by1 = (uint16_t)(best_y + 1);
            int32_t b_bad = 0, s = 0;
            for (int32_t kl = best_k - 2; kl >= new_lo; kl -= 2) {
                int32_t i2 = best_x - s - 2;
                if (i2 < 0) {
                    for (int32_t q = new_lo; q <= kl; q += 2) {
                        if (v_zero[q] != DEAD) { p_lo = q; break; }
                    }
                    break;
                }
                b_bad += (last_b16[a_ptr[i2 + 1]] >= by1) + (last_b16[a_ptr[i2]] >= by1);
                s += 2;
                int32_t x = v_zero[kl];
                if (x != DEAD) {
                    if (x <= best_x && (x - kl) <= best_y && (b_bad << 1) < s) v_zero[kl] = DEAD;
                    else p_lo = kl;
                }
            }
            uint16_t bx1 = (uint16_t)(best_x + 1);
            int32_t a_bad = 0; s = 0;
            for (int32_t kr = best_k + 2; kr <= new_hi; kr += 2) {
                int32_t j2 = best_y - s - 2;
                if (j2 < 0) {
                    for (int32_t q = new_hi; q >= kr; q -= 2) {
                        if (v_zero[q] != DEAD) { p_hi = q; break; }
                    }
                    break;
                }
                a_bad += (last_a16[b_ptr[j2 + 1]] >= bx1) + (last_a16[b_ptr[j2]] >= bx1);
                s += 2;
                int32_t x = v_zero[kr];
                if (x != DEAD) {
                    if (x <= best_x && (x - kr) <= best_y && (a_bad << 1) < s) v_zero[kr] = DEAD;
                    else p_hi = kr;
                }
            }
            k_lo = p_lo; k_hi = p_hi;
        } else {
            k_lo = new_lo; k_hi = new_hi;
        }

        if (k_lo == k_hi) {
            int32_t cx = v_zero[k_lo];
            v_zero[k_lo] = DEAD;
            if (flush_segment(ws, rel_d, cx, k_lo, seg_x, &cur_o, &cur_p, &total_del, &total_ins) < 0) return -1;
            seg_x = cx;
            k_base = k_lo;
            ws->v[ws->v_half] = cx;
            rel_d = 0;
            tr_ptr = 0;
        }
    }

    *out_del = total_del;
    *out_ins = total_ins;
    return 0;
}

static int run_myers_sse2(WS* ws, uint32_t* out_del, uint32_t* out_ins, int first_case) {
    size_t n0 = ws->a_len, m0 = ws->b_len;
    uint16_t* a_all = ws->a_ids;
    uint16_t* b_all = ws->b_ids;

    size_t prefix = 0, min_nm = n0 < m0 ? n0 : m0;
    while (prefix + 8 <= min_nm && (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(
        _mm_loadu_si128((const __m128i*)(a_all + prefix)),
        _mm_loadu_si128((const __m128i*)(b_all + prefix)))) == 0xFFFFu) prefix += 8;
    while (prefix < min_nm && a_all[prefix] == b_all[prefix]) prefix++;
    size_t suffix = 0, rem_a = n0 - prefix, rem_b = m0 - prefix, min_rem = rem_a < rem_b ? rem_a : rem_b;
    while (suffix < min_rem && a_all[n0 - 1 - suffix] == b_all[m0 - 1 - suffix]) suffix++;

    size_t n = n0 - prefix - suffix;
    size_t m = m0 - prefix - suffix;
    ws->chg_len = 0;

    if (n == 0 && m == 0) { *out_del = 0; *out_ins = 0; return 0; }
    if (ensure_chg(ws, 4096) < 0) return -1;
    if (n == 0) {
        ws->chg[0] = (Chg){ (uint32_t)prefix, (uint32_t)prefix, 0, (uint32_t)m };
        ws->chg_len = 1; *out_del = 0; *out_ins = (uint32_t)m; return 0;
    }
    if (m == 0) {
        ws->chg[0] = (Chg){ (uint32_t)prefix, (uint32_t)prefix, (uint32_t)n, 0 };
        ws->chg_len = 1; *out_del = (uint32_t)n; *out_ins = 0; return 0;
    }

    if (ensure_u16(&ws->arena, &ws->a_ids, &ws->a_ids_cap, n0, n0 + 18) < 0) return -1;
    if (ensure_u16(&ws->arena, &ws->b_ids, &ws->b_ids_cap, m0, m0 + 18) < 0) return -1;
    uint16_t* a_ptr = ws->a_ids + prefix;
    uint16_t* b_ptr = ws->b_ids + prefix;
    a_ptr[n] = 0xFFFEu;
    b_ptr[m] = 0xFFFFu;
    int32_t n_i = (int32_t)n, m_i = (int32_t)m;

    size_t num_ids = ws->next_id;
    if (num_ids > ws->last16_cap) {
        size_t nc = ws->last16_cap ? ws->last16_cap * 2 : ((num_ids + 63) & ~(size_t)63);
        while (nc < num_ids) nc *= 2;
        ws->last_a16 = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        ws->last_b16 = (uint16_t*)arena_alloc(&ws->arena, nc * 2, 32);
        if (!ws->last_a16 || !ws->last_b16) return -1;
        ws->last16_cap = nc;
    }
    uint16_t* __restrict last_a16 = ws->last_a16;
    uint16_t* __restrict last_b16 = ws->last_b16;
    if (!first_case) {
        mem_zero_bytes(last_a16, num_ids * 2);
        mem_zero_bytes(last_b16, num_ids * 2);
    }
    for (size_t i = 0; i < n; i++) last_a16[a_ptr[i]] = (uint16_t)(i + 1);
    for (size_t j = 0; j < m; j++) last_b16[b_ptr[j]] = (uint16_t)(j + 1);

    if (ws->v_half == 0 && ensure_v(ws, 1024) < 0) return -1;

    uint32_t cur_o = (uint32_t)prefix, cur_p = (uint32_t)prefix, total_del = 0, total_ins = 0;
    int32_t k_base = 0, k_lo = 0, k_hi = 0, seg_x = 0;
    size_t rel_d = 0, tr_ptr = 0;
    ws->v[ws->v_half] = 0;

    size_t max_edit = n + m;
    for (size_t d = 1; d <= max_edit; d++) {
        rel_d++;
        int32_t next_lo = k_lo - 1;
        int32_t next_hi = k_hi + 1;

        int32_t nl_d = k_base - (next_lo - 1);
        int32_t nr_d = (next_hi + 1) - k_base;
        size_t need_half = (size_t)(nl_d > nr_d ? nl_d : nr_d) + 2;
        if (__builtin_expect(need_half > ws->v_half, 0)) {
            if (ensure_v(ws, need_half) < 0) return -1;
        }
        size_t row_len = (size_t)(((next_hi - next_lo) >> 1) + 1);
        if (__builtin_expect(tr_ptr + row_len > ws->tr_cap || rel_d + 1 > ws->row_cap, 0)) {
            if (ensure_trace(ws, tr_ptr, tr_ptr + row_len, rel_d, rel_d + 1) < 0) return -1;
        }

        ws->row_base[rel_d] = (uint32_t)tr_ptr;
        ws->row_lo[rel_d] = next_lo;

        int32_t* v_zero = ws->v + ((ptrdiff_t)ws->v_half - (ptrdiff_t)k_base);
        int32_t* tr_xp = ws->tr_x + tr_ptr;
        uint8_t* tr_cp = ws->tr_c + tr_ptr;

        int32_t new_lo = next_hi + 1, new_hi = next_lo - 1;
        int32_t best_sum = -1, best_k = 0, best_x = 0, best_y = 0, best_snk = 0;
        int done = 0;

        int32_t* vp = v_zero + next_lo;
        for (int32_t k = next_lo; k <= next_hi; k += 2, vp += 2, tr_xp++, tr_cp++) {
            int32_t rem_x = vp[-1];
            int32_t add_x = vp[1];
            int can_rem = (rem_x != DEAD) & (rem_x < n_i);
            int can_add = (add_x != DEAD) & ((add_x - k) <= m_i);
            if (!(can_rem | can_add)) {
                *vp = DEAD;
                *tr_xp = DEAD;
                continue;
            }
            int32_t by_del = can_rem ? (rem_x + 1) : -2;
            int32_t by_ins = can_add ? add_x : -2;
            int32_t x; uint8_t is_del;
            if (by_del >= by_ins) { x = by_del; is_del = 1; }
            else                  { x = by_ins; is_del = 0; }
            int32_t sx = x;
            int32_t y = x - k;
            if (a_ptr[x] == b_ptr[y]) {
                x++; y++;
                for (;;) {
                    uint32_t eqm = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(
                        _mm_loadu_si128((const __m128i*)(a_ptr + x)),
                        _mm_loadu_si128((const __m128i*)(b_ptr + y))));
                    if (eqm == 0xFFFFu) { x += 8; y += 8; }
                    else { int32_t adv = (int32_t)(__builtin_ctz((~eqm) & 0xFFFFu) >> 1); x += adv; y += adv; break; }
                }
            }
            *vp = x;
            *tr_xp = x;
            *tr_cp = is_del;
            if (k < new_lo) new_lo = k;
            if (k > new_hi) new_hi = k;
            int32_t sum = x + y;
            if (sum > best_sum) { best_sum = sum; best_k = k; best_x = x; best_y = y; best_snk = x - sx; }
            if (__builtin_expect(x >= n_i && y >= m_i, 0)) { done = 1; break; }
        }
        tr_ptr += row_len;

        for (int32_t ck = k_lo; ck <= k_hi; ck += 2) v_zero[ck] = DEAD;

        if (done) {
            for (int32_t ck2 = new_lo; ck2 <= new_hi; ck2 += 2) v_zero[ck2] = DEAD;
            if (flush_segment(ws, rel_d, n_i, n_i - m_i, seg_x, &cur_o, &cur_p, &total_del, &total_ins) < 0) return -1;
            break;
        }

        if (best_snk > 0 || (rel_d & 7) == 0) {
            int32_t p_lo = best_k, p_hi = best_k;
            uint16_t by1 = (uint16_t)(best_y + 1);
            int32_t b_bad = 0, s = 0;
            for (int32_t kl = best_k - 2; kl >= new_lo; kl -= 2) {
                int32_t i2 = best_x - s - 2;
                if (i2 < 0) {
                    for (int32_t q = new_lo; q <= kl; q += 2) {
                        if (v_zero[q] != DEAD) { p_lo = q; break; }
                    }
                    break;
                }
                b_bad += (last_b16[a_ptr[i2 + 1]] >= by1) + (last_b16[a_ptr[i2]] >= by1);
                s += 2;
                int32_t x = v_zero[kl];
                if (x != DEAD) {
                    if (x <= best_x && (x - kl) <= best_y && (b_bad << 1) < s) v_zero[kl] = DEAD;
                    else p_lo = kl;
                }
            }
            uint16_t bx1 = (uint16_t)(best_x + 1);
            int32_t a_bad = 0; s = 0;
            for (int32_t kr = best_k + 2; kr <= new_hi; kr += 2) {
                int32_t j2 = best_y - s - 2;
                if (j2 < 0) {
                    for (int32_t q = new_hi; q >= kr; q -= 2) {
                        if (v_zero[q] != DEAD) { p_hi = q; break; }
                    }
                    break;
                }
                a_bad += (last_a16[b_ptr[j2 + 1]] >= bx1) + (last_a16[b_ptr[j2]] >= bx1);
                s += 2;
                int32_t x = v_zero[kr];
                if (x != DEAD) {
                    if (x <= best_x && (x - kr) <= best_y && (a_bad << 1) < s) v_zero[kr] = DEAD;
                    else p_hi = kr;
                }
            }
            k_lo = p_lo; k_hi = p_hi;
        } else {
            k_lo = new_lo; k_hi = new_hi;
        }

        if (k_lo == k_hi) {
            int32_t cx = v_zero[k_lo];
            v_zero[k_lo] = DEAD;
            if (flush_segment(ws, rel_d, cx, k_lo, seg_x, &cur_o, &cur_p, &total_del, &total_ins) < 0) return -1;
            seg_x = cx;
            k_base = k_lo;
            ws->v[ws->v_half] = cx;
            rel_d = 0;
            tr_ptr = 0;
        }
    }

    *out_del = total_del;
    *out_ins = total_ins;
    return 0;
}

// ---------------------------------------------------------------------------
// Fast branchless-overcopy JSON hunk formatter (2-digit LUT + AVX2/SSE2)
// ---------------------------------------------------------------------------

static const char DIGITS2[201] =
    "0001020304050607080910111213141516171819"
    "2021222324252627282930313233343536373839"
    "4041424344454647484950515253545556575859"
    "6061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

static inline uint8_t* write_u32_dec(uint8_t* dst, uint32_t x) {
    if (x < 10) { *dst = (uint8_t)('0' + x); return dst + 1; }
    if (x < 100) {
        dst[0] = (uint8_t)DIGITS2[x * 2];
        dst[1] = (uint8_t)DIGITS2[x * 2 + 1];
        return dst + 2;
    }
    uint8_t tmp[10];
    int p = 10;
    while (x >= 100) {
        uint32_t r = x % 100; x /= 100;
        p -= 2;
        tmp[p] = (uint8_t)DIGITS2[r * 2];
        tmp[p + 1] = (uint8_t)DIGITS2[r * 2 + 1];
    }
    if (x >= 10) { p -= 2; tmp[p] = (uint8_t)DIGITS2[x * 2]; tmp[p + 1] = (uint8_t)DIGITS2[x * 2 + 1]; }
    else { tmp[--p] = (uint8_t)('0' + x); }
    for (int i = p; i < 10; i++) *dst++ = tmp[i];
    return dst;
}

__attribute__((target("avx2")))
static int format_hunks_json_avx2(WS* ws, const uint8_t* base, size_t file_len, int old_bare, int new_bare, uint32_t total_del, uint32_t total_ins) {
    uint32_t n0 = (uint32_t)ws->a_len;
    uint32_t m0 = (uint32_t)ws->b_len;
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
    dst = copy_short_avx2(dst, (const uint8_t*)"makefun-diff/1 old=", 19);
    dst = write_u32_dec(dst, n0);
    dst = copy_short_avx2(dst, (const uint8_t*)" new=", 5);
    dst = write_u32_dec(dst, m0);
    dst = copy_short_avx2(dst, (const uint8_t*)" hunks=", 7);
    dst = write_u32_dec(dst, num_hunks);
    dst = copy_short_avx2(dst, (const uint8_t*)" deleted=", 9);
    dst = write_u32_dec(dst, total_del);
    dst = copy_short_avx2(dst, (const uint8_t*)" inserted=", 10);
    dst = write_u32_dec(dst, total_ins);
    dst[0] = '\\'; dst[1] = 'n'; dst += 2;
    ws->out_len = (size_t)(dst - ws->out);

    if (n_chg == 0) return 0;

    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    size_t safe_lim = file_len > 64 ? (file_len - 64) : 0;

    size_t h_start = 0;
    while (h_start < n_chg) {
        size_t h_end = h_start + 1;
        while (h_end < n_chg && chg[h_end].o - (chg[h_end - 1].o + chg[h_end - 1].dels) <= 6) h_end++;

        Chg first = chg[h_start], last = chg[h_end - 1];
        uint32_t old_start = first.o > 3 ? (first.o - 3) : 0;
        uint32_t new_start = first.p - (first.o - old_start);
        uint32_t last_o_end = last.o + last.dels;
        uint32_t last_p_end = last.p + last.ins;
        uint32_t rem = n0 - last_o_end;
        uint32_t after = rem < 3 ? rem : 3;
        uint32_t old_end = last_o_end + after;
        uint32_t new_end = last_p_end + after;

        size_t old_bytes = (size_t)(ooff[old_end] - ooff[old_start]);
        size_t new_bytes = (size_t)(noff[new_end] - noff[new_start]);
        size_t line_cnt = (size_t)((old_end - old_start) + (new_end - new_start));
        if (ensure_out(ws, ws->out_len + 128 + old_bytes + new_bytes + line_cnt + 64) < 0) return -1;

        dst = ws->out + ws->out_len;
        dst[0] = '@'; dst[1] = ' ';
        dst = write_u32_dec(dst + 2, old_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, old_end - old_start); *dst++ = ' ';
        dst = write_u32_dec(dst, new_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, new_end - new_start);
        dst[0] = '\\'; dst[1] = 'n'; dst += 2;

        int hunk_old_bare = old_bare && (old_end == n0);
        int hunk_new_bare = new_bare && (new_end == m0);
        int fast_ok = (!hunk_old_bare) && (!hunk_new_bare) &&
                      ((size_t)ooff[old_end] <= safe_lim) && ((size_t)noff[new_end] <= safe_lim);

        uint32_t at = old_start;
        if (__builtin_expect(fast_ok, 1)) {
            for (size_t ci = h_start; ci < h_end; ci++) {
                Chg c = chg[ci];
                for (; at < c.o; at++) {
                    uint32_t s = ooff[at], e = ooff[at + 1];
                    *dst++ = '=';
                    dst = copy_line_fast_avx2(dst, base + s, e - s);
                }
                for (uint32_t k = 0; k < c.dels; k++) {
                    uint32_t idx = c.o + k, s = ooff[idx], e = ooff[idx + 1];
                    *dst++ = '-';
                    dst = copy_line_fast_avx2(dst, base + s, e - s);
                }
                for (uint32_t k = 0; k < c.ins; k++) {
                    uint32_t idx = c.p + k, s = noff[idx], e = noff[idx + 1];
                    *dst++ = '+';
                    dst = copy_line_fast_avx2(dst, base + s, e - s);
                }
                at = c.o + c.dels;
            }
            for (; at < old_end; at++) {
                uint32_t s = ooff[at], e = ooff[at + 1];
                *dst++ = '=';
                dst = copy_line_fast_avx2(dst, base + s, e - s);
            }
        } else {
            for (size_t ci = h_start; ci < h_end; ci++) {
                Chg c = chg[ci];
                for (; at < c.o; at++) {
                    uint32_t s = ooff[at], e = ooff[at + 1];
                    *dst++ = '=';
                    dst = copy_short_avx2(dst, base + s, e - s);
                    if (hunk_old_bare && at + 1 == n0) dst = copy_short_avx2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                for (uint32_t k = 0; k < c.dels; k++) {
                    uint32_t idx = c.o + k, s = ooff[idx], e = ooff[idx + 1];
                    *dst++ = '-';
                    dst = copy_short_avx2(dst, base + s, e - s);
                    if (hunk_old_bare && idx + 1 == n0) dst = copy_short_avx2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                for (uint32_t k = 0; k < c.ins; k++) {
                    uint32_t idx = c.p + k, s = noff[idx], e = noff[idx + 1];
                    *dst++ = '+';
                    dst = copy_short_avx2(dst, base + s, e - s);
                    if (hunk_new_bare && idx + 1 == m0) dst = copy_short_avx2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                at = c.o + c.dels;
            }
            for (; at < old_end; at++) {
                uint32_t s = ooff[at], e = ooff[at + 1];
                *dst++ = '=';
                dst = copy_short_avx2(dst, base + s, e - s);
                if (hunk_old_bare && at + 1 == n0) dst = copy_short_avx2(dst, (const uint8_t*)"\\n%\\n", 5);
            }
        }
        ws->out_len = (size_t)(dst - ws->out);
        h_start = h_end;
    }
    return 0;
}

static int format_hunks_json_sse2(WS* ws, const uint8_t* base, size_t file_len, int old_bare, int new_bare, uint32_t total_del, uint32_t total_ins) {
    uint32_t n0 = (uint32_t)ws->a_len;
    uint32_t m0 = (uint32_t)ws->b_len;
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
    dst[0] = '\\'; dst[1] = 'n'; dst += 2;
    ws->out_len = (size_t)(dst - ws->out);

    if (n_chg == 0) return 0;

    const uint32_t* ooff = ws->old_off;
    const uint32_t* noff = ws->new_off;
    size_t safe_lim = file_len > 64 ? (file_len - 64) : 0;

    size_t h_start = 0;
    while (h_start < n_chg) {
        size_t h_end = h_start + 1;
        while (h_end < n_chg && chg[h_end].o - (chg[h_end - 1].o + chg[h_end - 1].dels) <= 6) h_end++;

        Chg first = chg[h_start], last = chg[h_end - 1];
        uint32_t old_start = first.o > 3 ? (first.o - 3) : 0;
        uint32_t new_start = first.p - (first.o - old_start);
        uint32_t last_o_end = last.o + last.dels;
        uint32_t last_p_end = last.p + last.ins;
        uint32_t rem = n0 - last_o_end;
        uint32_t after = rem < 3 ? rem : 3;
        uint32_t old_end = last_o_end + after;
        uint32_t new_end = last_p_end + after;

        size_t old_bytes = (size_t)(ooff[old_end] - ooff[old_start]);
        size_t new_bytes = (size_t)(noff[new_end] - noff[new_start]);
        size_t line_cnt = (size_t)((old_end - old_start) + (new_end - new_start));
        if (ensure_out(ws, ws->out_len + 128 + old_bytes + new_bytes + line_cnt + 64) < 0) return -1;

        dst = ws->out + ws->out_len;
        dst[0] = '@'; dst[1] = ' ';
        dst = write_u32_dec(dst + 2, old_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, old_end - old_start); *dst++ = ' ';
        dst = write_u32_dec(dst, new_start + 1); *dst++ = ' ';
        dst = write_u32_dec(dst, new_end - new_start);
        dst[0] = '\\'; dst[1] = 'n'; dst += 2;

        int hunk_old_bare = old_bare && (old_end == n0);
        int hunk_new_bare = new_bare && (new_end == m0);
        int fast_ok = (!hunk_old_bare) && (!hunk_new_bare) &&
                      ((size_t)ooff[old_end] <= safe_lim) && ((size_t)noff[new_end] <= safe_lim);

        uint32_t at = old_start;
        if (__builtin_expect(fast_ok, 1)) {
            for (size_t ci = h_start; ci < h_end; ci++) {
                Chg c = chg[ci];
                for (; at < c.o; at++) {
                    uint32_t s = ooff[at], e = ooff[at + 1];
                    *dst++ = '=';
                    dst = copy_line_fast_sse2(dst, base + s, e - s);
                }
                for (uint32_t k = 0; k < c.dels; k++) {
                    uint32_t idx = c.o + k, s = ooff[idx], e = ooff[idx + 1];
                    *dst++ = '-';
                    dst = copy_line_fast_sse2(dst, base + s, e - s);
                }
                for (uint32_t k = 0; k < c.ins; k++) {
                    uint32_t idx = c.p + k, s = noff[idx], e = noff[idx + 1];
                    *dst++ = '+';
                    dst = copy_line_fast_sse2(dst, base + s, e - s);
                }
                at = c.o + c.dels;
            }
            for (; at < old_end; at++) {
                uint32_t s = ooff[at], e = ooff[at + 1];
                *dst++ = '=';
                dst = copy_line_fast_sse2(dst, base + s, e - s);
            }
        } else {
            for (size_t ci = h_start; ci < h_end; ci++) {
                Chg c = chg[ci];
                for (; at < c.o; at++) {
                    uint32_t s = ooff[at], e = ooff[at + 1];
                    *dst++ = '=';
                    dst = copy_short_sse2(dst, base + s, e - s);
                    if (hunk_old_bare && at + 1 == n0) dst = copy_short_sse2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                for (uint32_t k = 0; k < c.dels; k++) {
                    uint32_t idx = c.o + k, s = ooff[idx], e = ooff[idx + 1];
                    *dst++ = '-';
                    dst = copy_short_sse2(dst, base + s, e - s);
                    if (hunk_old_bare && idx + 1 == n0) dst = copy_short_sse2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                for (uint32_t k = 0; k < c.ins; k++) {
                    uint32_t idx = c.p + k, s = noff[idx], e = noff[idx + 1];
                    *dst++ = '+';
                    dst = copy_short_sse2(dst, base + s, e - s);
                    if (hunk_new_bare && idx + 1 == m0) dst = copy_short_sse2(dst, (const uint8_t*)"\\n%\\n", 5);
                }
                at = c.o + c.dels;
            }
            for (; at < old_end; at++) {
                uint32_t s = ooff[at], e = ooff[at + 1];
                *dst++ = '=';
                dst = copy_short_sse2(dst, base + s, e - s);
                if (hunk_old_bare && at + 1 == n0) dst = copy_short_sse2(dst, (const uint8_t*)"\\n%\\n", 5);
            }
        }
        ws->out_len = (size_t)(dst - ws->out);
        h_start = h_end;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// JSON wrapper parser + full file pipeline
// ---------------------------------------------------------------------------

static inline size_t skip_ws(const uint8_t* base, size_t p, size_t len) {
    while (p < len && (base[p] == ' ' || base[p] == '\n' || base[p] == '\r' || base[p] == '\t')) p++;
    return p;
}

static size_t skip_json_string(const uint8_t* base, size_t p, size_t len) {
    __m128i v_bs = _mm_set1_epi8('\\');
    __m128i v_qt = _mm_set1_epi8('"');
    for (;;) {
        size_t hit;
        if (__builtin_expect(p + 16 <= len, 1)) {
            __m128i v = _mm_loadu_si128((const __m128i*)(base + p));
            uint32_t mask = (uint32_t)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v, v_bs), _mm_cmpeq_epi8(v, v_qt)));
            if (mask == 0) { p += 16; continue; }
            hit = p + (size_t)__builtin_ctz(mask);
        } else {
            hit = p;
            while (hit < len && base[hit] != '\\' && base[hit] != '"') hit++;
            if (hit >= len) return (size_t)-1;
        }
        if (base[hit] == '\\') {
            if (hit + 1 >= len) return (size_t)-1;
            p = hit + 2;
        } else return hit + 1;
    }
}

static size_t skip_json_value(const uint8_t* base, size_t p, size_t len) {
    p = skip_ws(base, p, len);
    if (p >= len) return (size_t)-1;
    uint8_t c = base[p];
    if (c == '"') return skip_json_string(base, p + 1, len);
    if (c == '{') {
        p = skip_ws(base, p + 1, len);
        if (p < len && base[p] == '}') return p + 1;
        for (;;) {
            p = skip_ws(base, p, len);
            if (p >= len || base[p] != '"') return (size_t)-1;
            p = skip_json_string(base, p + 1, len);
            if (p == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, p, len);
            if (p >= len || base[p] != ':') return (size_t)-1;
            p = skip_json_value(base, p + 1, len);
            if (p == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, p, len);
            if (p >= len) return (size_t)-1;
            if (base[p] == ',') p++;
            else if (base[p] == '}') return p + 1;
            else return (size_t)-1;
        }
    }
    if (c == '[') {
        p = skip_ws(base, p + 1, len);
        if (p < len && base[p] == ']') return p + 1;
        for (;;) {
            p = skip_json_value(base, p, len);
            if (p == (size_t)-1) return (size_t)-1;
            p = skip_ws(base, p, len);
            if (p >= len) return (size_t)-1;
            if (base[p] == ',') p++;
            else if (base[p] == ']') return p + 1;
            else return (size_t)-1;
        }
    }
    while (p < len) {
        uint8_t b = base[p];
        if (b == ',' || b == '}' || b == ']' || b == ' ' || b == '\n' || b == '\r' || b == '\t') break;
        p++;
    }
    return p;
}

static int flush_out_fd(long out_fd, WS* ws) {
    size_t sent = 0;
    while (sent < ws->out_len) {
        long w = sys3(SYS_write, out_fd, (long)(ws->out + sent), (long)(ws->out_len - sent));
        if (w <= 0) return -1;
        sent += (size_t)w;
    }
    ws->out_len = 0;
    return 0;
}

#ifdef STANDALONE_TEST
#include <time.h>
static inline double _t_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3 + ts.tv_nsec*1e-6; }
#endif

static int run_cases_file_impl(const char* in_path, const char* out_path, int force_sse2, int exit_on_ok) {
#ifdef STANDALONE_TEST
    double pt_old = 0, pt_new = 0, pt_myers = 0, pt_fmt = 0, pt_flush = 0;
#endif
    long in_fd = sys4(SYS_openat, AT_FDCWD, (long)in_path, O_RDONLY, 0);
    if (in_fd < 0) return -1;
    long sz_l = sys3(SYS_lseek, in_fd, 0, SEEK_END);
    if (sz_l <= 0) { sys1(SYS_close, in_fd); return -1; }
    size_t in_len = (size_t)sz_l;

    // Zero-copy read-only shared file mmap for 100% of files (all SIMD loops bounds-checked against in_len).
    uint8_t* mmap_base = NULL;
    size_t mmap_len = 0;
    uint8_t* base = NULL;
    {
        long mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_SHARED | MAP_POPULATE, in_fd, 0);
        if (mp < 0 && mp > -4096) mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, in_fd, 0);
        if (mp < 0 && mp > -4096) mp = sys6(SYS_mmap, 0, (long)in_len, PROT_READ, MAP_PRIVATE, in_fd, 0);
        if (!(mp < 0 && mp > -4096)) {
            mmap_base = (uint8_t*)mp;
            mmap_len = in_len;
            base = mmap_base;
        }
    }

    size_t est_lines = in_len / 240;
    if (est_lines < 1024) est_lines = 1024;
    if (est_lines > 42000) est_lines = 42000;
    size_t ht_est = in_len / 320;
    if (ht_est < 1024) ht_est = 1024;
    if (ht_est > 32768) ht_est = 32768;
    size_t ht_init_cap = 1024;
    while (ht_init_cap < ht_est * 2) ht_init_cap <<= 1;
    if (ht_init_cap > 65536) ht_init_cap = 65536;

    // Compact ~1.00 MB arena: permanent (old_off + new_off + chg + v) + aliased scratch (ht + ent_meta + a/b_ids + last16 + trace)
    size_t arena_sz = est_lines * 16 + ht_init_cap * 4 + ht_est * 8 + (64u << 10);
    if (arena_sz < (512u << 10)) arena_sz = (512u << 10);

    WS ws;
    mem_zero_bytes(&ws, sizeof(ws));
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
        mem_zero_bytes(base + in_len, 64);
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

    // 1. Allocate PERMANENT arrays first (live during formatting!)
    if (ensure_u32(&ws.arena, &ws.old_off, &ws.old_off_cap, 0, est_lines + 48) < 0 ||
        ensure_u32(&ws.arena, &ws.new_off, &ws.new_off_cap, 0, est_lines + 48) < 0 ||
        ensure_chg(&ws, 4096) < 0 ||
        ensure_v(&ws, 1024) < 0) {
        goto fail;
    }

    // 2. Allocate SCRATCH arrays contiguously in block 0 (reused as ws.out during formatting!)
    size_t scratch_off = (ws.arena.used + 31) & ~(size_t)31;
    ws.arena.used = scratch_off;
    ws.scratch_base = (uint8_t*)ws.arena.blocks[0] + scratch_off;
    ws.ht = (uint32_t*)arena_alloc(&ws.arena, ht_init_cap * 4, 32);
    ws.ht_cap = ht_init_cap;
    ws.last_a16 = (uint16_t*)arena_alloc(&ws.arena, ht_est * 2, 32);
    ws.last_b16 = (uint16_t*)arena_alloc(&ws.arena, ht_est * 2, 32);
    ws.last16_cap = ht_est;
    if (!ws.ht || !ws.last_a16 || !ws.last_b16 ||
        ensure_entries(&ws, ht_est) < 0 ||
        ensure_u16(&ws.arena, &ws.a_ids, &ws.a_ids_cap, 0, est_lines + 48) < 0 ||
        ensure_u16(&ws.arena, &ws.b_ids, &ws.b_ids_cap, 0, est_lines + 48) < 0 ||
        ensure_trace(&ws, 0, 4096, 0, 1024) < 0) {
        goto fail;
    }
    // Only alias scratch inside block 0 (guaranteed contiguous!)
    ws.scratch_cap = (ws.arena.n_blocks == 1) ? (ws.arena.used - scratch_off) : 0;

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
            size_t keq = skip_json_string(base, ks, in_len);
            if (keq == (size_t)-1) goto fail;
            size_t klen = (keq - 1) - ks;
            p = skip_ws(base, keq, in_len);
            if (p >= in_len || base[p] != ':') goto fail;
            p = skip_ws(base, p + 1, in_len);

            if (klen == 5 && eq_bytes_sse2(base + ks, (const uint8_t*)"input", 5)) {
                if (solved || p >= in_len || base[p] != '{') goto fail;
                p = skip_ws(base, p + 1, in_len);
                int old_done = 0, new_done = 0;
                old_bare = 0; new_bare = 0;
                size_t deferred_new = (size_t)-1;
                if (intern_reset(&ws, ht_est, use_avx2, case_idx == 0) < 0) goto fail;

                while (p < in_len && base[p] != '}') {
                    if (base[p] != '"') goto fail;
                    size_t iks = p + 1;
                    size_t ikeq = skip_json_string(base, iks, in_len);
                    if (ikeq == (size_t)-1) goto fail;
                    size_t iklen = (ikeq - 1) - iks;
                    p = skip_ws(base, ikeq, in_len);
                    if (p >= in_len || base[p] != ':') goto fail;
                    p = skip_ws(base, p + 1, in_len);

                    if (iklen == 3 && base[iks] == 'o' && base[iks + 1] == 'l' && base[iks + 2] == 'd') {
                        if (old_done || p >= in_len || base[p] != '"') goto fail;
#ifdef STANDALONE_TEST
                        double _a0 = _t_ms();
#endif
                        p = use_avx2 ? scan_json_old_avx2(&ws, base, p + 1, in_len, &old_bare)
                                     : scan_json_old_sse2(&ws, base, p + 1, in_len, &old_bare);
#ifdef STANDALONE_TEST
                        pt_old += _t_ms() - _a0;
#endif
                        if (p == (size_t)-1) goto fail;
                        old_done = 1;
                        if (deferred_new != (size_t)-1) {
#ifdef STANDALONE_TEST
                            double _b0 = _t_ms();
#endif
                            size_t r = use_avx2 ? scan_json_new_avx2(&ws, base, deferred_new, in_len, old_bare, &new_bare)
                                                : scan_json_new_sse2(&ws, base, deferred_new, in_len, old_bare, &new_bare);
#ifdef STANDALONE_TEST
                            pt_new += _t_ms() - _b0;
#endif
                            if (r == (size_t)-1) goto fail;
                            new_done = 1; deferred_new = (size_t)-1;
                        }
                    } else if (iklen == 3 && base[iks] == 'n' && base[iks + 1] == 'e' && base[iks + 2] == 'w') {
                        if (new_done || deferred_new != (size_t)-1 || p >= in_len || base[p] != '"') goto fail;
                        if (old_done) {
#ifdef STANDALONE_TEST
                            double _b0 = _t_ms();
#endif
                            p = use_avx2 ? scan_json_new_avx2(&ws, base, p + 1, in_len, old_bare, &new_bare)
                                         : scan_json_new_sse2(&ws, base, p + 1, in_len, old_bare, &new_bare);
#ifdef STANDALONE_TEST
                            pt_new += _t_ms() - _b0;
#endif
                            if (p == (size_t)-1) goto fail;
                            new_done = 1;
                        } else {
                            deferred_new = p + 1;
                            p = skip_json_string(base, p + 1, in_len);
                            if (p == (size_t)-1) goto fail;
                        }
                    } else {
                        p = skip_json_value(base, p, in_len);
                        if (p == (size_t)-1) goto fail;
                    }
                    p = skip_ws(base, p, in_len);
                    if (p < in_len && base[p] == ',') p = skip_ws(base, p + 1, in_len);
                }
                if (p >= in_len || base[p] != '}' || !old_done || !new_done) goto fail;
                p++;
                solved = 1;
            } else {
                p = skip_json_value(base, p, in_len);
                if (p == (size_t)-1) goto fail;
            }
            p = skip_ws(base, p, in_len);
            if (p < in_len && base[p] == ',') p = skip_ws(base, p + 1, in_len);
        }
        if (!solved || p >= in_len || base[p] != '}') goto fail;
        p = skip_ws(base, p + 1, in_len);

        // Run exact Myers diff while scratch is live
        uint32_t td = 0, ti = 0;
#ifdef STANDALONE_TEST
        double _m0 = _t_ms();
#endif
        int mrc = use_avx2 ? run_myers_avx2(&ws, &td, &ti, case_idx == 0) : run_myers_sse2(&ws, &td, &ti, case_idx == 0);
#ifdef STANDALONE_TEST
        pt_myers += _t_ms() - _m0;
#endif
        if (mrc < 0) goto fail;

        // Scratch (ht + ent_meta + a/b_ids + last16 + trace) is now 100% dead -> alias as ws.out!
        ws.out = ws.scratch_base;
        ws.out_cap = ws.scratch_cap;
        ws.out_len = 0;

        if (ensure_out(&ws, 4) < 0) goto fail;
        ws.out[ws.out_len++] = (case_idx == 0) ? '[' : ',';
        ws.out[ws.out_len++] = '"';
#ifdef STANDALONE_TEST
        double _f0 = _t_ms();
#endif
        int frc = use_avx2 ? format_hunks_json_avx2(&ws, base, in_len, old_bare, new_bare, td, ti)
                           : format_hunks_json_sse2(&ws, base, in_len, old_bare, new_bare, td, ti);
#ifdef STANDALONE_TEST
        pt_fmt += _t_ms() - _f0;
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
            p++;
            continue;
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
            fprintf(stderr, "[v5-prof] old=%.2f new=%.2f myers=%.2f fmt=%.2f flush=%.2f | compute=%.2fms\n",
                pt_old, pt_new, pt_myers, pt_fmt, pt_flush, pt_old + pt_new + pt_myers + pt_fmt);
            fprintf(stderr, "[stats] arena_blocks=%d arena_used=%zuKB arena_cap=%zuKB scratch=%zuKB ht_cap=%zu entries_cap=%zu\n",
                ws.arena.n_blocks, ws.arena.used >> 10, ws.arena.cap >> 10, ws.scratch_cap >> 10, ws.ht_cap, ws.entries_cap);
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
static napi_value js_run_file(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    int32_t rc = -1;
    if (argc >= 2) {
        char in_path[4096], out_path[4096];
        size_t l1 = 0, l2 = 0;
        if (napi_get_value_string_utf8(env, argv[0], in_path, sizeof(in_path), &l1) == 0 &&
            napi_get_value_string_utf8(env, argv[1], out_path, sizeof(out_path), &l2) == 0 &&
            l1 < sizeof(in_path) - 1 && l2 < sizeof(out_path) - 1) {
            int exit_on_ok = (argc >= 3) ? 1 : 0;
            rc = run_cases_file_impl(in_path, out_path, 0, exit_on_ok);
        }
    }
    napi_value res;
    napi_create_int32(env, rc, &res);
    return res;
}

static napi_value init_module(napi_env env, napi_value exports) {
    napi_value fn;
    napi_create_function(env, "runFile", 7, js_run_file, NULL, &fn);
    napi_set_named_property(env, exports, "runFile", fn);
    return exports;
}

static napi_module mod_def = {
    1, 0, __FILE__, init_module, "diff_native", NULL, {0, 0, 0, 0}
};

__attribute__((constructor))
static void register_diff_native(void) {
    napi_module_register(&mod_def);
}
#else
#include <stdio.h>
#include <sys/resource.h>

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
