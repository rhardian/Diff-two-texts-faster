#![no_std]
#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unreachable()
}

// ---------------------------------------------------------------------------
// Compact zero-indirection L2-packed memory map (ZERO cold gaps inside OUT_BUF!)
// Stack (16 KB) + static (~256B) live at 0x00000..0x04400 (--stack-first).
// Permanent arrays (survive across formatting): 0x05000..0x6F000 (424 KB)
// Scratch slab (100% dead during formatting -> aliased as OUT_BUF!):
//   0x70000 .. 0xB0000 (256 KB): HT       (65,536 * u32)  [100% hot]
//   0xB0000 .. 0xC7000 ( 92 KB): A_IDS    (47,104 * u16)  [hot]
//   0xC7000 .. 0xDE000 ( 92 KB): B_IDS    (47,104 * u16)  [hot]
//   0xDE000 .. 0x10B000(184 KB): ENT_META (47,104 * u32)  [hot]
//   0x10B000.. 0x122000( 92 KB): LAST_A   (47,104 * u16)
//   0x122000.. 0x139000( 92 KB): LAST_B   (47,104 * u16)
//   0x139000.. 0x15D000(144 KB): TR_X(64K)+TR_C(16K)+ROW_BASE(16K)+ROW_LO(16K)+SEG_OPS(32K)
//   Total SCRATCH_CAP = 0x15D000 - 0x70000 = 0xED000 = 970,752 B (~948 KB!)
//   Notice: first 510 KB of SCRATCH_BASE overlaps HT + A_IDS + B_IDS + ENT_META ->
//           ~95% already hot before format_hunks_json writes a single byte!
// 0x15E000 (1,400 KB): IN_BASE (streaming input window, grows rightwards as needed!)
// ---------------------------------------------------------------------------

const OLD_OFF_ADDR: usize  = 0x005000;
const NEW_OFF_ADDR: usize  = 0x033000;
const CHG_ADDR: usize      = 0x061000;
const V_ADDR: usize        = 0x071000; // 0x71000..0x73400 (9 KB)

const SCRATCH_BASE: usize  = 0x074000;
const HT_ADDR: usize       = 0x074000; // 256 KB -> 0xB4000
const A_IDS_ADDR: usize    = 0x0B4000; //  92 KB -> 0xCB000
const B_IDS_ADDR: usize    = 0x0CB000; //  92 KB -> 0xE2000
const ENT_META_ADDR: usize = 0x0E2000; // 184 KB -> 0x110000
const LAST_A_ADDR: usize   = 0x110000; //  92 KB -> 0x127000
const LAST_B_ADDR: usize   = 0x127000; //  92 KB -> 0x13E000
const TR_X_ADDR: usize     = 0x13E000; //  64 KB -> 0x14E000
const TR_C_ADDR: usize     = 0x14E000; //  16 KB -> 0x152000
const ROW_BASE_ADDR: usize = 0x152000; //  16 KB -> 0x156000
const ROW_LO_ADDR: usize   = 0x156000; //  16 KB -> 0x15A000
const SEG_OPS_ADDR: usize  = 0x15A000; //  32 KB -> 0x162000
const SCRATCH_CAP: usize   = 0x162000 - SCRATCH_BASE; // 974,848 bytes (~952 KB!)

const IN_BASE: usize       = 0x163000;

const MAX_LINES: usize     = 46500;
const MAX_CHG: usize       = 4096;
const V_HALF: usize        = 1140;
const MAX_TR: usize        = 16384;
const MAX_ROWS: usize      = 4096;
const MAX_SEG_OPS: usize   = 8192;
const HT_SIZE: usize       = 65536;
const HT_MASK: usize       = HT_SIZE - 1;

const NEW_ONLY_BIT: u32    = 0x8000_0000;
const MULTI_BIT: u32       = 0x4000_0000;
const IDX_MASK: u32        = 0x3FFF_FFFF;
const DEAD: i32            = -2;

const RES_ERR: i32         = -1;
const RES_TRUNC: i32       = -2;
const SCAN_ERR: usize      = usize::MAX;
const SCAN_TRUNC: usize    = usize::MAX - 1;

static mut ST_CONSUMED: usize = 0;
static mut ST_DONE: u32 = 0;
static mut ST_CASE_IDX: u32 = 0;
static mut ST_NEXT_ID: usize = 0;
static mut ST_A_LEN: usize = 0;
static mut ST_B_LEN: usize = 0;
static mut ST_CHG_LEN: usize = 0;
static mut ST_OUT_PTR: usize = SCRATCH_BASE;
static mut ST_OUT_LEN: usize = 0;
static mut ST_V_INITED: u32 = 0;
static mut ST_HT_DIRTY: u32 = 0;
static mut ST_LAST_DIRTY: u32 = 0;

static DIGITS2: [u8; 200] = *b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

#[cold]
#[inline(never)]
unsafe fn ensure_mem(end: usize) {
    let cur = memory_size(0) << 16;
    if end > cur {
        let pages = (end - cur + 65535) >> 16;
        if memory_grow(0, pages) == usize::MAX {
            unreachable();
        }
    }
}

#[inline(always)]
unsafe fn read_u64(p: *const u8) -> u64 { (p as *const u64).read_unaligned() }
#[inline(always)]
unsafe fn read_u32(p: *const u8) -> u32 { (p as *const u32).read_unaligned() }

// ---------------------------------------------------------------------------
// Compact fast 64-bit hash + SIMD128 byte equality & copy
// ---------------------------------------------------------------------------

const K1: u64 = 0x9E37_79B9_7F4A_7C15;
const K2: u64 = 0xBF58_476D_1CE4_E5B9;

#[inline(always)]
unsafe fn hash_line(p: *const u8, len: usize) -> u64 {
    if len >= 16 {
        let a = read_u64(p);
        let b = read_u64(p.add(len >> 2));
        let c = read_u64(p.add((len >> 1) - 4));
        let d = read_u64(p.add(len - 8));
        let mut h = ((a ^ (len as u64).wrapping_mul(K2)).wrapping_mul(K1) ^ b)
            .wrapping_mul(K2)
            ^ c.wrapping_mul(K1)
            ^ d;
        if len > 128 {
            let mut q = p.add(16);
            let e = p.add(len - 16);
            while q.add(16) <= e {
                h = (h ^ read_u64(q)).wrapping_mul(K1) ^ read_u64(q.add(8));
                q = q.add(16);
            }
        }
        return h ^ (h >> 32);
    }
    if len >= 4 {
        let a = read_u32(p) as u64;
        let b = read_u32(p.add(len - 4)) as u64;
        let c = if len >= 8 { read_u32(p.add(4)) as u64 } else { 0 };
        let h = ((a | (b << 32)) ^ c.wrapping_mul(K2) ^ (len as u64)).wrapping_mul(K1);
        return h ^ (h >> 32);
    }
    let v = if len > 0 { (*p as u64) | ((*p.add(len - 1) as u64) << 8) } else { 0 };
    let h = (v ^ ((len as u64) << 16)).wrapping_mul(K1);
    h ^ (h >> 32)
}

#[inline(always)]
unsafe fn eq_bytes(mut a: *const u8, mut b: *const u8, mut len: usize) -> bool {
    if len <= 16 {
        if len >= 8 {
            return ((read_u64(a) ^ read_u64(b)) | (read_u64(a.add(len - 8)) ^ read_u64(b.add(len - 8)))) == 0;
        }
        if len >= 4 {
            return ((read_u32(a) ^ read_u32(b)) | (read_u32(a.add(len - 4)) ^ read_u32(b.add(len - 4)))) == 0;
        }
        let mut i = 0;
        while i < len {
            if *a.add(i) != *b.add(i) { return false; }
            i += 1;
        }
        return true;
    }
    while len >= 16 {
        let eq = u8x16_eq(v128_load(a as *const v128), v128_load(b as *const v128));
        if u8x16_bitmask(eq) != 0xFFFF { return false; }
        a = a.add(16);
        b = b.add(16);
        len -= 16;
    }
    if len > 0 {
        let eq = u8x16_eq(
            v128_load(a.add(len).sub(16) as *const v128),
            v128_load(b.add(len).sub(16) as *const v128),
        );
        return u8x16_bitmask(eq) == 0xFFFF;
    }
    true
}

#[inline(always)]
unsafe fn copy_line_fast(mut dst: *mut u8, mut src: *const u8, mut len: usize) -> *mut u8 {
    let end = dst.add(len);
    v128_store(dst as *mut v128, v128_load(src as *const v128));
    if len <= 16 { return end; }
    v128_store(dst.add(16) as *mut v128, v128_load(src.add(16) as *const v128));
    if len <= 32 { return end; }
    dst = dst.add(32);
    src = src.add(32);
    len -= 32;
    while len >= 16 {
        v128_store(dst as *mut v128, v128_load(src as *const v128));
        dst = dst.add(16);
        src = src.add(16);
        len -= 16;
    }
    if len > 0 {
        v128_store(end.sub(16) as *mut v128, v128_load(src.add(len).sub(16) as *const v128));
    }
    end
}

#[inline(never)]
unsafe fn copy_short_exact(mut dst: *mut u8, mut src: *const u8, mut len: usize) -> *mut u8 {
    let end = dst.add(len);
    if len >= 16 {
        while len >= 16 {
            v128_store(dst as *mut v128, v128_load(src as *const v128));
            dst = dst.add(16);
            src = src.add(16);
            len -= 16;
        }
        if len > 0 {
            v128_store(end.sub(16) as *mut v128, v128_load(src.add(len).sub(16) as *const v128));
        }
        return end;
    }
    core::ptr::copy_nonoverlapping(src, dst, len);
    end
}

// ---------------------------------------------------------------------------
// Single unified compact line committers for `old` and `new`
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn commit_old_line(base: *const u8, ls: usize, le: usize, n: usize) -> bool {
    if n >= MAX_LINES { return false; }
    let ooff = OLD_OFF_ADDR as *mut u32;
    let a_ids = A_IDS_ADDR as *mut u16;
    *ooff.add(n + 1) = le as u32;
    let len = (le - ls) as u32;
    let h = hash_line(base.add(ls), len as usize);
    let tag16 = ((h >> 32) as u32) & 0xFFFF_0000;
    let mut slot = (h as usize) & HT_MASK;
    let ht = HT_ADDR as *mut u32;
    let em = ENT_META_ADDR as *mut u32;
    loop {
        let ent = *ht.add(slot);
        if ent == 0 {
            let id = ST_NEXT_ID;
            if id >= MAX_LINES { return false; }
            ST_NEXT_ID = id + 1;
            *ht.add(slot) = tag16 | ((id as u32) + 1);
            *em.add(id) = n as u32;
            *a_ids.add(n) = id as u16;
            return true;
        }
        if (ent & 0xFFFF_0000) == tag16 {
            let id = ((ent & 0xFFFF) - 1) as usize;
            let m = *em.add(id);
            let idx = (m & IDX_MASK) as usize;
            let eoff = *ooff.add(idx);
            let elen = *ooff.add(idx + 1) - eoff;
            if elen == len && eq_bytes(base.add(eoff as usize), base.add(ls), len as usize) {
                if (m & MULTI_BIT) == 0 { *em.add(id) = m | MULTI_BIT; }
                *a_ids.add(n) = id as u16;
                return true;
            }
        }
        slot = (slot + 1) & HT_MASK;
    }
}

#[inline(never)]
unsafe fn commit_new_line(base: *const u8, ls: usize, le: usize, m_lines: usize) -> i32 {
    if m_lines >= MAX_LINES { return -1; }
    let noff = NEW_OFF_ADDR as *mut u32;
    let b_ids = B_IDS_ADDR as *mut u16;
    *noff.add(m_lines + 1) = le as u32;
    let len = (le - ls) as u32;
    let h = hash_line(base.add(ls), len as usize);
    let tag16 = ((h >> 32) as u32) & 0xFFFF_0000;
    let mut slot = (h as usize) & HT_MASK;
    let ht = HT_ADDR as *mut u32;
    let em = ENT_META_ADDR as *mut u32;
    let ooff = OLD_OFF_ADDR as *const u32;
    loop {
        let ent = *ht.add(slot);
        if ent == 0 {
            let id = ST_NEXT_ID;
            if id >= MAX_LINES { return -1; }
            ST_NEXT_ID = id + 1;
            *ht.add(slot) = tag16 | ((id as u32) + 1);
            *em.add(id) = NEW_ONLY_BIT | (m_lines as u32);
            *b_ids.add(m_lines) = id as u16;
            return id as i32;
        }
        if (ent & 0xFFFF_0000) == tag16 {
            let id = ((ent & 0xFFFF) - 1) as usize;
            let m = *em.add(id);
            let idx = (m & IDX_MASK) as usize;
            let (eoff, eend) = if (m & NEW_ONLY_BIT) != 0 {
                (*noff.add(idx), *noff.add(idx + 1))
            } else {
                (*ooff.add(idx), *ooff.add(idx + 1))
            };
            if (eend - eoff) == len && eq_bytes(base.add(eoff as usize), base.add(ls), len as usize) {
                *b_ids.add(m_lines) = id as u16;
                return id as i32;
            }
        }
        slot = (slot + 1) & HT_MASK;
    }
}

// ---------------------------------------------------------------------------
// Ultra-compact 32B/iter SIMD128 JSON `old` scanner
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn scan_json_old(base: *const u8, start: usize, avail: usize, eof: bool, bare_out: &mut bool) -> usize {
    *bare_out = false;
    *(OLD_OFF_ADDR as *mut u32) = start as u32;
    let v_bs = u8x16_splat(b'\\');
    let v_qt = u8x16_splat(b'"');

    let mut p = start;
    let mut ls = start;
    let mut n: usize = 0;

    'outer: loop {
        if p + 32 <= avail {
            let v0 = v128_load(base.add(p) as *const v128);
            let v1 = v128_load(base.add(p + 16) as *const v128);
            let m0 = u8x16_bitmask(v128_or(u8x16_eq(v0, v_bs), u8x16_eq(v0, v_qt))) as u32;
            let m1 = u8x16_bitmask(v128_or(u8x16_eq(v1, v_bs), u8x16_eq(v1, v_qt))) as u32;
            let mut m = m0 | (m1 << 16);
            while m != 0 {
                let tz = m.trailing_zeros() as usize;
                let hit = p + tz;
                if *base.add(hit) == b'\\' {
                    if hit + 1 >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
                    let c = *base.add(hit + 1);
                    if c == b'n' {
                        let le = hit + 2;
                        if !commit_old_line(base, ls, le, n) { return SCAN_ERR; }
                        n += 1; ls = le;
                        m &= m - 1;
                    } else if c == b'r' {
                        if hit + 3 >= avail {
                            if !eof { return SCAN_TRUNC; }
                            m &= m - 1;
                        } else if *base.add(hit + 2) == b'\\' && *base.add(hit + 3) == b'n' {
                            let le = hit + 4;
                            if !commit_old_line(base, ls, le, n) { return SCAN_ERR; }
                            n += 1; ls = le;
                            m &= m - 1;
                            if tz + 2 < 32 { m &= !(1u32 << (tz + 2)); }
                            else { p = le; continue 'outer; }
                        } else {
                            m &= m - 1;
                        }
                    } else if c == b'\\' || c == b'"' {
                        m &= m - 1;
                        if tz + 1 < 32 { m &= !(1u32 << (tz + 1)); }
                        else { p = hit + 2; continue 'outer; }
                    } else if c == b't' || c == b'b' || c == b'f' {
                        m &= m - 1;
                    } else {
                        return SCAN_ERR;
                    }
                } else {
                    if hit > ls {
                        if !commit_old_line(base, ls, hit, n) { return SCAN_ERR; }
                        n += 1; *bare_out = true;
                    }
                    ST_A_LEN = n;
                    return hit + 1;
                }
            }
            p += 32;
        } else {
            while p < avail {
                let b = *base.add(p);
                if b == b'"' {
                    if p > ls {
                        if !commit_old_line(base, ls, p, n) { return SCAN_ERR; }
                        n += 1; *bare_out = true;
                    }
                    ST_A_LEN = n;
                    return p + 1;
                }
                if b == b'\\' {
                    if p + 1 >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
                    let c = *base.add(p + 1);
                    if c == b'n' {
                        let le = p + 2;
                        if !commit_old_line(base, ls, le, n) { return SCAN_ERR; }
                        n += 1; ls = le; p = le;
                    } else if c == b'r' {
                        if p + 3 >= avail {
                            if !eof { return SCAN_TRUNC; }
                            p += 2;
                        } else if *base.add(p + 2) == b'\\' && *base.add(p + 3) == b'n' {
                            let le = p + 4;
                            if !commit_old_line(base, ls, le, n) { return SCAN_ERR; }
                            n += 1; ls = le; p = le;
                        } else { p += 2; }
                    } else if c == b'\\' || c == b'"' || c == b't' || c == b'b' || c == b'f' {
                        p += 2;
                    } else { return SCAN_ERR; }
                } else { p += 1; }
            }
            return if eof { SCAN_ERR } else { SCAN_TRUNC };
        }
    }
}

// ---------------------------------------------------------------------------
// Ultra-compact B-Follows-A 32B/iter SIMD128 JSON `new` scanner
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn scan_json_new(base: *const u8, start: usize, avail: usize, eof: bool, old_bare: bool, bare_out: &mut bool) -> usize {
    *bare_out = false;
    let ooff = OLD_OFF_ADDR as *const u32;
    let a_ids = A_IDS_ADDR as *const u16;
    let noff = NEW_OFF_ADDR as *mut u32;
    let b_ids = B_IDS_ADDR as *mut u16;
    let em = ENT_META_ADDR as *const u32;
    *noff = start as u32;

    let n_old = ST_A_LEN;
    let n_old_nl = if old_bare && n_old > 0 { n_old - 1 } else { n_old };
    let old_limit = if n_old_nl > 0 { *ooff.add(n_old_nl) as usize } else { 0 };

    let v_bs = u8x16_splat(b'\\');
    let v_qt = u8x16_splat(b'"');

    let mut p = start;
    let mut pred: usize = 0;
    let mut m_lines: usize = 0;

    loop {
        if pred < n_old_nl && p < avail {
            let old_start = *ooff.add(pred) as usize;
            let avail_old = old_limit - old_start;
            let avail_new = avail - p;
            let max_cmp = if avail_old < avail_new { avail_old } else { avail_new };
            let ab = base.add(old_start);
            let bb = base.add(p);
            let mut matched: usize = 0;
            while matched + 16 <= max_cmp {
                let eqm = u8x16_bitmask(u8x16_eq(
                    v128_load(ab.add(matched) as *const v128),
                    v128_load(bb.add(matched) as *const v128),
                )) as u32;
                if eqm == 0xFFFF { matched += 16; }
                else { matched += ((!eqm) & 0xFFFF).trailing_zeros() as usize; break; }
            }
            if matched + 16 > max_cmp {
                while matched < max_cmp && *ab.add(matched) == *bb.add(matched) { matched += 1; }
            }
            let old_reached = old_start + matched;
            if old_reached >= (*ooff.add(pred + 1) as usize) {
                let delta = (p as isize) - (old_start as isize);
                let mut k = pred + 1;
                while k < n_old_nl && (*ooff.add(k + 1) as usize) <= old_reached { k += 1; }
                let count = k - pred;
                if m_lines + count >= MAX_LINES { return SCAN_ERR; }
                core::ptr::copy_nonoverlapping(a_ids.add(pred), b_ids.add(m_lines), count);
                let src_off = ooff.add(pred + 1);
                let dst_off = noff.add(m_lines + 1);
                let vd = i32x4_splat(delta as i32);
                let mut i = 0;
                while i + 4 <= count {
                    v128_store(dst_off.add(i) as *mut v128, i32x4_add(v128_load(src_off.add(i) as *const v128), vd));
                    i += 4;
                }
                while i < count {
                    *dst_off.add(i) = ((*src_off.add(i) as isize) + delta) as u32;
                    i += 1;
                }
                m_lines += count;
                p = ((*ooff.add(k) as isize) + delta) as usize;
                pred = k;
            }
        }

        let line_start = p;
        let mut is_eof_str = false;
        let line_end: usize;
        loop {
            let hit: usize;
            if p + 16 <= avail {
                let v = v128_load(base.add(p) as *const v128);
                let mask = u8x16_bitmask(v128_or(u8x16_eq(v, v_bs), u8x16_eq(v, v_qt))) as u32;
                if mask == 0 { p += 16; continue; }
                hit = p + (mask.trailing_zeros() as usize);
            } else {
                let mut q = p;
                while q < avail && *base.add(q) != b'\\' && *base.add(q) != b'"' { q += 1; }
                if q >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
                hit = q;
            }
            if *base.add(hit) == b'\\' {
                if hit + 1 >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
                let nxt = *base.add(hit + 1);
                if nxt == b'n' { line_end = hit + 2; break; }
                else if nxt == b'r' {
                    if hit + 3 >= avail {
                        if !eof { return SCAN_TRUNC; }
                        p = hit + 2; continue;
                    }
                    if *base.add(hit + 2) == b'\\' && *base.add(hit + 3) == b'n' { line_end = hit + 4; break; }
                    p = hit + 2; continue;
                } else if nxt == b'"' || nxt == b'\\' || nxt == b't' || nxt == b'b' || nxt == b'f' {
                    p = hit + 2; continue;
                } else { return SCAN_ERR; }
            } else {
                line_end = hit; is_eof_str = true; break;
            }
        }

        if is_eof_str {
            if line_end > line_start {
                if commit_new_line(base, line_start, line_end, m_lines) < 0 { return SCAN_ERR; }
                m_lines += 1;
                *bare_out = true;
            }
            ST_B_LEN = m_lines;
            return line_end + 1;
        }

        let id = commit_new_line(base, line_start, line_end, m_lines);
        if id < 0 { return SCAN_ERR; }
        m_lines += 1;
        p = line_end;

        let op = *em.add(id as usize);
        if (op & (NEW_ONLY_BIT | MULTI_BIT)) == 0 {
            pred = (op as usize) + 1;
        } else if pred < n_old_nl {
            if *a_ids.add(pred) == (id as u16) { pred += 1; }
            else if pred + 1 < n_old_nl && *a_ids.add(pred + 1) == (id as u16) { pred += 2; }
        }
    }
}

// ---------------------------------------------------------------------------
// Exact 16-bit-ID Dominance-Pruned Myers with 8-wide SIMD128 i16x8 Snake
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone)]
struct Chg { o: u32, p: u32, dels: u32, ins: u32 }

#[inline(never)]
unsafe fn flush_segment(
    rel_d: usize, end_x: i32, end_k: i32, seg_x: i32,
    cur_o: &mut u32, cur_p: &mut u32, total_del: &mut u32, total_ins: &mut u32,
) -> bool {
    let rbase = ROW_BASE_ADDR as *const u32;
    let rlo = ROW_LO_ADDR as *const i32;
    let trx = TR_X_ADDR as *const i32;
    let trc = TR_C_ADDR as *const u8;
    let chg = CHG_ADDR as *mut Chg;

    if rel_d == 1 {
        let idx = (*rbase.add(1) as usize) + (((end_k - *rlo.add(1)) >> 1) as usize);
        let is_del = *trc.add(idx) != 0;
        let start_x = if is_del { seg_x + 1 } else { seg_x };
        let co = *cur_o; let cp = *cur_p;
        let n = ST_CHG_LEN;
        let merge = n > 0 && { let prev = *chg.add(n - 1); prev.o + prev.dels == co && prev.p + prev.ins == cp };
        if !merge {
            if n >= MAX_CHG { return false; }
            *chg.add(n) = Chg { o: co, p: cp, dels: 0, ins: 0 };
            ST_CHG_LEN = n + 1;
        }
        let lc = &mut *chg.add(ST_CHG_LEN - 1);
        let snake = (end_x - start_x) as u32;
        if is_del { lc.dels += 1; *cur_o = co + 1 + snake; *cur_p = cp + snake; *total_del += 1; }
        else      { lc.ins  += 1; *cur_o = co + snake; *cur_p = cp + 1 + snake; *total_ins += 1; }
        return true;
    }

    if rel_d * 2 + 4 > MAX_SEG_OPS { return false; }
    let sops = SEG_OPS_ADDR as *mut u32;
    let mut nops: usize = 0;
    let mut cx = end_x;
    let mut ck = end_k;

    let mut rd = rel_d;
    while rd >= 1 {
        let idx = (*rbase.add(rd) as usize) + (((ck - *rlo.add(rd)) >> 1) as usize);
        let is_del = *trc.add(idx) != 0;
        let prev_k = if is_del { ck - 1 } else { ck + 1 };
        let prev_x = if rd == 1 {
            seg_x
        } else {
            let pidx = (*rbase.add(rd - 1) as usize) + (((prev_k - *rlo.add(rd - 1)) >> 1) as usize);
            *trx.add(pidx)
        };
        let start_x = if is_del { prev_x + 1 } else { prev_x };
        let snake = (cx - start_x) as u32;
        if snake > 0 { *sops.add(nops) = snake; nops += 1; }
        *sops.add(nops) = (if is_del { 1u32 << 30 } else { 2u32 << 30 }) | 1;
        nops += 1;
        cx = prev_x; ck = prev_k; rd -= 1;
    }

    let mut co = *cur_o; let mut cp = *cur_p; let mut td = *total_del; let mut ti = *total_ins;
    while nops > 0 {
        nops -= 1;
        let op = *sops.add(nops);
        let kind = op >> 30;
        let count = op & 0x3FFF_FFFF;
        if kind == 0 { co += count; cp += count; continue; }
        let n = ST_CHG_LEN;
        let merge = n > 0 && { let prev = *chg.add(n - 1); prev.o + prev.dels == co && prev.p + prev.ins == cp };
        if !merge {
            if n >= MAX_CHG { return false; }
            *chg.add(n) = Chg { o: co, p: cp, dels: 0, ins: 0 };
            ST_CHG_LEN = n + 1;
        }
        let lc = &mut *chg.add(ST_CHG_LEN - 1);
        if kind == 1 { lc.dels += count; co += count; td += count; }
        else         { lc.ins  += count; cp += count; ti += count; }
    }
    *cur_o = co; *cur_p = cp; *total_del = td; *total_ins = ti;
    true
}

#[inline(never)]
unsafe fn run_myers(out_del: &mut u32, out_ins: &mut u32) -> bool {
    let n0 = ST_A_LEN;
    let m0 = ST_B_LEN;
    let a_all = A_IDS_ADDR as *mut u16;
    let b_all = B_IDS_ADDR as *mut u16;

    let min_nm = if n0 < m0 { n0 } else { m0 };
    let mut prefix: usize = 0;
    while prefix + 8 <= min_nm {
        let eq = i16x8_eq(v128_load(a_all.add(prefix) as *const v128), v128_load(b_all.add(prefix) as *const v128));
        if u8x16_bitmask(eq) == 0xFFFF { prefix += 8; } else { break; }
    }
    while prefix < min_nm && *a_all.add(prefix) == *b_all.add(prefix) { prefix += 1; }
    let rem_a = n0 - prefix;
    let rem_b = m0 - prefix;
    let min_rem = if rem_a < rem_b { rem_a } else { rem_b };
    let mut suffix: usize = 0;
    while suffix < min_rem && *a_all.add(n0 - 1 - suffix) == *b_all.add(m0 - 1 - suffix) { suffix += 1; }

    let n = n0 - prefix - suffix;
    let m = m0 - prefix - suffix;
    ST_CHG_LEN = 0;

    if n == 0 && m == 0 { *out_del = 0; *out_ins = 0; return true; }
    let chg = CHG_ADDR as *mut Chg;
    if n == 0 {
        *chg = Chg { o: prefix as u32, p: prefix as u32, dels: 0, ins: m as u32 };
        ST_CHG_LEN = 1; *out_del = 0; *out_ins = m as u32; return true;
    }
    if m == 0 {
        *chg = Chg { o: prefix as u32, p: prefix as u32, dels: n as u32, ins: 0 };
        ST_CHG_LEN = 1; *out_del = n as u32; *out_ins = 0; return true;
    }

    let a_ptr = a_all.add(prefix);
    let b_ptr = b_all.add(prefix);
    *a_ptr.add(n) = 0xFFFE;
    *b_ptr.add(m) = 0xFFFF;
    let n_i = n as i32;
    let m_i = m as i32;

    let num_ids = ST_NEXT_ID;
    let last_a = LAST_A_ADDR as *mut u16;
    let last_b = LAST_B_ADDR as *mut u16;
    if ST_LAST_DIRTY != 0 {
        core::ptr::write_bytes(last_a, 0, num_ids);
        core::ptr::write_bytes(last_b, 0, num_ids);
    }
    ST_LAST_DIRTY = 1;
    let mut i = 0;
    while i < n { *last_a.add(*a_ptr.add(i) as usize) = (i + 1) as u16; i += 1; }
    let mut j = 0;
    while j < m { *last_b.add(*b_ptr.add(j) as usize) = (j + 1) as u16; j += 1; }

    let v_buf = V_ADDR as *mut i32;
    if ST_V_INITED == 0 {
        let mut vi = 0;
        let vlen = V_HALF * 2 + 4;
        while vi < vlen { *v_buf.add(vi) = DEAD; vi += 1; }
        ST_V_INITED = 1;
    }

    let rbase = ROW_BASE_ADDR as *mut u32;
    let rlo = ROW_LO_ADDR as *mut i32;
    let trx_base = TR_X_ADDR as *mut i32;
    let trc_base = TR_C_ADDR as *mut u8;

    let mut cur_o = prefix as u32;
    let mut cur_p = prefix as u32;
    let mut total_del: u32 = 0;
    let mut total_ins: u32 = 0;
    let mut k_base: i32 = 0;
    let mut k_lo: i32 = 0;
    let mut k_hi: i32 = 0;
    let mut seg_x: i32 = 0;
    let mut rel_d: usize = 0;
    let mut tr_ptr: usize = 0;
    *v_buf.add(V_HALF) = 0;

    let max_edit = n + m;
    let mut d = 1usize;
    while d <= max_edit {
        rel_d += 1;
        let next_lo = k_lo - 1;
        let next_hi = k_hi + 1;
        let nl_d = k_base - (next_lo - 1);
        let nr_d = (next_hi + 1) - k_base;
        let need_half = (if nl_d > nr_d { nl_d } else { nr_d }) as usize + 2;
        let row_len = (((next_hi - next_lo) >> 1) + 1) as usize;
        if need_half > V_HALF || tr_ptr + row_len > MAX_TR || rel_d + 1 > MAX_ROWS {
            let mut vi = 0;
            while vi < V_HALF * 2 + 4 { *v_buf.add(vi) = DEAD; vi += 1; }
            return false;
        }

        *rbase.add(rel_d) = tr_ptr as u32;
        *rlo.add(rel_d) = next_lo;

        let v_zero = v_buf.offset((V_HALF as isize) - (k_base as isize));
        let mut tr_xp = trx_base.add(tr_ptr);
        let mut tr_cp = trc_base.add(tr_ptr);

        let mut new_lo = next_hi + 1;
        let mut new_hi = next_lo - 1;
        let mut best_sum: i32 = -1;
        let mut best_k: i32 = 0;
        let mut best_x: i32 = 0;
        let mut best_y: i32 = 0;
        let mut best_snk: i32 = 0;
        let mut done = false;

        let mut k = next_lo;
        let mut vp = v_zero.offset(next_lo as isize);
        while k <= next_hi {
            let rem_x = *vp.offset(-1);
            let add_x = *vp.offset(1);
            let can_rem = (rem_x != DEAD) & (rem_x < n_i);
            let can_add = (add_x != DEAD) & ((add_x - k) <= m_i);
            if !(can_rem | can_add) {
                *vp = DEAD; *tr_xp = DEAD;
                k += 2; vp = vp.offset(2); tr_xp = tr_xp.add(1); tr_cp = tr_cp.add(1);
                continue;
            }
            let by_del = if can_rem { rem_x + 1 } else { -2 };
            let by_ins = if can_add { add_x } else { -2 };
            let (mut x, is_del) = if by_del >= by_ins { (by_del, 1u8) } else { (by_ins, 0u8) };
            let sx = x;
            let mut y = x - k;
            if *a_ptr.add(x as usize) == *b_ptr.add(y as usize) {
                x += 1; y += 1;
                loop {
                    let eqm = u8x16_bitmask(i16x8_eq(
                        v128_load(a_ptr.add(x as usize) as *const v128),
                        v128_load(b_ptr.add(y as usize) as *const v128),
                    )) as u32;
                    if eqm == 0xFFFF { x += 8; y += 8; }
                    else {
                        let adv = (((!eqm) & 0xFFFF).trailing_zeros() >> 1) as i32;
                        x += adv; y += adv; break;
                    }
                }
            }
            *vp = x; *tr_xp = x; *tr_cp = is_del;
            if k < new_lo { new_lo = k; }
            if k > new_hi { new_hi = k; }
            let sum = x + y;
            if sum > best_sum { best_sum = sum; best_k = k; best_x = x; best_y = y; best_snk = x - sx; }
            if x >= n_i && y >= m_i { done = true; break; }
            k += 2; vp = vp.offset(2); tr_xp = tr_xp.add(1); tr_cp = tr_cp.add(1);
        }
        tr_ptr += row_len;

        let mut ck = k_lo;
        while ck <= k_hi { *v_zero.offset(ck as isize) = DEAD; ck += 2; }

        if done {
            let mut ck2 = new_lo;
            while ck2 <= new_hi { *v_zero.offset(ck2 as isize) = DEAD; ck2 += 2; }
            if !flush_segment(rel_d, n_i, n_i - m_i, seg_x, &mut cur_o, &mut cur_p, &mut total_del, &mut total_ins) {
                return false;
            }
            break;
        }

        if best_snk > 0 || (rel_d & 7) == 0 {
            let mut p_lo = best_k;
            let mut p_hi = best_k;
            let by1 = (best_y + 1) as u16;
            let mut b_bad: i32 = 0;
            let mut s: i32 = 0;
            let mut kl = best_k - 2;
            while kl >= new_lo {
                let i2 = best_x - s - 2;
                if i2 < 0 {
                    let mut q = new_lo;
                    while q <= kl { if *v_zero.offset(q as isize) != DEAD { p_lo = q; break; } q += 2; }
                    break;
                }
                b_bad += (*last_b.add(*a_ptr.add((i2 + 1) as usize) as usize) >= by1) as i32
                       + (*last_b.add(*a_ptr.add(i2 as usize) as usize) >= by1) as i32;
                s += 2;
                let x = *v_zero.offset(kl as isize);
                if x != DEAD {
                    if x <= best_x && (x - kl) <= best_y && (b_bad << 1) < s { *v_zero.offset(kl as isize) = DEAD; }
                    else { p_lo = kl; }
                }
                kl -= 2;
            }
            let bx1 = (best_x + 1) as u16;
            let mut a_bad: i32 = 0;
            s = 0;
            let mut kr = best_k + 2;
            while kr <= new_hi {
                let j2 = best_y - s - 2;
                if j2 < 0 {
                    let mut q = new_hi;
                    while q >= kr { if *v_zero.offset(q as isize) != DEAD { p_hi = q; break; } q -= 2; }
                    break;
                }
                a_bad += (*last_a.add(*b_ptr.add((j2 + 1) as usize) as usize) >= bx1) as i32
                       + (*last_a.add(*b_ptr.add(j2 as usize) as usize) >= bx1) as i32;
                s += 2;
                let x = *v_zero.offset(kr as isize);
                if x != DEAD {
                    if x <= best_x && (x - kr) <= best_y && (a_bad << 1) < s { *v_zero.offset(kr as isize) = DEAD; }
                    else { p_hi = kr; }
                }
                kr += 2;
            }
            k_lo = p_lo; k_hi = p_hi;
        } else {
            k_lo = new_lo; k_hi = new_hi;
        }

        if k_lo == k_hi {
            let cx = *v_zero.offset(k_lo as isize);
            *v_zero.offset(k_lo as isize) = DEAD;
            if !flush_segment(rel_d, cx, k_lo, seg_x, &mut cur_o, &mut cur_p, &mut total_del, &mut total_ins) {
                return false;
            }
            seg_x = cx; k_base = k_lo;
            *v_buf.add(V_HALF) = cx;
            rel_d = 0; tr_ptr = 0;
        }
        d += 1;
    }

    *out_del = total_del;
    *out_ins = total_ins;
    true
}

// ---------------------------------------------------------------------------
// Ultra-compact Branchless-Overcopy JSON Hunk Formatter (zero code duplication!)
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn write_u32_dec(mut dst: *mut u8, mut x: u32) -> *mut u8 {
    let lut = DIGITS2.as_ptr();
    if x < 10 { *dst = b'0' + (x as u8); return dst.add(1); }
    if x < 100 {
        let i = (x as usize) * 2;
        *dst = *lut.add(i); *dst.add(1) = *lut.add(i + 1);
        return dst.add(2);
    }
    let mut tmp = [0u8; 10];
    let mut p = 10usize;
    while x >= 100 {
        let r = (x % 100) as usize; x /= 100;
        p -= 2;
        tmp[p] = *lut.add(r * 2); tmp[p + 1] = *lut.add(r * 2 + 1);
    }
    if x >= 10 {
        let i = (x as usize) * 2; p -= 2;
        tmp[p] = *lut.add(i); tmp[p + 1] = *lut.add(i + 1);
    } else { p -= 1; tmp[p] = b'0' + (x as u8); }
    let mut i = p;
    while i < 10 { *dst = tmp[i]; dst = dst.add(1); i += 1; }
    dst
}

#[inline(never)]
unsafe fn emit_range(
    mut dst: *mut u8, base: *const u8, offs: *const u32,
    mut from: u32, to: u32, tag: u8, fast_ok: bool, bare_idx: u32,
) -> *mut u8 {
    if fast_ok {
        while from < to {
            let s = *offs.add(from as usize) as usize;
            let e = *offs.add(from as usize + 1) as usize;
            *dst = tag;
            dst = copy_line_fast(dst.add(1), base.add(s), e - s);
            from += 1;
        }
    } else {
        while from < to {
            let s = *offs.add(from as usize) as usize;
            let e = *offs.add(from as usize + 1) as usize;
            *dst = tag;
            dst = copy_short_exact(dst.add(1), base.add(s), e - s);
            if from == bare_idx { dst = copy_short_exact(dst, b"\\n%\\n".as_ptr(), 5); }
            from += 1;
        }
    }
    dst
}

#[inline(never)]
unsafe fn format_hunks_json(
    base: *const u8, avail: usize,
    old_bare: bool, new_bare: bool, total_del: u32, total_ins: u32,
    prefix_ch: u8, close_arr: bool,
) -> bool {
    let n0 = ST_A_LEN as u32;
    let m0 = ST_B_LEN as u32;
    let n_chg = ST_CHG_LEN;
    let chg = CHG_ADDR as *const Chg;

    let mut num_hunks: u32 = 0;
    if n_chg > 0 {
        num_hunks = 1;
        let mut ci = 1usize;
        while ci < n_chg {
            let prev = *chg.add(ci - 1);
            if (*chg.add(ci)).o - (prev.o + prev.dels) > 6 { num_hunks += 1; }
            ci += 1;
        }
    }

    let out_base = SCRATCH_BASE as *mut u8;
    let out_lim = out_base.add(SCRATCH_CAP - 128);
    let mut dst = out_base;
    *dst = prefix_ch; dst = dst.add(1);
    *dst = b'"'; dst = dst.add(1);

    dst = copy_short_exact(dst, b"makefun-diff/1 old=".as_ptr(), 19);
    dst = write_u32_dec(dst, n0);
    dst = copy_short_exact(dst, b" new=".as_ptr(), 5);
    dst = write_u32_dec(dst, m0);
    dst = copy_short_exact(dst, b" hunks=".as_ptr(), 7);
    dst = write_u32_dec(dst, num_hunks);
    dst = copy_short_exact(dst, b" deleted=".as_ptr(), 9);
    dst = write_u32_dec(dst, total_del);
    dst = copy_short_exact(dst, b" inserted=".as_ptr(), 10);
    dst = write_u32_dec(dst, total_ins);
    *dst = b'\\'; *dst.add(1) = b'n'; dst = dst.add(2);

    if n_chg > 0 {
        let ooff = OLD_OFF_ADDR as *const u32;
        let noff = NEW_OFF_ADDR as *const u32;
        let safe_lim = if avail > 64 { avail - 64 } else { 0 };
        let old_bare_idx = if old_bare && n0 > 0 { n0 - 1 } else { u32::MAX };
        let new_bare_idx = if new_bare && m0 > 0 { m0 - 1 } else { u32::MAX };

        let mut h_start = 0usize;
        while h_start < n_chg {
            let mut h_end = h_start + 1;
            while h_end < n_chg {
                let prev = *chg.add(h_end - 1);
                if (*chg.add(h_end)).o - (prev.o + prev.dels) <= 6 { h_end += 1; } else { break; }
            }
            let first = *chg.add(h_start);
            let last = *chg.add(h_end - 1);
            let old_start = if first.o > 3 { first.o - 3 } else { 0 };
            let new_start = first.p - (first.o - old_start);
            let last_o_end = last.o + last.dels;
            let last_p_end = last.p + last.ins;
            let rem = n0 - last_o_end;
            let after = if rem < 3 { rem } else { 3 };
            let old_end = last_o_end + after;
            let new_end = last_p_end + after;

            let old_bytes = (*ooff.add(old_end as usize) - *ooff.add(old_start as usize)) as usize;
            let new_bytes = (*noff.add(new_end as usize) - *noff.add(new_start as usize)) as usize;
            let line_cnt = ((old_end - old_start) + (new_end - new_start)) as usize;
            if dst.add(128 + old_bytes + new_bytes + line_cnt) > out_lim { return false; }

            *dst = b'@'; *dst.add(1) = b' '; dst = dst.add(2);
            dst = write_u32_dec(dst, old_start + 1); *dst = b' '; dst = dst.add(1);
            dst = write_u32_dec(dst, old_end - old_start); *dst = b' '; dst = dst.add(1);
            dst = write_u32_dec(dst, new_start + 1); *dst = b' '; dst = dst.add(1);
            dst = write_u32_dec(dst, new_end - new_start);
            *dst = b'\\'; *dst.add(1) = b'n'; dst = dst.add(2);

            let fast_ok = !(old_bare && old_end == n0)
                && !(new_bare && new_end == m0)
                && (*ooff.add(old_end as usize) as usize) <= safe_lim
                && (*noff.add(new_end as usize) as usize) <= safe_lim;

            let mut at = old_start;
            let mut ci = h_start;
            while ci < h_end {
                let c = *chg.add(ci);
                if at < c.o { dst = emit_range(dst, base, ooff, at, c.o, b'=', fast_ok, old_bare_idx); }
                if c.dels > 0 { dst = emit_range(dst, base, ooff, c.o, c.o + c.dels, b'-', fast_ok, old_bare_idx); }
                if c.ins > 0 { dst = emit_range(dst, base, noff, c.p, c.p + c.ins, b'+', fast_ok, new_bare_idx); }
                at = c.o + c.dels;
                ci += 1;
            }
            if at < old_end { dst = emit_range(dst, base, ooff, at, old_end, b'=', fast_ok, old_bare_idx); }
            h_start = h_end;
        }
    }

    *dst = b'"'; dst = dst.add(1);
    if close_arr { *dst = b']'; dst = dst.add(1); }
    ST_OUT_PTR = SCRATCH_BASE;
    ST_OUT_LEN = (dst as usize) - SCRATCH_BASE;
    true
}

// ---------------------------------------------------------------------------
// Streaming JSON wrapper helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn skip_ws(base: *const u8, mut p: usize, avail: usize) -> usize {
    while p < avail {
        let b = *base.add(p);
        if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' { p += 1; } else { break; }
    }
    p
}

#[inline(never)]
unsafe fn skip_json_string(base: *const u8, mut p: usize, avail: usize, eof: bool) -> usize {
    let v_bs = u8x16_splat(b'\\');
    let v_qt = u8x16_splat(b'"');
    loop {
        let hit: usize;
        if p + 16 <= avail {
            let v = v128_load(base.add(p) as *const v128);
            let mask = u8x16_bitmask(v128_or(u8x16_eq(v, v_bs), u8x16_eq(v, v_qt))) as u32;
            if mask == 0 { p += 16; continue; }
            hit = p + (mask.trailing_zeros() as usize);
        } else {
            let mut q = p;
            while q < avail && *base.add(q) != b'\\' && *base.add(q) != b'"' { q += 1; }
            if q >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            hit = q;
        }
        if *base.add(hit) == b'\\' {
            if hit + 1 >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            p = hit + 2;
        } else {
            return hit + 1;
        }
    }
}

#[inline(never)]
unsafe fn skip_json_value(base: *const u8, mut p: usize, avail: usize, eof: bool) -> usize {
    p = skip_ws(base, p, avail);
    if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
    let c = *base.add(p);
    if c == b'"' { return skip_json_string(base, p + 1, avail, eof); }
    if c == b'{' {
        p = skip_ws(base, p + 1, avail);
        if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
        if *base.add(p) == b'}' { return p + 1; }
        loop {
            p = skip_ws(base, p, avail);
            if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            if *base.add(p) != b'"' { return SCAN_ERR; }
            p = skip_json_string(base, p + 1, avail, eof);
            if p >= SCAN_TRUNC { return p; }
            p = skip_ws(base, p, avail);
            if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            if *base.add(p) != b':' { return SCAN_ERR; }
            p = skip_json_value(base, p + 1, avail, eof);
            if p >= SCAN_TRUNC { return p; }
            p = skip_ws(base, p, avail);
            if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            if *base.add(p) == b',' { p += 1; }
            else if *base.add(p) == b'}' { return p + 1; }
            else { return SCAN_ERR; }
        }
    }
    if c == b'[' {
        p = skip_ws(base, p + 1, avail);
        if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
        if *base.add(p) == b']' { return p + 1; }
        loop {
            p = skip_json_value(base, p, avail, eof);
            if p >= SCAN_TRUNC { return p; }
            p = skip_ws(base, p, avail);
            if p >= avail { return if eof { SCAN_ERR } else { SCAN_TRUNC }; }
            if *base.add(p) == b',' { p += 1; }
            else if *base.add(p) == b']' { return p + 1; }
            else { return SCAN_ERR; }
        }
    }
    while p < avail {
        let b = *base.add(p);
        if b == b',' || b == b'}' || b == b']' || b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
            return p;
        }
        p += 1;
    }
    if eof { p } else { SCAN_TRUNC }
}

// ---------------------------------------------------------------------------
// Minimal 4-export WASM API: init_stream, grow_input, get_info, step_case
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn init_stream(in_cap: usize) -> usize {
    ensure_mem(IN_BASE + in_cap + 64);
    ST_CONSUMED = 0;
    ST_DONE = 0;
    ST_CASE_IDX = 0;
    ST_OUT_PTR = SCRATCH_BASE;
    ST_OUT_LEN = 0;
    IN_BASE
}

#[no_mangle]
pub unsafe extern "C" fn grow_input(new_cap: usize) -> usize {
    ensure_mem(IN_BASE + new_cap + 64);
    IN_BASE
}

#[no_mangle]
pub unsafe extern "C" fn get_info(which: u32) -> usize {
    match which {
        0 => ST_OUT_PTR,
        1 => ST_CONSUMED,
        2 => ST_DONE as usize,
        _ => IN_BASE,
    }
}

#[no_mangle]
pub unsafe extern "C" fn step_case(avail: usize, eof_flag: u32) -> i32 {
    let eof = eof_flag != 0;
    let base = IN_BASE as *mut u8;
    let mut p: usize = 0;

    if ST_CASE_IDX == 0 {
        p = skip_ws(base, p, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
        if *base.add(p) != b'[' { return RES_ERR; }
        p = skip_ws(base, p + 1, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
        if *base.add(p) == b']' {
            p = skip_ws(base, p + 1, avail);
            if p < avail { return RES_ERR; }
            if !eof { return RES_TRUNC; }
            let out = SCRATCH_BASE as *mut u8;
            *out = b'['; *out.add(1) = b']';
            ST_OUT_PTR = SCRATCH_BASE;
            ST_OUT_LEN = 2;
            ST_CONSUMED = avail;
            ST_DONE = 1;
            ST_CASE_IDX = 1;
            return 2;
        }
    } else {
        p = skip_ws(base, p, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
    }

    if *base.add(p) != b'{' { return RES_ERR; }
    p = skip_ws(base, p + 1, avail);
    if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }

    let mut solved = false;
    let mut old_bare = false;
    let mut new_bare = false;

    while p < avail && *base.add(p) != b'}' {
        if *base.add(p) != b'"' { return RES_ERR; }
        let ks = p + 1;
        let keq = skip_json_string(base, ks, avail, eof);
        if keq == SCAN_ERR { return RES_ERR; }
        if keq == SCAN_TRUNC { return RES_TRUNC; }
        let klen = (keq - 1) - ks;
        p = skip_ws(base, keq, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
        if *base.add(p) != b':' { return RES_ERR; }
        p = skip_ws(base, p + 1, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }

        if klen == 5 && eq_bytes(base.add(ks), b"input".as_ptr(), 5) {
            if solved || *base.add(p) != b'{' { return RES_ERR; }
            p = skip_ws(base, p + 1, avail);
            if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }

            if ST_HT_DIRTY != 0 {
                core::ptr::write_bytes(HT_ADDR as *mut u32, 0, HT_SIZE);
            }
            ST_HT_DIRTY = 1;
            ST_NEXT_ID = 0;

            let mut old_done = false;
            let mut new_done = false;
            let mut deferred_new: usize = SCAN_ERR;

            while p < avail && *base.add(p) != b'}' {
                if *base.add(p) != b'"' { return RES_ERR; }
                let iks = p + 1;
                let ikeq = skip_json_string(base, iks, avail, eof);
                if ikeq == SCAN_ERR { return RES_ERR; }
                if ikeq == SCAN_TRUNC { return RES_TRUNC; }
                let iklen = (ikeq - 1) - iks;
                p = skip_ws(base, ikeq, avail);
                if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
                if *base.add(p) != b':' { return RES_ERR; }
                p = skip_ws(base, p + 1, avail);
                if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }

                if iklen == 3 && *base.add(iks) == b'o' && *base.add(iks + 1) == b'l' && *base.add(iks + 2) == b'd' {
                    if old_done || *base.add(p) != b'"' { return RES_ERR; }
                    p = scan_json_old(base, p + 1, avail, eof, &mut old_bare);
                    if p == SCAN_ERR { return RES_ERR; }
                    if p == SCAN_TRUNC { return RES_TRUNC; }
                    old_done = true;
                    if deferred_new != SCAN_ERR {
                        let r = scan_json_new(base, deferred_new, avail, eof, old_bare, &mut new_bare);
                        if r == SCAN_ERR { return RES_ERR; }
                        if r == SCAN_TRUNC { return RES_TRUNC; }
                        new_done = true;
                        deferred_new = SCAN_ERR;
                    }
                } else if iklen == 3 && *base.add(iks) == b'n' && *base.add(iks + 1) == b'e' && *base.add(iks + 2) == b'w' {
                    if new_done || deferred_new != SCAN_ERR || *base.add(p) != b'"' { return RES_ERR; }
                    if old_done {
                        p = scan_json_new(base, p + 1, avail, eof, old_bare, &mut new_bare);
                        if p == SCAN_ERR { return RES_ERR; }
                        if p == SCAN_TRUNC { return RES_TRUNC; }
                        new_done = true;
                    } else {
                        deferred_new = p + 1;
                        p = skip_json_string(base, p + 1, avail, eof);
                        if p == SCAN_ERR { return RES_ERR; }
                        if p == SCAN_TRUNC { return RES_TRUNC; }
                    }
                } else {
                    p = skip_json_value(base, p, avail, eof);
                    if p == SCAN_ERR { return RES_ERR; }
                    if p == SCAN_TRUNC { return RES_TRUNC; }
                }
                p = skip_ws(base, p, avail);
                if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
                if *base.add(p) == b',' {
                    p = skip_ws(base, p + 1, avail);
                    if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
                }
            }
            if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
            if *base.add(p) != b'}' || !old_done || !new_done { return RES_ERR; }
            p += 1;
            solved = true;
        } else {
            p = skip_json_value(base, p, avail, eof);
            if p == SCAN_ERR { return RES_ERR; }
            if p == SCAN_TRUNC { return RES_TRUNC; }
        }
        p = skip_ws(base, p, avail);
        if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
        if *base.add(p) == b',' {
            p = skip_ws(base, p + 1, avail);
            if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
        }
    }
    if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }
    if !solved || *base.add(p) != b'}' { return RES_ERR; }
    p = skip_ws(base, p + 1, avail);
    if p >= avail { return if eof { RES_ERR } else { RES_TRUNC }; }

    let close_arr: bool;
    let next_p: usize;
    if *base.add(p) == b',' {
        close_arr = false;
        next_p = p + 1;
    } else if *base.add(p) == b']' {
        let tail = skip_ws(base, p + 1, avail);
        if tail < avail { return RES_ERR; }
        if !eof { return RES_TRUNC; }
        close_arr = true;
        next_p = avail;
    } else {
        return RES_ERR;
    }

    let mut td = 0u32;
    let mut ti = 0u32;
    if !run_myers(&mut td, &mut ti) { return RES_ERR; }
    let prefix_ch = if ST_CASE_IDX == 0 { b'[' } else { b',' };
    if !format_hunks_json(base, avail, old_bare, new_bare, td, ti, prefix_ch, close_arr) {
        return RES_ERR;
    }
    ST_HT_DIRTY = 1;
    ST_LAST_DIRTY = 1;
    ST_CASE_IDX += 1;
    if close_arr { ST_DONE = 1; }

    let rem = avail - next_p;
    if rem > 0 {
        core::ptr::copy(base.add(next_p), base, rem);
    }
    ST_CONSUMED = next_p;
    ST_OUT_LEN as i32
}
