#![no_std]
#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unreachable()
}

// ---------------------------------------------------------------------------
// Bump / arena allocator over WASM linear memory
// ---------------------------------------------------------------------------

extern "C" {
    static __heap_base: u8;
}

static mut HEAP_PTR: usize = 0;

#[inline(always)]
unsafe fn heap_base() -> usize {
    &__heap_base as *const u8 as usize
}

#[cold]
#[inline(never)]
unsafe fn grow_mem(end: usize) {
    let cur_bytes = memory_size(0) << 16;
    if end > cur_bytes {
        let need_pages = (end - cur_bytes + 65535) >> 16;
        if memory_grow(0, need_pages) == usize::MAX {
            unreachable();
        }
    }
}

#[inline(always)]
unsafe fn ensure_mem(end: usize) {
    if end > (memory_size(0) << 16) {
        grow_mem(end);
    }
}

#[inline(never)]
unsafe fn arena_alloc(size: usize, align: usize) -> usize {
    let p = (HEAP_PTR + align - 1) & !(align - 1);
    let end = p + size;
    ensure_mem(end);
    HEAP_PTR = end;
    p
}

#[no_mangle]
pub unsafe extern "C" fn alloc_input(size: usize) -> usize {
    let base = (heap_base() + 15) & !15;
    let end = base + size + 64;
    ensure_mem(end);
    core::ptr::write_bytes((base + size) as *mut u8, 0, 64);
    HEAP_PTR = (end + 15) & !15;
    let ws = &mut *core::ptr::addr_of_mut!(WS);
    ws.reset_all();
    base
}

// ---------------------------------------------------------------------------
// Fast 64-bit hashing & SIMD byte equality / copy
// ---------------------------------------------------------------------------

const HASH_SEED: u64 = 0x517C_C1B7_2722_0A95;
const K1: u64 = 0x9E37_79B9_7F4A_7C15;
const K2: u64 = 0xBF58_476D_1CE4_E5B9;

#[inline(always)]
unsafe fn mix_tail(mut h: u64, ptr: *const u8, rem: usize) -> u64 {
    if rem >= 8 {
        let v0 = (ptr as *const u64).read_unaligned();
        h = (h ^ v0).wrapping_mul(K1);
        if rem > 8 {
            let v1 = (ptr.add(rem - 8) as *const u64).read_unaligned();
            h = (h ^ v1).wrapping_mul(K2);
        }
    } else if rem >= 4 {
        let v0 = (ptr as *const u32).read_unaligned() as u64;
        let v1 = (ptr.add(rem - 4) as *const u32).read_unaligned() as u64;
        h = (h ^ (v0 | (v1 << 32))).wrapping_mul(K1);
    } else if rem > 0 {
        let b0 = *ptr as u64;
        let b1 = *ptr.add(rem >> 1) as u64;
        let b2 = *ptr.add(rem - 1) as u64;
        h = (h ^ (b0 | (b1 << 8) | (b2 << 16))).wrapping_mul(K1);
    }
    h = (h ^ (rem as u64)).wrapping_mul(K2);
    h ^ (h >> 32)
}

#[inline(never)]
unsafe fn hash_bytes(mut ptr: *const u8, mut len: usize) -> u64 {
    let mut h = HASH_SEED;
    while len >= 16 {
        let v = v128_load(ptr as *const v128);
        h = (h ^ u64x2_extract_lane::<0>(v)).wrapping_mul(K1);
        h = (h ^ u64x2_extract_lane::<1>(v)).wrapping_mul(K2);
        ptr = ptr.add(16);
        len -= 16;
    }
    mix_tail(h, ptr, len)
}

#[inline(always)]
unsafe fn eq_bytes(mut a: *const u8, mut b: *const u8, mut len: usize) -> bool {
    if len <= 16 {
        if len >= 8 {
            return (a as *const u64).read_unaligned() == (b as *const u64).read_unaligned()
                && (a.add(len - 8) as *const u64).read_unaligned()
                    == (b.add(len - 8) as *const u64).read_unaligned();
        }
        if len >= 4 {
            return (a as *const u32).read_unaligned() == (b as *const u32).read_unaligned()
                && (a.add(len - 4) as *const u32).read_unaligned()
                    == (b.add(len - 4) as *const u32).read_unaligned();
        }
        let mut i = 0;
        while i < len {
            if *a.add(i) != *b.add(i) {
                return false;
            }
            i += 1;
        }
        return true;
    }
    while len >= 16 {
        if !i8x16_all_true(i8x16_eq(v128_load(a as *const v128), v128_load(b as *const v128))) {
            return false;
        }
        a = a.add(16);
        b = b.add(16);
        len -= 16;
    }
    if len > 0 {
        return i8x16_all_true(i8x16_eq(
            v128_load(a.add(len).sub(16) as *const v128),
            v128_load(b.add(len).sub(16) as *const v128),
        ));
    }
    true
}

#[inline(always)]
unsafe fn copy_fast(mut dst: *mut u8, mut src: *const u8, mut len: usize) -> *mut u8 {
    let end_dst = dst.add(len);
    if len >= 16 {
        while len >= 16 {
            v128_store(dst as *mut v128, v128_load(src as *const v128));
            dst = dst.add(16);
            src = src.add(16);
            len -= 16;
        }
        if len > 0 {
            v128_store(end_dst.sub(16) as *mut v128, v128_load(src.add(len).sub(16) as *const v128));
        }
        return end_dst;
    }
    if len >= 8 {
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        (end_dst.sub(8) as *mut u64).write_unaligned((src.add(len - 8) as *const u64).read_unaligned());
        return end_dst;
    }
    if len >= 4 {
        (dst as *mut u32).write_unaligned((src as *const u32).read_unaligned());
        (end_dst.sub(4) as *mut u32).write_unaligned((src.add(len - 4) as *const u32).read_unaligned());
        return end_dst;
    }
    let mut i = 0;
    while i < len {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
    end_dst
}

static DIGITS2: [u8; 200] = *b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

#[inline(never)]
unsafe fn write_u32_dec(mut dst: *mut u8, mut x: u32) -> *mut u8 {
    if x < 10 {
        *dst = b'0' + (x as u8);
        return dst.add(1);
    }
    if x < 100 {
        let i = (x as usize) << 1;
        *dst = *DIGITS2.as_ptr().add(i);
        *dst.add(1) = *DIGITS2.as_ptr().add(i + 1);
        return dst.add(2);
    }
    let mut tmp = [0u8; 10];
    let mut p = 10usize;
    while x >= 100 {
        let rem = (x % 100) as usize;
        x /= 100;
        p -= 2;
        let di = rem << 1;
        tmp[p] = *DIGITS2.as_ptr().add(di);
        tmp[p + 1] = *DIGITS2.as_ptr().add(di + 1);
    }
    if x >= 10 {
        let di = (x as usize) << 1;
        p -= 2;
        tmp[p] = *DIGITS2.as_ptr().add(di);
        tmp[p + 1] = *DIGITS2.as_ptr().add(di + 1);
    } else {
        p -= 1;
        tmp[p] = b'0' + (x as u8);
    }
    let n = 10 - p;
    let mut i = 0;
    while i < n {
        *dst = tmp[p + i];
        dst = dst.add(1);
        i += 1;
    }
    dst
}

// ---------------------------------------------------------------------------
// Compact growable vectors (cold growth -> tiny WASM bytecode!)
// ---------------------------------------------------------------------------

struct VecU32 {
    ptr: *mut u32,
    len: usize,
    cap: usize,
}

impl VecU32 {
    const fn empty() -> Self {
        Self { ptr: core::ptr::null_mut(), len: 0, cap: 0 }
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow(&mut self, need: usize) {
        let mut ncap = if self.cap == 0 {
            (need + 63) & !63
        } else {
            let mut c = self.cap << 1;
            while c < need { c <<= 1; }
            c
        };
        if ncap < 256 { ncap = 256; }
        let nptr = arena_alloc(ncap * 4, 16) as *mut u32;
        if self.len > 0 {
            core::ptr::copy_nonoverlapping(self.ptr, nptr, self.len);
        }
        self.ptr = nptr;
        self.cap = ncap;
    }

    #[inline(always)]
    unsafe fn ensure(&mut self, need: usize) {
        if need > self.cap {
            self.grow(need);
        }
    }

    #[inline(always)]
    unsafe fn push(&mut self, v: u32) {
        let l = self.len;
        if l == self.cap {
            self.grow(l + 1);
        }
        *self.ptr.add(l) = v;
        self.len = l + 1;
    }
}

struct VecU8 {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl VecU8 {
    const fn empty() -> Self {
        Self { ptr: core::ptr::null_mut(), len: 0, cap: 0 }
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow(&mut self, need: usize) {
        let mut ncap = if self.cap == 0 {
            (need + 4095) & !4095
        } else {
            let mut c = self.cap << 1;
            while c < need { c <<= 1; }
            c
        };
        if ncap < 4096 { ncap = 4096; }
        let nptr = arena_alloc(ncap, 16) as *mut u8;
        if self.len > 0 {
            core::ptr::copy_nonoverlapping(self.ptr, nptr, self.len);
        }
        self.ptr = nptr;
        self.cap = ncap;
    }

    #[inline(always)]
    unsafe fn ensure(&mut self, need: usize) {
        if need > self.cap {
            self.grow(need);
        }
    }

    #[inline(always)]
    unsafe fn push(&mut self, v: u8) {
        let l = self.len;
        if l == self.cap {
            self.grow(l + 1);
        }
        *self.ptr.add(l) = v;
        self.len = l + 1;
    }
}

// ---------------------------------------------------------------------------
// Single-cache-line Interner Entry (16 bytes per unique line!)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone)]
struct Entry {
    off: u32,
    len: u32,
    tag: u32,
    old_pos: u32,
}

const UNIQUE_BIT: u32 = 0x4000_0000;
const MULTI_POS: u32 = 0x8000_0000;
const UNSEEN_POS: u32 = 0xFFFF_FFFF;

struct Interner {
    ht: *mut u32,
    ht_mask: usize,
    ht_cap: usize,
    entries: *mut Entry,
    entries_cap: usize,
    next_id: usize,
}

impl Interner {
    const fn new() -> Self {
        Self {
            ht: core::ptr::null_mut(),
            ht_mask: 0,
            ht_cap: 0,
            entries: core::ptr::null_mut(),
            entries_cap: 0,
            next_id: 0,
        }
    }

    #[inline(never)]
    unsafe fn reset(&mut self, est_lines: usize) {
        let mut cap = 1024usize;
        let target = est_lines.saturating_mul(2);
        while cap < target {
            cap <<= 1;
        }
        if cap > self.ht_cap {
            self.ht = arena_alloc(cap * 4, 16) as *mut u32;
            self.ht_cap = cap;
        }
        core::ptr::write_bytes(self.ht, 0, cap);
        self.ht_mask = cap - 1;
        if est_lines > self.entries_cap {
            self.grow_entries(est_lines);
        }
        self.next_id = 0;
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow_entries(&mut self, need: usize) {
        let mut ncap = if self.entries_cap == 0 { 4096 } else { self.entries_cap << 1 };
        while ncap < need {
            ncap <<= 1;
        }
        let nptr = arena_alloc(ncap * 16, 16) as *mut Entry;
        if self.next_id > 0 {
            core::ptr::copy_nonoverlapping(self.entries, nptr, self.next_id);
        }
        self.entries = nptr;
        self.entries_cap = ncap;
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow_ht(&mut self, base: *const u8) {
        let new_cap = (self.ht_mask + 1) << 1;
        if new_cap > self.ht_cap {
            self.ht = arena_alloc(new_cap * 4, 16) as *mut u32;
            self.ht_cap = new_cap;
        }
        core::ptr::write_bytes(self.ht, 0, new_cap);
        let mask = new_cap - 1;
        self.ht_mask = mask;
        let n = self.next_id;
        let ep = self.entries;
        let mut id = 0usize;
        while id < n {
            let e = *ep.add(id);
            let h = hash_bytes(base.add(e.off as usize), e.len as usize);
            let mut slot = (h as usize) & mask;
            while *self.ht.add(slot) != 0 {
                slot = (slot + 1) & mask;
            }
            *self.ht.add(slot) = (id + 1) as u32;
            id += 1;
        }
    }

    #[inline(never)]
    unsafe fn intern_old(
        &mut self,
        base: *const u8,
        off: u32,
        len: u32,
        h: u64,
        line_idx: u32,
    ) -> u32 {
        if (self.next_id << 1) >= self.ht_mask {
            self.grow_ht(base);
        }
        let mask = self.ht_mask;
        let tag = (h >> 32) as u32;
        let mut slot = (h as usize) & mask;
        let ht = self.ht;
        let ep = self.entries;
        loop {
            let entry = *ht.add(slot);
            if entry == 0 {
                let id = self.next_id;
                if id == self.entries_cap {
                    self.grow_entries(id + 1);
                }
                self.next_id = id + 1;
                *ht.add(slot) = (id + 1) as u32;
                *self.entries.add(id) = Entry { off, len, tag, old_pos: line_idx | UNIQUE_BIT };
                return id as u32;
            }
            let id = (entry - 1) as usize;
            let e = ep.add(id);
            if (*e).tag == tag
                && (*e).len == len
                && eq_bytes(base.add((*e).off as usize), base.add(off as usize), len as usize)
            {
                (*e).old_pos = MULTI_POS;
                return id as u32;
            }
            slot = (slot + 1) & mask;
        }
    }

    #[inline(never)]
    unsafe fn intern_new(&mut self, base: *const u8, off: u32, len: u32, h: u64) -> u32 {
        if (self.next_id << 1) >= self.ht_mask {
            self.grow_ht(base);
        }
        let mask = self.ht_mask;
        let tag = (h >> 32) as u32;
        let mut slot = (h as usize) & mask;
        let ht = self.ht;
        let ep = self.entries;
        loop {
            let entry = *ht.add(slot);
            if entry == 0 {
                let id = self.next_id;
                if id == self.entries_cap {
                    self.grow_entries(id + 1);
                }
                self.next_id = id + 1;
                *ht.add(slot) = (id + 1) as u32;
                *self.entries.add(id) = Entry { off, len, tag, old_pos: UNSEEN_POS };
                return id as u32;
            }
            let id = (entry - 1) as usize;
            let e = ep.add(id);
            if (*e).tag == tag
                && (*e).len == len
                && eq_bytes(base.add((*e).off as usize), base.add(off as usize), len as usize)
            {
                return id as u32;
            }
            slot = (slot + 1) & mask;
        }
    }
}

// ---------------------------------------------------------------------------
// Workspace (all buffers pre-sized once and reused across all cases!)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone)]
struct Chg {
    o: u32,
    p: u32,
    dels: u32,
    ins: u32,
}

struct Workspace {
    interner: Interner,
    old_off: VecU32,
    new_off: VecU32,
    a_ids: VecU32,
    b_ids: VecU32,
    last_a: VecU32,
    last_b: VecU32,
    v: VecU32,
    v_half: usize,
    tr_x: VecU32,
    tr_choice: VecU8,
    row_base: VecU32,
    row_lo: VecU32,
    seg_ops: VecU32,
    chg_ptr: *mut Chg,
    chg_len: usize,
    chg_cap: usize,
    out: VecU8,
}

const DEAD: i32 = -2;

static mut WS: Workspace = Workspace {
    interner: Interner::new(),
    old_off: VecU32::empty(),
    new_off: VecU32::empty(),
    a_ids: VecU32::empty(),
    b_ids: VecU32::empty(),
    last_a: VecU32::empty(),
    last_b: VecU32::empty(),
    v: VecU32::empty(),
    v_half: 0,
    tr_x: VecU32::empty(),
    tr_choice: VecU8::empty(),
    row_base: VecU32::empty(),
    row_lo: VecU32::empty(),
    seg_ops: VecU32::empty(),
    chg_ptr: core::ptr::null_mut(),
    chg_len: 0,
    chg_cap: 0,
    out: VecU8::empty(),
};

impl Workspace {
    unsafe fn reset_all(&mut self) {
        self.interner.ht_cap = 0;
        self.interner.entries_cap = 0;
        self.interner.next_id = 0;
        self.old_off = VecU32::empty();
        self.new_off = VecU32::empty();
        self.a_ids = VecU32::empty();
        self.b_ids = VecU32::empty();
        self.last_a = VecU32::empty();
        self.last_b = VecU32::empty();
        self.v = VecU32::empty();
        self.v_half = 0;
        self.tr_x = VecU32::empty();
        self.tr_choice = VecU8::empty();
        self.row_base = VecU32::empty();
        self.row_lo = VecU32::empty();
        self.seg_ops = VecU32::empty();
        self.chg_ptr = core::ptr::null_mut();
        self.chg_len = 0;
        self.chg_cap = 0;
        self.out = VecU8::empty();
    }

    #[inline(never)]
    unsafe fn presize(&mut self, est_lines: usize) {
        self.old_off.ensure(est_lines + 1);
        self.new_off.ensure(est_lines + 1);
        self.a_ids.ensure(est_lines);
        self.b_ids.ensure(est_lines);
        self.tr_x.ensure(4096);
        self.tr_choice.ensure(4096);
        self.row_base.ensure(1024);
        self.row_lo.ensure(1024);
        self.seg_ops.ensure(1024);
        if self.chg_cap < 4096 {
            self.grow_chg(4096);
        }
        if self.v_half == 0 {
            self.ensure_v(1024);
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow_chg(&mut self, need: usize) {
        let mut ncap = if self.chg_cap == 0 { 4096 } else { self.chg_cap << 1 };
        while ncap < need {
            ncap <<= 1;
        }
        let nptr = arena_alloc(ncap * 16, 16) as *mut Chg;
        if self.chg_len > 0 {
            core::ptr::copy_nonoverlapping(self.chg_ptr, nptr, self.chg_len);
        }
        self.chg_ptr = nptr;
        self.chg_cap = ncap;
    }

    #[cold]
    #[inline(never)]
    unsafe fn ensure_v(&mut self, half_need: usize) {
        if half_need <= self.v_half {
            return;
        }
        let mut nh = if self.v_half == 0 { 1024 } else { self.v_half << 1 };
        while nh < half_need {
            nh <<= 1;
        }
        let nlen = nh * 2 + 4;
        let nptr = arena_alloc(nlen * 4, 16) as *mut i32;
        let mut i = 0;
        while i < nlen {
            *nptr.add(i) = DEAD;
            i += 1;
        }
        if self.v_half > 0 {
            let old_p = self.v.ptr as *const i32;
            let old_len = self.v_half * 2 + 4;
            let shift = nh - self.v_half;
            core::ptr::copy_nonoverlapping(old_p, nptr.add(shift), old_len);
        }
        self.v.ptr = nptr as *mut u32;
        self.v.cap = nlen;
        self.v.len = nlen;
        self.v_half = nh;
    }
}

// ---------------------------------------------------------------------------
// Canonical JSON string scanners (SIMD128 + Predictive Bulk Memcmp for `new`!)
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn scan_json_old(
    ws: &mut Workspace,
    base: *const u8,
    start: usize,
    file_len: usize,
    bare_out: &mut bool,
) -> usize {
    ws.old_off.len = 0;
    ws.a_ids.len = 0;
    *bare_out = false;

    let splat_bs = i8x16_splat(b'\\' as i8);
    let splat_qt = i8x16_splat(b'"' as i8);

    let mut p = start;
    let mut line_start = start;
    let mut h = HASH_SEED;
    let mut rehash = false;
    let mut n_lines: u32 = 0;

    ws.old_off.push(start as u32);

    loop {
        if p >= file_len {
            return usize::MAX;
        }
        let v = v128_load(base.add(p) as *const v128);
        let mask = i8x16_bitmask(v128_or(i8x16_eq(v, splat_bs), i8x16_eq(v, splat_qt)));
        if mask == 0 {
            h = (h ^ u64x2_extract_lane::<0>(v)).wrapping_mul(K1);
            h = (h ^ u64x2_extract_lane::<1>(v)).wrapping_mul(K2);
            p += 16;
            continue;
        }
        let tz = mask.trailing_zeros() as usize;
        let hit = p + tz;
        if hit >= file_len {
            return usize::MAX;
        }
        let line_end: usize;
        let is_eof: bool;
        if *base.add(hit) == b'\\' {
            if hit + 1 >= file_len {
                return usize::MAX;
            }
            let nxt = *base.add(hit + 1);
            if nxt == b'n' {
                line_end = hit + 2;
                is_eof = false;
            } else if nxt == b'r' {
                if hit + 3 < file_len && *base.add(hit + 2) == b'\\' && *base.add(hit + 3) == b'n' {
                    line_end = hit + 4;
                    is_eof = false;
                } else {
                    p = hit + 2;
                    rehash = true;
                    continue;
                }
            } else if nxt == b'"' || nxt == b'\\' || nxt == b't' || nxt == b'b' || nxt == b'f' {
                p = hit + 2;
                rehash = true;
                continue;
            } else {
                return usize::MAX;
            }
        } else {
            line_end = hit;
            is_eof = true;
        }

        if is_eof {
            if line_end > line_start {
                let line_len = line_end - line_start;
                let final_h = if !rehash {
                    mix_tail(h, base.add(p), line_end - p)
                } else {
                    hash_bytes(base.add(line_start), line_len)
                };
                let id = ws.interner.intern_old(base, line_start as u32, line_len as u32, final_h, n_lines);
                ws.a_ids.push(id);
                ws.old_off.push(line_end as u32);
                *bare_out = true;
            }
            return line_end + 1;
        }

        let line_len = line_end - line_start;
        let final_h = if !rehash {
            let rem = line_end - p;
            if rem < 16 {
                mix_tail(h, base.add(p), rem)
            } else {
                let v0 = v128_load(base.add(p) as *const v128);
                let h1 = (h ^ u64x2_extract_lane::<0>(v0)).wrapping_mul(K1);
                let h2 = (h1 ^ u64x2_extract_lane::<1>(v0)).wrapping_mul(K2);
                mix_tail(h2, base.add(p + 16), rem - 16)
            }
        } else {
            hash_bytes(base.add(line_start), line_len)
        };
        let id = ws.interner.intern_old(base, line_start as u32, line_len as u32, final_h, n_lines);
        ws.a_ids.push(id);
        ws.old_off.push(line_end as u32);
        n_lines += 1;
        p = line_end;
        line_start = line_end;
        h = HASH_SEED;
        rehash = false;
    }
}

#[inline(always)]
unsafe fn bulk_match_old(
    ws: &mut Workspace,
    base: *const u8,
    p: &mut usize,
    end: usize,
    pred: &mut usize,
    n_old_nl: usize,
    old_limit: usize,
) {
    let pr = *pred;
    if pr >= n_old_nl || *p >= end {
        return;
    }
    let old_start = *ws.old_off.ptr.add(pr) as usize;
    let avail_old = old_limit - old_start;
    let avail_new = end - *p;
    let max_cmp = if avail_old < avail_new { avail_old } else { avail_new };
    let mut matched = 0usize;
    let a_base = base.add(old_start);
    let b_base = base.add(*p);
    while matched + 16 <= max_cmp {
        let eq = i8x16_eq(
            v128_load(a_base.add(matched) as *const v128),
            v128_load(b_base.add(matched) as *const v128),
        );
        if i8x16_all_true(eq) {
            matched += 16;
        } else {
            matched += ((!i8x16_bitmask(eq)) & 0xFFFF).trailing_zeros() as usize;
            break;
        }
    }
    if matched + 16 > max_cmp {
        while matched < max_cmp && *a_base.add(matched) == *b_base.add(matched) {
            matched += 1;
        }
    }
    let old_reached = old_start + matched;
    let old_off_p = ws.old_off.ptr;
    if old_reached < (*old_off_p.add(pr + 1) as usize) {
        return;
    }
    let delta = (*p as isize) - (old_start as isize);
    let mut k = pr + 1;
    while k < n_old_nl && (*old_off_p.add(k + 1) as usize) <= old_reached {
        k += 1;
    }
    let count = k - pr;
    ws.b_ids.ensure(ws.b_ids.len + count);
    ws.new_off.ensure(ws.new_off.len + count);
    core::ptr::copy_nonoverlapping(ws.a_ids.ptr.add(pr), ws.b_ids.ptr.add(ws.b_ids.len), count);
    ws.b_ids.len += count;

    // SIMD128 4-wide offset relocation: new_off[i] = old_off[pr + 1 + i] + delta
    let src_off = ws.old_off.ptr.add(pr + 1);
    let dst_off = ws.new_off.ptr.add(ws.new_off.len);
    let v_delta = i32x4_splat(delta as i32);
    let mut i = 0usize;
    while i + 4 <= count {
        let vo = v128_load(src_off.add(i) as *const v128);
        v128_store(dst_off.add(i) as *mut v128, i32x4_add(vo, v_delta));
        i += 4;
    }
    while i < count {
        *dst_off.add(i) = ((*src_off.add(i) as isize) + delta) as u32;
        i += 1;
    }
    ws.new_off.len += count;
    *p = ((*ws.old_off.ptr.add(k) as isize) + delta) as usize;
    *pred = k;
}

#[inline(never)]
unsafe fn scan_json_new(
    ws: &mut Workspace,
    base: *const u8,
    start: usize,
    file_len: usize,
    old_bare: bool,
    bare_out: &mut bool,
) -> usize {
    ws.new_off.len = 0;
    ws.b_ids.len = 0;
    *bare_out = false;

    let n_old = ws.a_ids.len;
    let n_old_nl = if old_bare && n_old > 0 { n_old - 1 } else { n_old };
    let old_limit = if n_old_nl > 0 { *ws.old_off.ptr.add(n_old_nl) as usize } else { 0 };

    let splat_bs = i8x16_splat(b'\\' as i8);
    let splat_qt = i8x16_splat(b'"' as i8);

    let mut p = start;
    let mut pred: usize = 0;
    ws.new_off.push(start as u32);

    loop {
        bulk_match_old(ws, base, &mut p, file_len, &mut pred, n_old_nl, old_limit);

        let line_start = p;
        let mut h = HASH_SEED;
        let mut rehash = false;
        let line_end: usize;
        let mut is_eof = false;
        loop {
            if p >= file_len {
                return usize::MAX;
            }
            let v = v128_load(base.add(p) as *const v128);
            let mask = i8x16_bitmask(v128_or(i8x16_eq(v, splat_bs), i8x16_eq(v, splat_qt)));
            if mask == 0 {
                h = (h ^ u64x2_extract_lane::<0>(v)).wrapping_mul(K1);
                h = (h ^ u64x2_extract_lane::<1>(v)).wrapping_mul(K2);
                p += 16;
                continue;
            }
            let tz = mask.trailing_zeros() as usize;
            let hit = p + tz;
            if hit >= file_len {
                return usize::MAX;
            }
            if *base.add(hit) == b'\\' {
                if hit + 1 >= file_len {
                    return usize::MAX;
                }
                let nxt = *base.add(hit + 1);
                if nxt == b'n' {
                    line_end = hit + 2;
                    break;
                } else if nxt == b'r' {
                    if hit + 3 < file_len && *base.add(hit + 2) == b'\\' && *base.add(hit + 3) == b'n' {
                        line_end = hit + 4;
                        break;
                    }
                    p = hit + 2;
                    rehash = true;
                    continue;
                } else if nxt == b'"' || nxt == b'\\' || nxt == b't' || nxt == b'b' || nxt == b'f' {
                    p = hit + 2;
                    rehash = true;
                    continue;
                } else {
                    return usize::MAX;
                }
            } else {
                line_end = hit;
                is_eof = true;
                break;
            }
        }

        if is_eof {
            if line_end > line_start {
                let line_len = line_end - line_start;
                let final_h = if !rehash {
                    mix_tail(h, base.add(p), line_end - p)
                } else {
                    hash_bytes(base.add(line_start), line_len)
                };
                let id = ws.interner.intern_new(base, line_start as u32, line_len as u32, final_h);
                ws.b_ids.push(id);
                ws.new_off.push(line_end as u32);
                *bare_out = true;
            }
            return line_end + 1;
        }

        let line_len = line_end - line_start;
        let final_h = if !rehash {
            let rem = line_end - p;
            if rem < 16 {
                mix_tail(h, base.add(p), rem)
            } else {
                let v0 = v128_load(base.add(p) as *const v128);
                let h1 = (h ^ u64x2_extract_lane::<0>(v0)).wrapping_mul(K1);
                let h2 = (h1 ^ u64x2_extract_lane::<1>(v0)).wrapping_mul(K2);
                mix_tail(h2, base.add(p + 16), rem - 16)
            }
        } else {
            hash_bytes(base.add(line_start), line_len)
        };
        let id = ws.interner.intern_new(base, line_start as u32, line_len as u32, final_h);
        ws.b_ids.push(id);
        ws.new_off.push(line_end as u32);
        p = line_end;

        let op = (*ws.interner.entries.add(id as usize)).old_pos;
        if (op & UNIQUE_BIT) != 0 {
            pred = ((op & 0x3FFF_FFFF) as usize) + 1;
        } else if pred < n_old_nl {
            if *ws.a_ids.ptr.add(pred) == id {
                pred += 1;
            } else if pred + 1 < n_old_nl && *ws.a_ids.ptr.add(pred + 1) == id {
                pred += 2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Raw UTF-8 line scanners (for solve.js `run({ old, new })`)
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn scan_raw_old(ws: &mut Workspace, base: *const u8, start: usize, len: usize, bare_out: &mut bool) {
    ws.old_off.len = 0;
    ws.a_ids.len = 0;
    *bare_out = false;
    ws.old_off.push(start as u32);
    if len == 0 {
        return;
    }
    let end = start + len;
    let splat_nl = i8x16_splat(b'\n' as i8);
    let mut p = start;
    let mut line_start = start;
    let mut h = HASH_SEED;
    let mut n_lines: u32 = 0;

    while p < end {
        let avail = end - p;
        let line_end: usize;
        let mut is_bare = false;
        if avail >= 16 {
            let v = v128_load(base.add(p) as *const v128);
            let mask = i8x16_bitmask(i8x16_eq(v, splat_nl));
            if mask == 0 {
                h = (h ^ u64x2_extract_lane::<0>(v)).wrapping_mul(K1);
                h = (h ^ u64x2_extract_lane::<1>(v)).wrapping_mul(K2);
                p += 16;
                continue;
            }
            line_end = p + (mask.trailing_zeros() as usize) + 1;
        } else {
            let mut q = p;
            while q < end && *base.add(q) != b'\n' {
                q += 1;
            }
            if q < end {
                line_end = q + 1;
            } else {
                line_end = end;
                is_bare = true;
            }
        }
        let rem = line_end - p;
        let final_h = if rem < 16 {
            mix_tail(h, base.add(p), rem)
        } else {
            let v0 = v128_load(base.add(p) as *const v128);
            let h1 = (h ^ u64x2_extract_lane::<0>(v0)).wrapping_mul(K1);
            let h2 = (h1 ^ u64x2_extract_lane::<1>(v0)).wrapping_mul(K2);
            mix_tail(h2, base.add(p + 16), 0)
        };
        let id = ws.interner.intern_old(base, line_start as u32, (line_end - line_start) as u32, final_h, n_lines);
        ws.a_ids.push(id);
        ws.old_off.push(line_end as u32);
        n_lines += 1;
        if is_bare {
            *bare_out = true;
            return;
        }
        p = line_end;
        line_start = line_end;
        h = HASH_SEED;
    }
    if line_start < end {
        let final_h = mix_tail(h, base.add(end), 0);
        let id = ws.interner.intern_old(base, line_start as u32, (end - line_start) as u32, final_h, n_lines);
        ws.a_ids.push(id);
        ws.old_off.push(end as u32);
        *bare_out = true;
    }
}

#[inline(never)]
unsafe fn scan_raw_new(ws: &mut Workspace, base: *const u8, start: usize, len: usize, old_bare: bool, bare_out: &mut bool) {
    ws.new_off.len = 0;
    ws.b_ids.len = 0;
    *bare_out = false;
    ws.new_off.push(start as u32);
    if len == 0 {
        return;
    }
    let end = start + len;
    let n_old = ws.a_ids.len;
    let n_old_nl = if old_bare && n_old > 0 { n_old - 1 } else { n_old };
    let old_limit = if n_old_nl > 0 { *ws.old_off.ptr.add(n_old_nl) as usize } else { 0 };
    let splat_nl = i8x16_splat(b'\n' as i8);
    let mut p = start;
    let mut pred: usize = 0;

    while p < end {
        bulk_match_old(ws, base, &mut p, end, &mut pred, n_old_nl, old_limit);
        if p >= end {
            return;
        }
        let line_start = p;
        let mut h = HASH_SEED;
        let mut line_end = end;
        let mut is_bare = true;
        while p < end {
            if end - p >= 16 {
                let v = v128_load(base.add(p) as *const v128);
                let mask = i8x16_bitmask(i8x16_eq(v, splat_nl));
                if mask == 0 {
                    h = (h ^ u64x2_extract_lane::<0>(v)).wrapping_mul(K1);
                    h = (h ^ u64x2_extract_lane::<1>(v)).wrapping_mul(K2);
                    p += 16;
                    continue;
                }
                line_end = p + (mask.trailing_zeros() as usize) + 1;
                is_bare = false;
                break;
            } else {
                let mut q = p;
                while q < end && *base.add(q) != b'\n' {
                    q += 1;
                }
                if q < end {
                    line_end = q + 1;
                    is_bare = false;
                }
                break;
            }
        }
        let rem = line_end - p;
        let final_h = if rem < 16 {
            mix_tail(h, base.add(p), rem)
        } else {
            let v0 = v128_load(base.add(p) as *const v128);
            let h1 = (h ^ u64x2_extract_lane::<0>(v0)).wrapping_mul(K1);
            let h2 = (h1 ^ u64x2_extract_lane::<1>(v0)).wrapping_mul(K2);
            mix_tail(h2, base.add(p + 16), 0)
        };
        let id = ws.interner.intern_new(base, line_start as u32, (line_end - line_start) as u32, final_h);
        ws.b_ids.push(id);
        ws.new_off.push(line_end as u32);
        if is_bare {
            *bare_out = true;
            return;
        }
        p = line_end;
        let op = (*ws.interner.entries.add(id as usize)).old_pos;
        if (op & UNIQUE_BIT) != 0 {
            pred = ((op & 0x3FFF_FFFF) as usize) + 1;
        } else if pred < n_old_nl {
            if *ws.a_ids.ptr.add(pred) == id {
                pred += 1;
            } else if pred + 1 < n_old_nl && *ws.a_ids.ptr.add(pred + 1) == id {
                pred += 2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exact Zero-LIS Dominance-Pruned Myers with Width-1 Trace Flushing
// ---------------------------------------------------------------------------

#[inline(never)]
unsafe fn flush_segment(
    ws: &mut Workspace,
    rel_d: usize,
    end_x: i32,
    end_k: i32,
    seg_x: i32,
    cur_o: &mut u32,
    cur_p: &mut u32,
    total_del: &mut u32,
    total_ins: &mut u32,
) {
    ws.seg_ops.len = 0;
    let mut cx = end_x;
    let mut ck = end_k;
    let row_base_p = ws.row_base.ptr;
    let row_lo_p = ws.row_lo.ptr as *const i32;
    let tr_x_p = ws.tr_x.ptr as *const i32;
    let tr_c_p = ws.tr_choice.ptr;

    let mut rd = rel_d;
    while rd >= 1 {
        let base = *row_base_p.add(rd) as usize;
        let rlo = *row_lo_p.add(rd);
        let idx = base + (((ck - rlo) >> 1) as usize);
        let is_del = *tr_c_p.add(idx);
        let prev_k = if is_del != 0 { ck - 1 } else { ck + 1 };
        let prev_x = if rd == 1 {
            seg_x
        } else {
            let pbase = *row_base_p.add(rd - 1) as usize;
            let prlo = *row_lo_p.add(rd - 1);
            *tr_x_p.add(pbase + (((prev_k - prlo) >> 1) as usize))
        };
        let start_x = if is_del != 0 { prev_x + 1 } else { prev_x };
        let snake = (cx - start_x) as u32;
        if snake > 0 {
            ws.seg_ops.push(snake);
        }
        ws.seg_ops.push((if is_del != 0 { 1u32 << 30 } else { 2u32 << 30 }) | 1);
        cx = prev_x;
        ck = prev_k;
        rd -= 1;
    }

    let mut co = *cur_o;
    let mut cp = *cur_p;
    let mut td = *total_del;
    let mut ti = *total_ins;
    let mut i = ws.seg_ops.len;
    while i > 0 {
        i -= 1;
        let op = *ws.seg_ops.ptr.add(i);
        let kind = op >> 30;
        let count = op & 0x3FFF_FFFF;
        if kind == 0 {
            co += count;
            cp += count;
            continue;
        }
        let n = ws.chg_len;
        let merge = if n > 0 {
            let lc = ws.chg_ptr.add(n - 1);
            (*lc).o + (*lc).dels == co && (*lc).p + (*lc).ins == cp
        } else {
            false
        };
        if !merge {
            if n == ws.chg_cap {
                ws.grow_chg(n + 1);
            }
            *ws.chg_ptr.add(n) = Chg { o: co, p: cp, dels: 0, ins: 0 };
            ws.chg_len = n + 1;
        }
        let lc = ws.chg_ptr.add(ws.chg_len - 1);
        if kind == 1 {
            (*lc).dels += count;
            co += count;
            td += count;
        } else {
            (*lc).ins += count;
            cp += count;
            ti += count;
        }
    }
    *cur_o = co;
    *cur_p = cp;
    *total_del = td;
    *total_ins = ti;
}

#[inline(never)]
unsafe fn run_myers(ws: &mut Workspace) -> (u32, u32) {
    let n0 = ws.a_ids.len;
    let m0 = ws.b_ids.len;
    let a_all = ws.a_ids.ptr;
    let b_all = ws.b_ids.ptr;

    let mut prefix = 0usize;
    let min_nm = if n0 < m0 { n0 } else { m0 };
    while prefix < min_nm && *a_all.add(prefix) == *b_all.add(prefix) {
        prefix += 1;
    }
    let mut suffix = 0usize;
    let rem_a = n0 - prefix;
    let rem_b = m0 - prefix;
    let min_rem = if rem_a < rem_b { rem_a } else { rem_b };
    while suffix < min_rem && *a_all.add(n0 - 1 - suffix) == *b_all.add(m0 - 1 - suffix) {
        suffix += 1;
    }

    let n = n0 - prefix - suffix;
    let m = m0 - prefix - suffix;
    ws.chg_len = 0;

    if n == 0 && m == 0 {
        return (0, 0);
    }
    if ws.chg_cap == 0 {
        ws.grow_chg(4096);
    }
    if n == 0 {
        *ws.chg_ptr = Chg { o: prefix as u32, p: prefix as u32, dels: 0, ins: m as u32 };
        ws.chg_len = 1;
        return (0, m as u32);
    }
    if m == 0 {
        *ws.chg_ptr = Chg { o: prefix as u32, p: prefix as u32, dels: n as u32, ins: 0 };
        ws.chg_len = 1;
        return (n as u32, 0);
    }

    // Ensure 4-element sentinel slack for branchless SIMD128 snake loop!
    ws.a_ids.ensure(n0 + 4);
    ws.b_ids.ensure(m0 + 4);
    let a_ptr = ws.a_ids.ptr.add(prefix);
    let b_ptr = ws.b_ids.ptr.add(prefix);
    *a_ptr.add(n) = 0xFFFF_FFFE;
    *b_ptr.add(m) = 0xFFFF_FFFF;
    let n_i = n as i32;
    let m_i = m as i32;
    let num_ids = ws.interner.next_id;

    ws.last_a.ensure(num_ids);
    ws.last_b.ensure(num_ids);
    core::ptr::write_bytes(ws.last_a.ptr, 0xFF, num_ids);
    core::ptr::write_bytes(ws.last_b.ptr, 0xFF, num_ids);
    let last_a = ws.last_a.ptr as *mut i32;
    let last_b = ws.last_b.ptr as *mut i32;
    let mut i = 0usize;
    while i < n {
        *last_a.add(*a_ptr.add(i) as usize) = i as i32;
        i += 1;
    }
    let mut j = 0usize;
    while j < m {
        *last_b.add(*b_ptr.add(j) as usize) = j as i32;
        j += 1;
    }

    if ws.v_half == 0 {
        ws.ensure_v(1024);
    }

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

    // Because common prefix was already stripped, a_ptr[0] != b_ptr[0], so x0 = 0!
    *(ws.v.ptr as *mut i32).add(ws.v_half) = 0;

    let max_edit = n + m;
    let mut d = 1usize;
    while d <= max_edit {
        rel_d += 1;
        let next_lo = if k_lo - 1 < -(d as i32) { -(d as i32) } else { k_lo - 1 };
        let next_hi = if k_hi + 1 > (d as i32) { d as i32 } else { k_hi + 1 };

        let need_left = (k_base - (next_lo - 1)).max(0) as usize;
        let need_right = ((next_hi + 1) - k_base).max(0) as usize;
        let need_half = if need_left > need_right { need_left } else { need_right } + 2;
        if need_half > ws.v_half {
            ws.ensure_v(need_half);
        }

        let row_len = (((next_hi - next_lo) >> 1) + 1) as usize;
        ws.tr_x.len = tr_ptr;
        ws.tr_choice.len = tr_ptr;
        ws.row_base.len = rel_d;
        ws.row_lo.len = rel_d;
        ws.tr_x.ensure(tr_ptr + row_len);
        ws.tr_choice.ensure(tr_ptr + row_len);
        ws.row_base.ensure(rel_d + 1);
        ws.row_lo.ensure(rel_d + 1);

        *ws.row_base.ptr.add(rel_d) = tr_ptr as u32;
        *(ws.row_lo.ptr as *mut i32).add(rel_d) = next_lo;

        let v_ptr = ws.v.ptr as *mut i32;
        let v_mid = ws.v_half as isize - k_base as isize;
        let tr_x_p = ws.tr_x.ptr as *mut i32;
        let tr_c_p = ws.tr_choice.ptr;

        let mut new_lo = next_hi + 1;
        let mut new_hi = next_lo - 1;
        let mut best_sum: i32 = -1;
        let mut best_k: i32 = 0;
        let mut best_x: i32 = 0;
        let mut best_y: i32 = 0;
        let mut done = false;

        let mut k = next_lo;
        let mut idx = tr_ptr;
        while k <= next_hi {
            let rem_x = *v_ptr.offset(v_mid + (k as isize) - 1);
            let add_x = *v_ptr.offset(v_mid + (k as isize) + 1);
            let can_rem = rem_x != DEAD && rem_x < n_i;
            let can_add = add_x != DEAD && (add_x - k) <= m_i;
            if !can_rem && !can_add {
                *v_ptr.offset(v_mid + (k as isize)) = DEAD;
                *tr_x_p.add(idx) = DEAD;
                k += 2;
                idx += 1;
                continue;
            }
            let by_del = if can_rem { rem_x + 1 } else { -2 };
            let by_ins = if can_add { add_x } else { -2 };
            let (mut x, is_del) = if by_del >= by_ins { (by_del, 1u8) } else { (by_ins, 0u8) };
            let mut y = x - k;
            // Branchless sentinel-terminated SIMD128 4-wide snake extension!
            if *a_ptr.add(x as usize) == *b_ptr.add(y as usize) {
                x += 1;
                y += 1;
                while i32x4_all_true(i32x4_eq(
                    v128_load(a_ptr.add(x as usize) as *const v128),
                    v128_load(b_ptr.add(y as usize) as *const v128),
                )) {
                    x += 4;
                    y += 4;
                }
                while *a_ptr.add(x as usize) == *b_ptr.add(y as usize) {
                    x += 1;
                    y += 1;
                }
            }
            *v_ptr.offset(v_mid + (k as isize)) = x;
            *tr_x_p.add(idx) = x;
            *tr_c_p.add(idx) = is_del;
            if k < new_lo { new_lo = k; }
            if k > new_hi { new_hi = k; }
            let sum = x + y;
            if sum > best_sum {
                best_sum = sum;
                best_k = k;
                best_x = x;
                best_y = y;
            }
            if x >= n_i && y >= m_i {
                done = true;
                break;
            }
            k += 2;
            idx += 1;
        }
        tr_ptr += row_len;

        let mut ck = k_lo;
        while ck <= k_hi {
            *v_ptr.offset(v_mid + (ck as isize)) = DEAD;
            ck += 2;
        }

        if done {
            let mut ck2 = new_lo;
            while ck2 <= new_hi {
                *v_ptr.offset(v_mid + (ck2 as isize)) = DEAD;
                ck2 += 2;
            }
            flush_segment(ws, rel_d, n_i, n_i - m_i, seg_x, &mut cur_o, &mut cur_p, &mut total_del, &mut total_ins);
            break;
        }

        let mut p_lo = best_k;
        let mut p_hi = best_k;

        let mut b_bad: i32 = 0;
        let mut s: i32 = 0;
        let mut kl = best_k - 2;
        while kl >= new_lo {
            let i2 = best_x - s - 2;
            if i2 < 0 {
                let mut q = new_lo;
                while q <= kl {
                    if *v_ptr.offset(v_mid + (q as isize)) != DEAD {
                        p_lo = q;
                        break;
                    }
                    q += 2;
                }
                break;
            }
            if *last_b.add(*a_ptr.add((i2 + 1) as usize) as usize) >= best_y { b_bad += 1; }
            if *last_b.add(*a_ptr.add(i2 as usize) as usize) >= best_y { b_bad += 1; }
            s += 2;
            let x = *v_ptr.offset(v_mid + (kl as isize));
            if x != DEAD {
                if x <= best_x && (x - kl) <= best_y && (b_bad << 1) < s {
                    *v_ptr.offset(v_mid + (kl as isize)) = DEAD;
                } else {
                    p_lo = kl;
                }
            }
            kl -= 2;
        }

        let mut a_bad: i32 = 0;
        s = 0;
        let mut kr = best_k + 2;
        while kr <= new_hi {
            let j2 = best_y - s - 2;
            if j2 < 0 {
                let mut q = new_hi;
                while q >= kr {
                    if *v_ptr.offset(v_mid + (q as isize)) != DEAD {
                        p_hi = q;
                        break;
                    }
                    q -= 2;
                }
                break;
            }
            if *last_a.add(*b_ptr.add((j2 + 1) as usize) as usize) >= best_x { a_bad += 1; }
            if *last_a.add(*b_ptr.add(j2 as usize) as usize) >= best_x { a_bad += 1; }
            s += 2;
            let x = *v_ptr.offset(v_mid + (kr as isize));
            if x != DEAD {
                if x <= best_x && (x - kr) <= best_y && (a_bad << 1) < s {
                    *v_ptr.offset(v_mid + (kr as isize)) = DEAD;
                } else {
                    p_hi = kr;
                }
            }
            kr += 2;
        }

        k_lo = p_lo;
        k_hi = p_hi;

        if k_lo == k_hi {
            let cx = *v_ptr.offset(v_mid + (k_lo as isize));
            *v_ptr.offset(v_mid + (k_lo as isize)) = DEAD;
            flush_segment(ws, rel_d, cx, k_lo, seg_x, &mut cur_o, &mut cur_p, &mut total_del, &mut total_ins);
            seg_x = cx;
            k_base = k_lo;
            *(ws.v.ptr as *mut i32).add(ws.v_half) = cx;
            rel_d = 0;
            tr_ptr = 0;
        }

        d += 1;
    }

    (total_del, total_ins)
}

// ---------------------------------------------------------------------------
// Fast single-pointer hunk formatter (1 ensure per hunk, zero per-line checks!)
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn emit_line(
    mut dst: *mut u8,
    base: *const u8,
    offs: *const u32,
    idx: u32,
    op: u8,
    is_bare_last: bool,
    json_mode: bool,
) -> *mut u8 {
    let s = *offs.add(idx as usize) as usize;
    let e = *offs.add(idx as usize + 1) as usize;
    *dst = op;
    dst = copy_fast(dst.add(1), base.add(s), e - s);
    if is_bare_last {
        if json_mode {
            dst = copy_fast(dst, b"\\n%\\n".as_ptr(), 5);
        } else {
            *dst = b'\n';
            *dst.add(1) = b'%';
            *dst.add(2) = b'\n';
            dst = dst.add(3);
        }
    }
    dst
}

#[inline(never)]
unsafe fn format_hunks(
    ws: &mut Workspace,
    base: *const u8,
    old_bare: bool,
    new_bare: bool,
    json_mode: bool,
    total_del: u32,
    total_ins: u32,
) {
    let n0 = ws.a_ids.len as u32;
    let m0 = ws.b_ids.len as u32;
    let n_chg = ws.chg_len;
    let chg = ws.chg_ptr;

    let mut num_hunks: u32 = 0;
    if n_chg > 0 {
        num_hunks = 1;
        let mut ci = 1usize;
        while ci < n_chg {
            let prev = *chg.add(ci - 1);
            if (*chg.add(ci)).o - (prev.o + prev.dels) > 6 {
                num_hunks += 1;
            }
            ci += 1;
        }
    }

    ws.out.ensure(ws.out.len + 128);
    let mut dst = ws.out.ptr.add(ws.out.len);
    dst = copy_fast(dst, b"makefun-diff/1 old=".as_ptr(), 19);
    dst = write_u32_dec(dst, n0);
    dst = copy_fast(dst, b" new=".as_ptr(), 5);
    dst = write_u32_dec(dst, m0);
    dst = copy_fast(dst, b" hunks=".as_ptr(), 7);
    dst = write_u32_dec(dst, num_hunks);
    dst = copy_fast(dst, b" deleted=".as_ptr(), 9);
    dst = write_u32_dec(dst, total_del);
    dst = copy_fast(dst, b" inserted=".as_ptr(), 10);
    dst = write_u32_dec(dst, total_ins);
    if json_mode {
        *dst = b'\\';
        *dst.add(1) = b'n';
        dst = dst.add(2);
    } else {
        *dst = b'\n';
        dst = dst.add(1);
    }
    ws.out.len = (dst as usize) - (ws.out.ptr as usize);

    if n_chg == 0 {
        return;
    }

    let old_off = ws.old_off.ptr;
    let new_off = ws.new_off.ptr;

    let mut h_start = 0usize;
    while h_start < n_chg {
        let mut h_end = h_start + 1;
        while h_end < n_chg {
            let prev = *chg.add(h_end - 1);
            if (*chg.add(h_end)).o - (prev.o + prev.dels) > 6 {
                break;
            }
            h_end += 1;
        }
        let first = *chg.add(h_start);
        let last = *chg.add(h_end - 1);
        let old_start = if first.o > 3 { first.o - 3 } else { 0 };
        let new_start = first.p - (first.o - old_start);
        let last_o_end = last.o + last.dels;
        let last_p_end = last.p + last.ins;
        let after = {
            let rem = n0 - last_o_end;
            if rem < 3 { rem } else { 3 }
        };
        let old_end = last_o_end + after;
        let new_end = last_p_end + after;

        // Exact upper bound on bytes needed for this entire hunk!
        let old_bytes = (*old_off.add(old_end as usize) - *old_off.add(old_start as usize)) as usize;
        let new_bytes = (*new_off.add(new_end as usize) - *new_off.add(new_start as usize)) as usize;
        let line_cnt = (old_end - old_start + new_end - new_start) as usize;
        ws.out.ensure(ws.out.len + 64 + old_bytes + new_bytes + line_cnt + 16);

        let mut dst = ws.out.ptr.add(ws.out.len);
        *dst = b'@';
        *dst.add(1) = b' ';
        dst = write_u32_dec(dst.add(2), old_start + 1);
        *dst = b' ';
        dst = write_u32_dec(dst.add(1), old_end - old_start);
        *dst = b' ';
        dst = write_u32_dec(dst.add(1), new_start + 1);
        *dst = b' ';
        dst = write_u32_dec(dst.add(1), new_end - new_start);
        if json_mode {
            *dst = b'\\';
            *dst.add(1) = b'n';
            dst = dst.add(2);
        } else {
            *dst = b'\n';
            dst = dst.add(1);
        }

        let mut at = old_start;
        let mut ci = h_start;
        while ci < h_end {
            let c = *chg.add(ci);
            while at < c.o {
                dst = emit_line(dst, base, old_off, at, b'=', old_bare && at + 1 == n0, json_mode);
                at += 1;
            }
            let mut k = 0u32;
            while k < c.dels {
                let idx = c.o + k;
                dst = emit_line(dst, base, old_off, idx, b'-', old_bare && idx + 1 == n0, json_mode);
                k += 1;
            }
            let mut k = 0u32;
            while k < c.ins {
                let idx = c.p + k;
                dst = emit_line(dst, base, new_off, idx, b'+', new_bare && idx + 1 == m0, json_mode);
                k += 1;
            }
            at = c.o + c.dels;
            ci += 1;
        }
        while at < old_end {
            dst = emit_line(dst, base, old_off, at, b'=', old_bare && at + 1 == n0, json_mode);
            at += 1;
        }

        ws.out.len = (dst as usize) - (ws.out.ptr as usize);
        h_start = h_end;
    }
}

// ---------------------------------------------------------------------------
// Minimal compliant JSON wrapper parser for `/input/cases.json`
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn skip_ws(base: *const u8, mut p: usize, len: usize) -> usize {
    while p < len {
        let b = *base.add(p);
        if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
            p += 1;
        } else {
            break;
        }
    }
    p
}

#[inline(never)]
unsafe fn skip_json_string(base: *const u8, mut p: usize, len: usize) -> usize {
    let splat_bs = i8x16_splat(b'\\' as i8);
    let splat_qt = i8x16_splat(b'"' as i8);
    loop {
        if p >= len {
            return usize::MAX;
        }
        let v = v128_load(base.add(p) as *const v128);
        let mask = i8x16_bitmask(v128_or(i8x16_eq(v, splat_bs), i8x16_eq(v, splat_qt)));
        if mask == 0 {
            p += 16;
            continue;
        }
        let hit = p + (mask.trailing_zeros() as usize);
        if hit >= len {
            return usize::MAX;
        }
        if *base.add(hit) == b'\\' {
            p = hit + 2;
        } else {
            return hit + 1;
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn skip_json_value(base: *const u8, mut p: usize, len: usize) -> usize {
    p = skip_ws(base, p, len);
    if p >= len {
        return usize::MAX;
    }
    match *base.add(p) {
        b'"' => skip_json_string(base, p + 1, len),
        b'{' => {
            p = skip_ws(base, p + 1, len);
            if p < len && *base.add(p) == b'}' {
                return p + 1;
            }
            loop {
                p = skip_ws(base, p, len);
                if p >= len || *base.add(p) != b'"' { return usize::MAX; }
                p = skip_json_string(base, p + 1, len);
                if p == usize::MAX { return usize::MAX; }
                p = skip_ws(base, p, len);
                if p >= len || *base.add(p) != b':' { return usize::MAX; }
                p = skip_json_value(base, p + 1, len);
                if p == usize::MAX { return usize::MAX; }
                p = skip_ws(base, p, len);
                if p >= len { return usize::MAX; }
                if *base.add(p) == b',' { p += 1; }
                else if *base.add(p) == b'}' { return p + 1; }
                else { return usize::MAX; }
            }
        }
        b'[' => {
            p = skip_ws(base, p + 1, len);
            if p < len && *base.add(p) == b']' {
                return p + 1;
            }
            loop {
                p = skip_json_value(base, p, len);
                if p == usize::MAX { return usize::MAX; }
                p = skip_ws(base, p, len);
                if p >= len { return usize::MAX; }
                if *base.add(p) == b',' { p += 1; }
                else if *base.add(p) == b']' { return p + 1; }
                else { return usize::MAX; }
            }
        }
        _ => {
            while p < len {
                let b = *base.add(p);
                if b == b',' || b == b'}' || b == b']' || b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                    break;
                }
                p += 1;
            }
            p
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn solve_cases_json(in_ptr: usize, in_len: usize) -> i32 {
    let ws = &mut *core::ptr::addr_of_mut!(WS);
    let base = in_ptr as *const u8;

    let est_lines = (in_len / 200).clamp(1024, 48000);
    ws.presize(est_lines);
    ws.out.len = 0;
    ws.out.ensure(((in_len >> 2) + 131072).max(65536));
    ws.out.push(b'[');

    let mut p = skip_ws(base, 0, in_len);
    if p >= in_len || *base.add(p) != b'[' {
        return -1;
    }
    p = skip_ws(base, p + 1, in_len);
    if p < in_len && *base.add(p) == b']' {
        if skip_ws(base, p + 1, in_len) != in_len { return -1; }
        ws.out.push(b']');
        return ws.out.len as i32;
    }

    let mut case_idx = 0usize;
    loop {
        p = skip_ws(base, p, in_len);
        if p >= in_len || *base.add(p) != b'{' { return -1; }
        p = skip_ws(base, p + 1, in_len);

        let mut solved_case = false;
        while p < in_len && *base.add(p) != b'}' {
            if *base.add(p) != b'"' { return -1; }
            let k_start = p + 1;
            let k_end_q = skip_json_string(base, k_start, in_len);
            if k_end_q == usize::MAX { return -1; }
            let k_len = (k_end_q - 1) - k_start;
            p = skip_ws(base, k_end_q, in_len);
            if p >= in_len || *base.add(p) != b':' { return -1; }
            p = skip_ws(base, p + 1, in_len);

            if k_len == 5 && eq_bytes(base.add(k_start), b"input".as_ptr(), 5) {
                if p >= in_len || *base.add(p) != b'{' { return -1; }
                p = skip_ws(base, p + 1, in_len);
                let mut old_done = false;
                let mut new_done = false;
                let mut deferred_new_start: usize = usize::MAX;
                let mut old_bare = false;
                let mut new_bare = false;

                ws.interner.reset((in_len / 320).clamp(1024, 32768));

                while p < in_len && *base.add(p) != b'}' {
                    if *base.add(p) != b'"' { return -1; }
                    let ik_start = p + 1;
                    let ik_end_q = skip_json_string(base, ik_start, in_len);
                    if ik_end_q == usize::MAX { return -1; }
                    let ik_len = (ik_end_q - 1) - ik_start;
                    p = skip_ws(base, ik_end_q, in_len);
                    if p >= in_len || *base.add(p) != b':' { return -1; }
                    p = skip_ws(base, p + 1, in_len);

                    if ik_len == 3 && eq_bytes(base.add(ik_start), b"old".as_ptr(), 3) {
                        if old_done || p >= in_len || *base.add(p) != b'"' { return -1; }
                        p = scan_json_old(ws, base, p + 1, in_len, &mut old_bare);
                        if p == usize::MAX { return -1; }
                        old_done = true;
                        if deferred_new_start != usize::MAX {
                            let r = scan_json_new(ws, base, deferred_new_start, in_len, old_bare, &mut new_bare);
                            if r == usize::MAX { return -1; }
                            new_done = true;
                            deferred_new_start = usize::MAX;
                        }
                    } else if ik_len == 3 && eq_bytes(base.add(ik_start), b"new".as_ptr(), 3) {
                        if new_done || deferred_new_start != usize::MAX || p >= in_len || *base.add(p) != b'"' {
                            return -1;
                        }
                        if old_done {
                            p = scan_json_new(ws, base, p + 1, in_len, old_bare, &mut new_bare);
                            if p == usize::MAX { return -1; }
                            new_done = true;
                        } else {
                            deferred_new_start = p + 1;
                            p = skip_json_string(base, p + 1, in_len);
                            if p == usize::MAX { return -1; }
                        }
                    } else {
                        p = skip_json_value(base, p, in_len);
                        if p == usize::MAX { return -1; }
                    }
                    p = skip_ws(base, p, in_len);
                    if p < in_len && *base.add(p) == b',' {
                        p = skip_ws(base, p + 1, in_len);
                    }
                }
                if p >= in_len || *base.add(p) != b'}' || !old_done || !new_done { return -1; }
                p += 1;

                if case_idx > 0 { ws.out.push(b','); }
                ws.out.push(b'"');
                let (td, ti) = run_myers(ws);
                format_hunks(ws, base, old_bare, new_bare, true, td, ti);
                ws.out.push(b'"');
                solved_case = true;
            } else {
                p = skip_json_value(base, p, in_len);
                if p == usize::MAX { return -1; }
            }
            p = skip_ws(base, p, in_len);
            if p < in_len && *base.add(p) == b',' {
                p = skip_ws(base, p + 1, in_len);
            }
        }
        if !solved_case || p >= in_len || *base.add(p) != b'}' { return -1; }
        p = skip_ws(base, p + 1, in_len);
        case_idx += 1;
        if p < in_len && *base.add(p) == b',' {
            p += 1;
            continue;
        }
        if p < in_len && *base.add(p) == b']' {
            if skip_ws(base, p + 1, in_len) != in_len { return -1; }
            ws.out.push(b']');
            return ws.out.len as i32;
        }
        return -1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn solve_raw_pair(buf_ptr: usize, old_len: usize, new_len: usize) -> i32 {
    let ws = &mut *core::ptr::addr_of_mut!(WS);
    let base = buf_ptr as *const u8;
    let est = ((old_len + new_len) / 64).clamp(256, 48000);
    ws.presize(est);
    ws.interner.reset(est.min(32768));
    let mut old_bare = false;
    let mut new_bare = false;
    scan_raw_old(ws, base, 0, old_len, &mut old_bare);
    scan_raw_new(ws, base, old_len, new_len, old_bare, &mut new_bare);
    ws.out.len = 0;
    let (td, ti) = run_myers(ws);
    format_hunks(ws, base, old_bare, new_bare, false, td, ti);
    ws.out.len as i32
}

#[no_mangle]
pub unsafe extern "C" fn get_out_ptr() -> usize {
    let ws = &*core::ptr::addr_of!(WS);
    ws.out.ptr as usize
}
