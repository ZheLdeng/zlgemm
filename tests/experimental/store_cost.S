// Measure scatter vs contiguous st1w throughput on this core.
// x0 = iters, x1 = base buffer (>=4KB), offsets in z0 preloaded by caller? no: build here.
	.text
	.arch armv8.6-a+sve+i8mm

// void scatter_store(uint64_t iters, int32_t *buf)
// 16 scatter st1w per iter (mirrors one 8x16 i32 tile store), offsets = identity*1 (in-L1)
	.global scatter_store
	.type scatter_store, %function
scatter_store:
	ptrue p1.s
	index z0.s, #0, #1        // 0,1,2,...,7  (element offsets)
	mov z16.s, #7
1:
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	st1w z16.s, p1, [x1, z0.s, sxtw #2]
	subs x0, x0, #1
	b.gt 1b
	ret
	.size scatter_store, .-scatter_store

// void contig_store(uint64_t iters, int32_t *buf)
// 16 contiguous st1w per iter
	.global contig_store
	.type contig_store, %function
contig_store:
	ptrue p1.s
	mov z16.s, #7
1:
	st1w z16.s, p1, [x1]
	st1w z16.s, p1, [x1, #1, mul vl]
	st1w z16.s, p1, [x1, #2, mul vl]
	st1w z16.s, p1, [x1, #3, mul vl]
	st1w z16.s, p1, [x1, #4, mul vl]
	st1w z16.s, p1, [x1, #5, mul vl]
	st1w z16.s, p1, [x1, #6, mul vl]
	st1w z16.s, p1, [x1, #7, mul vl]
	subs x0, x0, #1
	b.gt 1b
	ret
	.size contig_store, .-contig_store
	.section .note.GNU-stack,"",@progbits
