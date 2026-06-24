	.text
	.arch armv8.6-a+sve+i8mm
// void ld_rqb(uint64_t iters, void *buf) : 8 ld1rqb per iter
	.global ld_rqb
	.type ld_rqb,%function
ld_rqb:
	ptrue p0.b
1:	ld1rqb z0.b,p0/z,[x1]
	ld1rqb z1.b,p0/z,[x1,#16]
	ld1rqb z2.b,p0/z,[x1,#32]
	ld1rqb z3.b,p0/z,[x1,#48]
	ld1rqb z4.b,p0/z,[x1,#64]
	ld1rqb z5.b,p0/z,[x1,#80]
	ld1rqb z6.b,p0/z,[x1,#96]
	ld1rqb z7.b,p0/z,[x1,#112]
	subs x0,x0,#1
	b.gt 1b
	ret
	.size ld_rqb,.-ld_rqb
// void ld_1b(uint64_t iters, void *buf) : 8 ld1b per iter
	.global ld_1b
	.type ld_1b,%function
ld_1b:
	ptrue p0.b
1:	ld1b z0.b,p0/z,[x1]
	ld1b z1.b,p0/z,[x1,#1,mul vl]
	ld1b z2.b,p0/z,[x1,#2,mul vl]
	ld1b z3.b,p0/z,[x1,#3,mul vl]
	ld1b z4.b,p0/z,[x1,#4,mul vl]
	ld1b z5.b,p0/z,[x1,#5,mul vl]
	ld1b z6.b,p0/z,[x1,#6,mul vl]
	ld1b z7.b,p0/z,[x1,#7,mul vl]
	subs x0,x0,#1
	b.gt 1b
	ret
	.size ld_1b,.-ld_1b
// void ld_q(uint64_t iters, void *buf) : 8 ldr q (NEON) per iter
	.global ld_q
	.type ld_q,%function
ld_q:
1:	ldr q0,[x1]
	ldr q1,[x1,#16]
	ldr q2,[x1,#32]
	ldr q3,[x1,#48]
	ldr q4,[x1,#64]
	ldr q5,[x1,#80]
	ldr q6,[x1,#96]
	ldr q7,[x1,#112]
	subs x0,x0,#1
	b.gt 1b
	ret
	.size ld_q,.-ld_q
	.section .note.GNU-stack,"",@progbits
