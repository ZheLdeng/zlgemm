// Reproduce 96^3 crash with guard pages right after each buffer.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <ucontext.h>

static void *g_cbeg,*g_cend,*g_bbeg,*g_bend;
static void segv(int sig, siginfo_t *si, void *uc){
    (void)sig;
    void *a = si->si_addr;
    ucontext_t *u=(ucontext_t*)uc;
    unsigned long pc = u->uc_mcontext.pc;
    unsigned long *r = u->uc_mcontext.regs;
    const char *which = "?";
    if (a>=g_cbeg && a<=(void*)((char*)g_cend+8192)) which="C OVERFLOW";
    else if (a>=g_bbeg && a<=(void*)((char*)g_bend+8192)) which="Breo OVERREAD";
    char buf[300]; int n=snprintf(buf,sizeof buf,
        "FAULT addr=%p pc=0x%lx %s\n x2(Cblk)=0x%lx x16(n0)=%lu x18=0x%lx x12(ldc)=%lu x8(M)=%lu\n Cbeg=%p Cend=%p\n",
        a,pc,which, r[2], r[16], r[18], r[12], r[8], g_cbeg, g_cend);
    write(2,buf,n); _exit(42);
}

typedef int8_t i8_t; typedef int32_t i32_t;
void i8_pack_B(const i8_t*,i8_t*,int,int);
void i8gemm_mt_dispatch(const i8_t*,const i8_t*,i32_t*,int,int,int,int);

static int rup(int x,int q){return ((x+q-1)/q)*q;}

// allocate `bytes` with a PROT_NONE guard page immediately after.
static void* guard_alloc(size_t bytes, void **base_out, size_t *map_out){
    long pg = sysconf(_SC_PAGESIZE);
    size_t usable = (bytes + pg - 1) & ~((size_t)pg-1);
    size_t total = usable + pg;
    char *m = mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (m==MAP_FAILED){perror("mmap");exit(1);}
    mprotect(m + usable, pg, PROT_NONE);   // guard
    *base_out = m; *map_out = total;
    // place buffer so its END aligns to the guard boundary -> catch overflow
    return m + (usable - bytes);
}

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):96, K = argc>2?atoi(argv[2]):96, N = argc>3?atoi(argv[3]):96;
    int Kr=rup(K,16), Nr=rup(N,16);
    int8_t *A=calloc((size_t)M*Kr,1), *Bp=calloc((size_t)Kr*Nr,1);
    for(int i=0;i<M*Kr;i++)A[i]=(i%17)-8;
    for(int i=0;i<Kr*Nr;i++)Bp[i]=(i%13)-6;
    void *bb,*cb; size_t bm,cm;
    int8_t *Breo = guard_alloc((size_t)Kr*Nr,&bb,&bm);
    i8_pack_B(Bp,Breo,Kr,Nr);
    i32_t *C = guard_alloc((size_t)M*Nr*sizeof(i32_t),&cb,&cm);
    g_cbeg=C; g_cend=C+(size_t)M*Nr; g_bbeg=Breo; g_bend=Breo+(size_t)Kr*Nr;
    struct sigaction sa; memset(&sa,0,sizeof sa);
    sa.sa_sigaction=segv; sa.sa_flags=SA_SIGINFO; sigaction(SIGSEGV,&sa,NULL);
    fprintf(stderr,"M=%d Kr=%d Nr=%d C=%p Cend=%p Breo=%p Bend=%p\n",
            M,Kr,Nr,(void*)C,(void*)(C+(size_t)M*Nr),(void*)Breo,(void*)(Breo+(size_t)Kr*Nr));
    i8gemm_mt_dispatch(A,Breo,C,M,Kr,Nr,1);
    fprintf(stderr,"OK no fault\n");
    return 0;
}
