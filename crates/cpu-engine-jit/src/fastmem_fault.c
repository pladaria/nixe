#define _GNU_SOURCE

#include <setjmp.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void (*nixe_native_gateway)(void *frame, uintptr_t entry);

static struct sigaction previous_segv;
static struct sigaction previous_bus;
static pthread_once_t install_once = PTHREAD_ONCE_INIT;

static _Thread_local sigjmp_buf recovery;
static _Thread_local uintptr_t arena_start;
static _Thread_local uintptr_t arena_end;
static _Thread_local uintptr_t fault_address;
static _Thread_local sig_atomic_t active;

static void forward_signal(int signal_number, siginfo_t *information, void *context) {
    const struct sigaction *previous =
        signal_number == SIGBUS ? &previous_bus : &previous_segv;
    uintptr_t action = (uintptr_t)previous->sa_handler;
    if ((previous->sa_flags & SA_SIGINFO) != 0 &&
        previous->sa_sigaction != NULL &&
        action != (uintptr_t)SIG_DFL && action != (uintptr_t)SIG_IGN) {
        previous->sa_sigaction(signal_number, information, context);
        return;
    }
    if (previous->sa_handler == SIG_IGN) {
        return;
    }
    if (previous->sa_handler != SIG_DFL && previous->sa_handler != NULL) {
        previous->sa_handler(signal_number);
        return;
    }
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

static void handle_fault(int signal_number, siginfo_t *information, void *context) {
    uintptr_t address = (uintptr_t)information->si_addr;
    if (active && address >= arena_start && address < arena_end) {
        fault_address = address;
        active = 0;
        siglongjmp(recovery, 1);
    }
    forward_signal(signal_number, information, context);
}

static void install_handlers(void) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    sigemptyset(&action.sa_mask);
    action.sa_sigaction = handle_fault;
    action.sa_flags = SA_SIGINFO;
    if (sigaction(SIGSEGV, &action, &previous_segv) != 0 ||
        sigaction(SIGBUS, &action, &previous_bus) != 0) {
        _exit(127);
    }
}

int nixe_fastmem_execute(
    nixe_native_gateway gateway,
    void *frame,
    uintptr_t entry,
    uintptr_t base,
    uintptr_t size,
    uintptr_t *reported_fault
) {
    pthread_once(&install_once, install_handlers);
    arena_start = base;
    arena_end = base + size;
    fault_address = 0;
    if (sigsetjmp(recovery, 1) == 0) {
        active = base != 0 && size != 0;
        gateway(frame, entry);
        active = 0;
        return 0;
    }
    active = 0;
    *reported_fault = fault_address;
    return 1;
}

void nixe_fastmem_test_gateway(void *frame, uintptr_t entry) {
    (void)frame;
    volatile unsigned char value = *(volatile unsigned char *)entry;
    (void)value;
}
