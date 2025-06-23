// user_thread.rs
#![feature(naked_functions)]
use std::arch::asm;
use std::arch::naked_asm;
use std::borrow::BorrowMut;
use std::cell::{RefCell, RefMut};
use std::ptr;
use std::vec::Vec;
use std::rc::Rc;
// 定义上下文结构 (只保存必要寄存器)
#[derive(Debug, Default, Clone)]
#[repr(C, align(16))] // 确保16字节对齐
pub struct ThreadContext {
    ra: u64,    // 返回地址寄存器 (x1)
    sp: u64,    // 栈指针寄存器 (x2)
    s: [u64; 12], // s0-s11 保存寄存器 (x8-x9, x18-x27)
}

// 实现上下文切换
#[naked]
#[no_mangle]
pub unsafe extern "C" fn context_switch() {
    // 使用纯汇编字符串，不包含任何 Rust 表达式
    naked_asm!(
        // 参数通过a0和a1寄存器传递
        // a0 = current, a1 = next
        
        // ==== 保存当前上下文 ====
        "sd ra, 0(a0)",
        "sd sp, 8(a0)",
        "sd s0, 16(a0)", 
        "sd s1, 24(a0)",
        "sd s2, 32(a0)", 
        "sd s3, 40(a0)",
        "sd s4, 48(a0)", 
        "sd s5, 56(a0)",
        "sd s6, 64(a0)", 
        "sd s7, 72(a0)",
        "sd s8, 80(a0)", 
        "sd s9, 88(a0)",
        "sd s10, 96(a0)", 
        "sd s11, 104(a0)",
        
        // ==== 恢复下一个上下文 ====
        "ld ra, 0(a1)",
        "ld sp, 8(a1)",
        "ld s0, 16(a1)", 
        "ld s1, 24(a1)",
        "ld s2, 32(a1)", 
        "ld s3, 40(a1)",
        "ld s4, 48(a1)", 
        "ld s5, 56(a1)",
        "ld s6, 64(a1)", 
        "ld s7, 72(a1)",
        "ld s8, 80(a1)", 
        "ld s9, 88(a1)",
        "ld s10, 96(a1)", 
        "ld s11, 104(a1)",
        
        // ==== 切换到新控制流 ====
        "ret"
    );
}


// 线程状态
#[derive(PartialEq, Eq, Clone, Copy)]
enum ThreadState {
    Ready,
    Running,
    Exited,
}

// RISC-V 专用栈分配器
struct ThreadStackAllocator;

impl ThreadStackAllocator {
    const STACK_ALIGN: usize = 16;
    const STACK_SIZE: usize = 16 * 1024; // 16KB
    
    fn allocate_stack() -> Vec<u8> {
        // 使用 mmap 确保正确权限 (RW)
        unsafe {
            let addr = libc::mmap(
                ptr::null_mut(),
                Self::STACK_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
                -1,
                0
            );
            
            if addr == libc::MAP_FAILED {
                panic!("Failed to allocate thread stack");
            }
            
            Vec::from_raw_parts(addr as *mut u8, Self::STACK_SIZE, Self::STACK_SIZE)
        }
    }
}

// 用户态线程结构
pub struct UserThread {
    id: usize,
    stack: Vec<u8>, // 线程栈
    context: ThreadContext, // 线程上下文
    state: ThreadState, // 线程状态
}

impl Drop for UserThread {
    fn drop(&mut self) {
        // 释放栈内存
        unsafe {
            libc::munmap(self.stack.as_ptr() as *mut libc::c_void, self.stack.len());
        }
    }
}

impl UserThread {
    pub fn new(id: usize, entry: extern "C" fn(usize), arg: usize) -> Self {
        let stack = ThreadStackAllocator::allocate_stack();
        let stack_top = stack.as_ptr() as usize + stack.len();
        let aligned_stack_top = stack_top - (stack_top % ThreadStackAllocator::STACK_ALIGN);
        
        // 在栈顶放置线程退出地址
        unsafe {
            let stack_ptr = (aligned_stack_top - 8) as *mut u64;
            *stack_ptr = thread_exit as u64;
        }
        
        // 初始化上下文
        let mut context = ThreadContext::default();
        context.ra = entry as u64;     // 入口函数
        context.sp = (aligned_stack_top - 16) as u64; // 栈指针（预留空间）
        
        // 设置第一个参数 (a0) 通过内联汇编
        // 注意：这会在当前线程执行，但参数会在新线程启动时使用
        unsafe {
            asm!(
                "mv a0, {0}",
                in(reg) arg,
                options(nomem, nostack)
            );
        }
        
        UserThread {
            id,
            stack,
            context,
            state: ThreadState::Ready,
        }
    }
}

// 线程退出函数
extern "C" fn thread_exit() {
    // 标记当前线程结束
    let mut sched = SCHEDULER.borrow_mut();
    sched.mark_current_exited();
    
    // 切换到下一个线程
    sched.schedule();
    
    // 永不返回
    loop { unsafe { asm!("wfi"); } }
}

// 主动让出函数
pub fn thread_yield() {
    let mut sched = SCHEDULER.borrow_mut();
    sched.schedule();
}

// 调度器
pub struct Scheduler {
    threads: Vec<Rc<RefCell<UserThread>>>, // 使用Rc<RefCell>实现内部可变性
    current: usize, // 当前运行线程索引
    main_context: ThreadContext, // 主线程上下文
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            threads: Vec::new(),
            current: usize::MAX, // 初始无当前线程
            main_context: ThreadContext::default(),
        }
    }
    
    pub fn add_thread(&mut self, thread: UserThread) {
        self.threads.push(Rc::new(RefCell::new(thread)));
    }
    
    pub fn run(&mut self) -> ! {
        // 保存主线程上下文
        let mut main_ctx = ThreadContext::default();
        
        // 确保至少有一个线程
        if self.threads.is_empty() {
            panic!("No threads to run");
        }
        
        // 设置第一个线程为运行状态
        self.threads[0].borrow_mut().state = ThreadState::Running;
        self.current = 0;
        
        let first_thread_ctx = &self.threads[0].borrow().context;
        
        unsafe {
            // 首次切换到第一个线程
            context_switch(
                &mut main_ctx as *mut _,
                first_thread_ctx as *const _
            );
        }
        
        // 永远不会到达这里
        loop {}
    }
    
    pub fn schedule(&mut self) {
        let next = self.find_next_ready();
        let current = self.current;
        
        // 更新状态
        if current != usize::MAX {
            self.threads[current].borrow_mut().state = ThreadState::Ready;
        }
        self.threads[next].borrow_mut().state = ThreadState::Running;
        
        let current_ptr = if current != usize::MAX {
            &mut self.threads[current].borrow_mut().context as *mut ThreadContext
        } else {
            &mut self.main_context as *mut ThreadContext
        };
        
        let next_ptr = &self.threads[next].borrow().context as *const ThreadContext;
        
        self.current = next;
        
        unsafe {
            context_switch(current_ptr, next_ptr);
        }
    }
    
    fn find_next_ready(&self) -> usize {
        if self.threads.is_empty() {
            panic!("No threads available");
        }
        
        let start = if self.current == usize::MAX {
            0
        } else {
            (self.current + 1) % self.threads.len()
        };
        
        for i in 0..self.threads.len() {
            let idx = (start + i) % self.threads.len();
            if self.threads[idx].borrow().state == ThreadState::Ready {
                return idx;
            }
        }
        
        // 如果没有就绪线程，检查是否有运行中的线程
        if self.current != usize::MAX && self.threads[self.current].borrow().state == ThreadState::Running {
            return self.current;
        }
        
        panic!("No ready threads found");
    }
    
    pub fn mark_current_exited(&mut self) {
        if self.current != usize::MAX {
            self.threads[self.current].borrow_mut().state = ThreadState::Exited;
            // 不立即移除线程，避免悬垂指针
        }
    }
}

// 全局调度器
thread_local! {
    static SCHEDULER: RefCell<Scheduler> = RefCell::new(Scheduler::new());
}

// 线程入口函数
extern "C" fn thread_entry(arg: usize) {
    println!("Thread {} started", arg);
    
    for i in 0..5 {
        println!("Thread {}: {}", arg, i);
        thread_yield(); // 主动让出CPU
    }
    
    println!("Thread {} exiting", arg);
    // 线程退出（通过返回触发）
}

fn main() {
    // 创建线程
    let mut scheduler = Scheduler::new();
    for i in 0..3 {
        let thread = UserThread::new(i, thread_entry, i);
        scheduler.add_thread(thread);
    }
    
    // 设置全局调度器
    SCHEDULER.with(|s| {
        *s.borrow_mut() = scheduler;
    });
    
    // 启动调度器
    SCHEDULER.with(|s| {
        s.borrow_mut().run();
    });
}

/* 
// 协程结构（共享线程栈）
struct Coroutine {
    id: u64,
    stack_snapshot: Vec<u8>, // 栈快照
    context: ThreadContext,  // 协程上下文
    state: CoroutineState,
    thread_id: u64,         // 所属线程
}

// 协程切换函数
fn coro_switch(current_thread: &mut UserThread, next_coro: &Coroutine) {
    // 保存当前协程栈快照
    if let Some(current_coro) = current_thread.current_coro() {
        save_stack_snapshot(current_coro);
    }
    // 恢复目标协程栈
    restore_stack_snapshot(next_coro);
    // 切换上下文
    unsafe {
        let mut dummy_ctx = ThreadContext::default();
        context_switch(&mut dummy_ctx, &next_coro.context);
    }
}
// 协程创建
impl UserThread {
    fn spawn_coro(&mut self, entry: fn()) -> Coroutine {
        let coro_id = self.next_coro_id();
        let mut coro = Coroutine {
            id: coro_id,
            stack_snapshot: Vec::new(),
            context: ThreadContext::default(),
            state: CoroutineState::Ready,
            thread_id: self.id,
        };
        
        // 设置协程入口点
        coro.context.rip = entry as u64;
        self.coros.push(coro);
        coro
    }
}
struct Scheduler {
    threads: Vec<UserThread>,        // 所有用户态线程
    ready_threads: VecDeque<u64>,   // 就绪线程ID队列
    thread_coroutines: HashMap<u64, VecDeque<u64>>, // 各线程的就绪协程
}

impl Scheduler {
    fn run(&mut self) {
        loop {
            // 调度线程
            if let Some(thread_id) = self.ready_threads.pop_front() {
                let thread = self.get_thread_mut(thread_id);
                // 在该线程上调度协程
                if let Some(coro_id) = self.thread_coroutines[&thread_id].pop_front() {
                    let coro = thread.get_coroutine(coro_id);
                    thread.execute_coroutine(coro);
                }
                // 将线程放回队列（如果还有任务）
                if !self.thread_coroutines[&thread_id].is_empty() {
                    self.ready_threads.push_back(thread_id);
                }
            }
            // 处理I/O事件
            self.process_io_events();
        }
    }
}
// I/O事件管理器
struct IoManager {
    epoll_fd: i32, // Linux epoll
    // 或 kqueue: i32, // macOS
    waiting_coroutines: HashMap<i32, u64>, // fd -> 等待的协程ID
}

impl Scheduler {
    fn process_io_events(&mut self) {
        let mut events = [EpollEvent::empty(); 1024];
        let timeout = 10; // ms
        
        // 等待事件
        let n = epoll_wait(
            self.io_manager.epoll_fd,
            &mut events,
            timeout
        ).unwrap();
        
        // 唤醒等待的协程
        for event in &events[..n] {
            if let Some(coro_id) = self.io_manager.waiting_coroutines.remove(&event.data.fd) {
                let thread_id = /* 根据协程找到线程ID */;
                self.wake_coroutine(thread_id, coro_id);
            }
        }
    }
}

// 协程友好的read函数
async fn async_read(fd: i32, buf: &mut [u8]) -> Result<usize> {
    loop {
        match libc::read(fd, buf.as_mut_ptr() as _, buf.len()) {
            n if n >= 0 => return Ok(n as usize),
            _ => {
                if errno() == libc::EAGAIN {
                    // 注册等待并挂起协程
                    scheduler.register_io_wait(fd, current_coroutine_id());
                    coroutine_yield().await;
                } else {
                    return Err(io::Error::last_os_error());
                }
            }
        }
    }
}
    */