#![no_std]
#![no_main]
#![feature(naked_functions)]
extern crate alloc;
#[macro_use]
extern crate user_lib;
use core::arch::naked_asm;
use alloc::vec;
use alloc::vec::Vec;
use user_lib::exit;
// In our simple example we set most constraints here.
const DEFAULT_STACK_SIZE: usize = 4096; //128 got  SEGFAULT, 256(1024, 4096) got right results.
const MAX_THREADS: usize = 5;
static mut RUNTIME: usize = 0;
// 程序运行时
pub struct Runtime {
    threads: Vec<Thread>,
    current: usize,
}

#[derive(PartialEq, Eq, Debug)]
enum State {
    Available,
    Running,
    Ready,
}
// 用户态线程定义
struct Thread {
    id: usize,
    stack: Vec<u8>,
    ctx: ThreadContext,
    state: State,
}

#[derive(Debug, Default)]
#[repr(C)] // not strictly needed but Rust ABI is not guaranteed to be stable
pub struct ThreadContext {
    // 15 u64
    x1: u64,  //ra: return addres
    x2: u64,  //sp
    x8: u64,  //s0,fp
    x9: u64,  //s1
    x18: u64, //x18-27: s2-11
    x19: u64,
    x20: u64,
    x21: u64,
    x22: u64,
    x23: u64,
    x24: u64,
    x25: u64,
    x26: u64,
    x27: u64,
    nx1: u64, //new return addres
}

impl Thread {
    fn new(id: usize) -> Self {
        Thread {
            id: id,
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Available,
        }
    }
}

impl Runtime {
    pub fn new() -> Self {
        // This will be our base thread, which will be initialized in the `running` state
        let base_thread = Thread {
            id: 0,
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Running,
        };

        // We initialize the rest of our threads.
        let mut threads = vec![base_thread];
        let mut available_threads: Vec<Thread> = (1..MAX_THREADS).map(|i| Thread::new(i)).collect();
        threads.append(&mut available_threads);

        Runtime { threads, current: 0 }
    }
    pub fn init(&self) {
        unsafe {
            let r_ptr: *const Runtime = self;
            RUNTIME = r_ptr as usize;
        }
    }
    pub fn run(&mut self) {
        while self.t_yield() {}
        println!("All threads finished!");
    }
    fn t_return(&mut self) {
        if self.current != 0 {
            self.threads[self.current].state = State::Available;
            self.t_yield();
        }
    }
    #[inline(never)]
    fn t_yield(&mut self) -> bool {
        let mut pos = self.current;
        while self.threads[pos].state != State::Ready {
            pos += 1;
            if pos == self.threads.len() {
                pos = 0;
            }
            if pos == self.current {
                return false;
            }
        }

        if self.threads[self.current].state != State::Available {
            self.threads[self.current].state = State::Ready;
        }

        self.threads[pos].state = State::Running;
        let old_pos = self.current;
        self.current = pos;

        unsafe {
            switch(&mut self.threads[old_pos].ctx, &self.threads[pos].ctx);
        }
        self.threads.len() > 0
    }

    pub fn spawn(&mut self, f: fn()) {
        let available = self
            .threads
            .iter_mut()
            .find(|t| t.state == State::Available)
            .expect("no available thread.");

        println!("RUNTIME: spawning thread {}\n", available.id);
        let size = available.stack.len();
        unsafe {
            let s_ptr = available.stack.as_mut_ptr().offset(size as isize);
            let s_ptr = (s_ptr as usize & !7) as *mut u8;

            available.ctx.x1 = guard as u64; //ctx.x1  is old return address
            available.ctx.nx1 = f as u64; //ctx.nx2 is new return address
            available.ctx.x2 = s_ptr as u64; //cxt.x2 is sp
        }
        available.state = State::Ready;
    }
}

/// This is our guard function that we place on top of the stack. All this function does is set the
/// state of our current thread and then `yield` which will then schedule a new thread to be run.
fn guard() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_return();
    };
}
pub fn yield_thread() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_yield();
    };
}
// 上下文切换函数
#[naked]
#[unsafe(no_mangle)]
unsafe extern "C" fn switch(old: *mut ThreadContext, new: *const ThreadContext) {
    unsafe {
        // a0: _old, a1: _new
        naked_asm!(
            "
            sd x1, 0x00(a0)
            sd x2, 0x08(a0)
            sd x8, 0x10(a0)
            sd x9, 0x18(a0)
            sd x18, 0x20(a0)
            sd x19, 0x28(a0)
            sd x20, 0x30(a0)
            sd x21, 0x38(a0)
            sd x22, 0x40(a0)
            sd x23, 0x48(a0)
            sd x24, 0x50(a0)
            sd x25, 0x58(a0)
            sd x26, 0x60(a0)
            sd x27, 0x68(a0)
            sd x1, 0x70(a0)

            ld x1, 0x00(a1)
            ld x2, 0x08(a1)
            ld x8, 0x10(a1)
            ld x9, 0x18(a1)
            ld x18, 0x20(a1)
            ld x19, 0x28(a1)
            ld x20, 0x30(a1)
            ld x21, 0x38(a1)
            ld x22, 0x40(a1)
            ld x23, 0x48(a1)
            ld x24, 0x50(a1)
            ld x25, 0x58(a1)
            ld x26, 0x60(a1)
            ld x27, 0x68(a1)
            ld t0, 0x70(a1)

            jr t0
            "
        );
    }
}

// 协程实现
use alloc::boxed::Box;
use alloc::collections::{BTreeMap,VecDeque};
use alloc::sync::{Arc,Weak};
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize,Ordering};
use core::task::{Context, Poll, Waker,RawWaker,RawWakerVTable};
use spin::Mutex;


// 协程状态
#[derive(Copy, Clone, PartialEq)]
enum CoroutineState {
    Initial,
    Running,
    Halted,
    Completed,
}

// 协程控制块
struct Coroutine {
    state: CoroutineState,
}

impl Coroutine {
    fn new() -> Self {
        Coroutine {
            state: CoroutineState::Running,
        }
    }
    
    // 创建等待器
    fn waiter(&mut self) -> Waiter<'_> {
        Waiter { coroutine: self }
    }
}

// 协程等待器 (Future实现)
struct Waiter<'a> {
    coroutine: &'a mut Coroutine,
}

impl Future for Waiter<'_> {
    type Output = ();
    
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context) -> Poll<()> {
        let coroutine = &mut self.as_mut().get_mut().coroutine;
        
        match coroutine.state {
            CoroutineState::Initial | CoroutineState::Halted => {
                coroutine.state = CoroutineState::Running;
                Poll::Ready(())
            }
            CoroutineState::Running => {
                coroutine.state = CoroutineState::Halted;
                Poll::Pending
            }
            CoroutineState::Completed =>{
                Poll::Ready(())
            }
        }
    }
}
// 任务ID生成器
static TASK_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
// 真实唤醒器结构
struct RealWaker {
    task_id: usize,
    executor: Weak<Mutex<CoroutineExecutor>>,
}
// 实现唤醒器逻辑
impl RealWaker {
    fn new(task_id: usize, executor: Weak<Mutex<CoroutineExecutor>>) -> Self {
        RealWaker { task_id, executor }
    }

    // 唤醒关联任务
    fn wake(&self) {
        if let Some(exec) = self.executor.upgrade() {
            let mut exec = exec.lock();
            exec.wake_task(self.task_id);
        }
    }

    // 转换为RawWaker
    fn into_raw_waker(self) -> RawWaker {
        let ptr = Box::into_raw(Box::new(self)) as *const ();
        let vtable = &REAL_WAKER_VTABLE;
        RawWaker::new(ptr, vtable)
    }
}
// 定义RawWakerVTable
static REAL_WAKER_VTABLE: RawWakerVTable = {
    unsafe fn clone(ptr: *const ()) -> RawWaker {
        let waker = &*(ptr as *const RealWaker);
        let cloned = RealWaker {
            task_id: waker.task_id,
            executor: waker.executor.clone(),
        };
        cloned.into_raw_waker()
    }

    unsafe fn wake(ptr: *const ()) {
        let waker = Box::from_raw(ptr as *mut RealWaker);
        waker.wake();
    }

    unsafe fn wake_by_ref(ptr: *const ()) {
        let waker = &*(ptr as *const RealWaker);
        waker.wake();
    }

    unsafe fn drop(ptr: *const ()) {
        let _ = Box::from_raw(ptr as *mut RealWaker);
    }

    RawWakerVTable::new(clone, wake, wake_by_ref, drop)
};
// 协程执行器
struct CoroutineExecutor {
    tasks: BTreeMap<usize, Pin<Box<dyn Future<Output = ()>>>>,
    ready_queue: VecDeque<usize>,
    pending: BTreeMap<usize, Waker>,
    executor_ref: Weak<Mutex<Self>>,
}

impl CoroutineExecutor {
    fn new() -> Arc<Mutex<Self>> {
        let executor = Arc::new(Mutex::new(CoroutineExecutor {
            tasks: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            pending: BTreeMap::new(),
            executor_ref: Weak::new(),
        }));
        
        // 设置自引用
        let mut lock = executor.lock();
        lock.executor_ref = Arc::downgrade(&executor);
        Arc::clone(&executor)
    }
    
    fn spawn<F>(&mut self, f: F) -> usize
    where
        F: Future<Output = ()> + 'static
    {
        let task_id = TASK_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        self.tasks.insert(task_id, Box::pin(f));
        self.ready_queue.push_back(task_id);
        task_id
    }
    
    fn wake_task(&mut self, task_id: usize) {
        if self.tasks.contains_key(&task_id) {
            self.ready_queue.push_back(task_id);
            self.pending.remove(&task_id);
        }
    }
    
    fn run_step(&mut self) {
        // 处理所有就绪任务
        while let Some(task_id) = self.ready_queue.pop_front() {
            if let Some(mut task) = self.tasks.remove(&task_id) {
                // 创建真实唤醒器
                let waker = {
                    let real_waker = RealWaker::new(
                        task_id, 
                        self.executor_ref.clone()
                    );
                    unsafe { Waker::from_raw(real_waker.into_raw_waker()) }
                };
                
                let mut cx = Context::from_waker(&waker);
                
                match task.as_mut().poll(&mut cx) {
                    Poll::Pending => {
                        // 任务挂起，保存唤醒器
                        self.pending.insert(task_id, waker);
                        self.tasks.insert(task_id, task);
                    }
                    Poll::Ready(()) => {
                        // 任务完成
                        self.pending.remove(&task_id);
                    }
                }
            }
        }
    }
}


// 协程让出
async fn yield_now() {
    let mut coroutine = Coroutine::new();
    coroutine.waiter().await;
}

use crate::alloc::string::ToString;
use core::ptr::addr_of_mut;
use core::sync::atomic::AtomicBool;
use user_lib::{get_time, thread_create, waittid};

// 自旋锁
struct Spinlock {
    locked: AtomicBool,
}

impl Spinlock {
    const fn new() -> Self {
        Spinlock {
            locked: AtomicBool::new(false),
        }
    }
    fn lock(&self) {
        // 自旋等待直到获取锁
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // 在等待时降低CPU使用率
            core::hint::spin_loop();
        }
    }
    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}


// 测试函数

// 全局自旋锁
static LOCK: Spinlock = Spinlock::new();
const PER_THREAD: usize = 1000;
const THREAD_COUNT: usize = 10;

// fn thread_work() -> ! {
//     let count = unsafe {THREAD_COUNT};
//     for _ in 0..PER_THREAD {
//         // 执行计算密集型操作
//         for _ in 0..count {
//             t = t * t % 10007;
//         }
//         // 获取锁保护共享变量
//         LOCK.lock();
//         unsafe {
//             let a_ptr = addr_of_mut!(A);
//             let cur = a_ptr.read_volatile();
//             a_ptr.write_volatile(cur + 1);
//         }
//         LOCK.unlock();
//     }
//     exit(t as i32)
// }

// // 主函数
// #[unsafe(no_mangle)]
// pub fn main(argc: usize, argv: &[&str]) -> i32 {
//     let count: usize;
//     if argc == 1 {
//         count = THREAD_COUNT;
//     } else if argc == 2 {
//         count = argv[1].to_string().parse::<usize>().unwrap();
//     } else {
//         println!(
//             "ERROR in argv, argc is {}, argv[0] {} , argv[1] {} , argv[2] {}",
//             argc, argv[0], argv[1], argv[2]
//         );
//         exit(-1);
//     }

//     let start = get_time();
//     let mut v = Vec::new();
//     for _ in 0..THREAD_COUNT {
//         v.push(thread_create(f as usize, count) as usize);
//     }
//     let mut time_cost = Vec::new();
//     for tid in v.iter() {
//         time_cost.push(waittid(*tid));
//     }
//     println!("time cost is {}ms", get_time() - start);
//     LOCK.lock();
//     let final_value = unsafe { 
//         let a_ptr = addr_of_mut!(A);
//         a_ptr.read_volatile()
//     };
//     LOCK.unlock();
//     println!("Final value of A: {}", final_value);
//     assert_eq!(final_value, PER_THREAD * THREAD_COUNT);
//     0
// }