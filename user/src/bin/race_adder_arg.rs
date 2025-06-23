#![no_std]
#![no_main]

#[macro_use]
extern crate user_lib;
extern crate alloc;

use crate::alloc::string::ToString;
use alloc::vec::Vec;
use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicBool, Ordering};
use user_lib::{exit, get_time, thread_create, waittid};

// 共享全局变量（受锁保护）
static mut A: usize = 0;

// 自旋锁结构体
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

// 全局自旋锁
static LOCK: Spinlock = Spinlock::new();

const PER_THREAD: usize = 1000;
const THREAD_COUNT: usize = 10;

fn f(count: usize) -> ! {
    let mut t = 2usize;
    for _ in 0..PER_THREAD {
        // 执行计算密集型操作
        for _ in 0..count {
            t = t * t % 10007;
        }
        
        // 获取锁保护共享变量
        LOCK.lock();
        unsafe {
            let a_ptr = addr_of_mut!(A);
            let cur = a_ptr.read_volatile();
            a_ptr.write_volatile(cur + 1);
        }
        LOCK.unlock();
    }
    exit(t as i32)
}

#[unsafe(no_mangle)]
pub fn main(argc: usize, argv: &[&str]) -> i32 {
    let count: usize;
    if argc == 1 {
        count = THREAD_COUNT;
    } else if argc == 2 {
        count = argv[1].to_string().parse::<usize>().unwrap();
    } else {
        println!(
            "ERROR in argv, argc is {}, argv[0] {} , argv[1] {} , argv[2] {}",
            argc, argv[0], argv[1], argv[2]
        );
        exit(-1);
    }

    let start = get_time();
    let mut v = Vec::new();
    for _ in 0..THREAD_COUNT {
        v.push(thread_create(f as usize, count) as usize);
    }
    let mut time_cost = Vec::new();
    for tid in v.iter() {
        time_cost.push(waittid(*tid));
    }
    println!("time cost is {}ms", get_time() - start);
    
    //在锁的保护下访问共享变量
    LOCK.lock();
    let final_value = unsafe { 
        let a_ptr = addr_of_mut!(A);
        a_ptr.read_volatile()
    };
    LOCK.unlock();
    
    println!("Final value of A: {}", final_value);
    assert_eq!(final_value, PER_THREAD * THREAD_COUNT);
    
    0
}