use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use upmixer_dsp_core::mastering::limiter::LimiterParams;
use upmixer_dsp_core::stream::conv::StreamingConvolver;
use upmixer_dsp_core::stream::limiter::StreamingLimiter;
use upmixer_dsp_core::stream::params::SendParams;
use upmixer_dsp_core::stream::routing::StemRouteState;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_TEST_LOCK: Mutex<()> = Mutex::new(());
const SR: u32 = 48_000;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn phase_four_stages_do_not_allocate_after_warmup() {
    let _lock = ALLOCATION_TEST_LOCK.lock().unwrap();
    let kernel: Vec<f64> = (0..6128)
        .map(|i| (i as f64 * 0.017).sin() / (i + 1) as f64)
        .collect();
    let mut pair = [
        StreamingConvolver::new(kernel.clone()),
        StreamingConvolver::new(kernel),
    ];
    let block = vec![0.25; 128];
    let (mut left, mut right) = (Vec::new(), Vec::new());
    StreamingConvolver::process_pair_into(&mut pair, &block, &mut left, &mut right);

    let mut limiter = StreamingLimiter::new(
        LimiterParams {
            ceiling_dbtp: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
            safety_margin_db: 0.1,
        },
        48_000,
        2,
        None,
    );
    let mut queue = vec![vec![0.8; 2048], vec![0.7; 2048]];
    limiter.process(&mut queue, 0, 0, 128, false);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    StreamingConvolver::process_pair_into(&mut pair, &block, &mut left, &mut right);
    limiter.process(&mut queue, 0, 128, 256, false);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}

fn send_params() -> SendParams {
    SendParams {
        surround_bass_cutoff_hz: 250.0,
        height_low_rolloff_hz: 150.0,
        height_low_rolloff_gain: 0.15,
        height_crossover_hz: 3000.0,
        height_high_shelf_gain: 1.5,
        height_directional_band_hz: 8000.0,
        height_directional_band_gain: 1.0,
        lfe_cutoff_hz: 120.0,
        lfe_filter_order: 4,
        lfe_gain: 1.0,
    }
}

#[test]
fn ambient_expansion_does_not_add_callback_allocations_for_a_large_block() {
    let _lock = ALLOCATION_TEST_LOCK.lock().unwrap();
    const N: usize = 4096;
    let params = send_params();
    let left = vec![0.25f32; N];
    let right = vec![0.125f32; N];
    let destinations = ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"];

    let mut active = StemRouteState::new_for_layout(SR, &params, None, None, None, &destinations);
    active.set_ambient(SR, &params, true, true, 2000.0, 2000.0, 0.0, 0.0);
    let mut absent = StemRouteState::new_for_layout(SR, &params, None, None, None, &[]);
    absent.set_ambient(SR, &params, true, true, 2000.0, 2000.0, 0.0, 0.0);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    active.process_block(
        &left,
        &right,
        0,
        N,
        [0.8, 0.8],
        [0.8, 0.8],
        [0.0, 0.0],
        false,
        false,
    );
    let active_allocations = ALLOCATIONS.load(Ordering::Relaxed);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    absent.process_block(
        &left,
        &right,
        0,
        N,
        [0.8, 0.8],
        [0.8, 0.8],
        [0.0, 0.0],
        false,
        false,
    );
    let absent_allocations = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(
        active_allocations, absent_allocations,
        "ambient output buffers allocated in process_block"
    );
}
