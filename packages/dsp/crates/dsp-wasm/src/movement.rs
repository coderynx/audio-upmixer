//! One-shot movement compiler ABI for the browser worker.
//!
//! The host copies one JSON request into linear memory, compiles it once, and
//! reads one owned result buffer. A result is always returned so parse and
//! validation failures can cross the C ABI as `{"error":"..."}` without a
//! second shape or an allocation owned by JavaScript.

use upmixer_dsp_core::movement::{compile_movement, MovementCompileRequest};

fn result_bytes(result: Result<String, String>) -> *mut Vec<u8> {
    let bytes = match result {
        Ok(json) => json.into_bytes(),
        Err(error) => serde_json::json!({ "error": error })
            .to_string()
            .into_bytes(),
    };
    Box::into_raw(Box::new(bytes))
}

/// Compile one canonical movement request into an owned JSON result buffer.
/// The handle must be released with [`dsp_movement_result_free`].
///
/// # Safety
/// `request_ptr` must address `request_len` readable UTF-8 JSON bytes, or be
/// null only when `request_len` is zero.
#[no_mangle]
pub unsafe extern "C" fn dsp_movement_compile_json(
    request_ptr: *const u8,
    request_len: usize,
) -> *mut Vec<u8> {
    if request_ptr.is_null() && request_len != 0 {
        return result_bytes(Err("movement request pointer is null".into()));
    }
    let input = if request_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(request_ptr, request_len)
    };
    let request = match serde_json::from_slice::<MovementCompileRequest>(input) {
        Ok(request) => request,
        Err(error) => return result_bytes(Err(format!("invalid movement request: {error}"))),
    };
    result_bytes(compile_movement(&request).and_then(|schedule| {
        serde_json::to_string(&schedule)
            .map_err(|error| format!("movement result encoding failed: {error}"))
    }))
}

/// Pointer to the result bytes. The pointer remains valid until the result is
/// released and must not be passed to [`dsp_free`].
///
/// # Safety
/// `handle` must be a live result returned by [`dsp_movement_compile_json`].
#[no_mangle]
pub unsafe extern "C" fn dsp_movement_result_ptr(handle: *const Vec<u8>) -> *const u8 {
    handle
        .as_ref()
        .map_or(std::ptr::null(), |bytes| bytes.as_ptr())
}

/// Number of bytes available through [`dsp_movement_result_ptr`].
///
/// # Safety
/// `handle` must be a live result returned by [`dsp_movement_compile_json`].
#[no_mangle]
pub unsafe extern "C" fn dsp_movement_result_len(handle: *const Vec<u8>) -> usize {
    handle.as_ref().map_or(0, Vec::len)
}

/// Release a compiler result buffer.
///
/// # Safety
/// `handle` must be a live result returned by [`dsp_movement_compile_json`].
#[no_mangle]
pub unsafe extern "C" fn dsp_movement_result_free(handle: *mut Vec<u8>) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
