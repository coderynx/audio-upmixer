//! Canonical movement feature and schedule bindings.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use upmixer_dsp_core::movement::{
    self, CanonicalFeatures, MovementFeatureSidecar, MovementSchedule,
};

#[pyfunction]
fn extract_movement_features<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: Option<PyReadonlyArray1<'py, f32>>,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let left = left.as_array().to_vec();
    let right = right.map(|value| value.as_array().to_vec());
    let features = movement::extract_features(&left, right.as_deref(), sample_rate)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("version", features.version)?;
    result.set_item("sample_rate", features.sample_rate)?;
    result.set_item("frame_count", features.frame_count)?;
    result.set_item("window_frames", features.window_frames)?;
    result.set_item(
        "energies",
        PyArray1::from_vec(py, features.energies.clone()),
    )?;
    result.set_item("levels_db", PyArray1::from_vec(py, features.levels_db()))?;
    result.set_item(
        "envelope_db",
        PyArray1::from_vec(py, features.envelope_db()),
    )?;
    result.set_item("onsets_db", PyArray1::from_vec(py, features.onsets_db()))?;
    Ok(result)
}

#[pyfunction]
fn extract_movement_features_json(
    left: PyReadonlyArray1<'_, f32>,
    right: Option<PyReadonlyArray1<'_, f32>>,
    sample_rate: u32,
) -> PyResult<String> {
    let left = left.as_array().to_vec();
    let right = right.map(|value| value.as_array().to_vec());
    let features = movement::extract_features(&left, right.as_deref(), sample_rate)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    serde_json::to_string(&features)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_movement_features(features_json: &str) -> PyResult<()> {
    let features: CanonicalFeatures = serde_json::from_str(features_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    features
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_movement_sidecar(sidecar_json: &str, stem_keys: Vec<String>) -> PyResult<()> {
    let sidecar: MovementFeatureSidecar = serde_json::from_str(sidecar_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let keys: Vec<&str> = stem_keys.iter().map(String::as_str).collect();
    sidecar
        .validate(&keys)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn compile_movement_schedule(request_json: &str) -> PyResult<String> {
    let request = serde_json::from_str(request_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let schedule =
        movement::compile_movement(&request).map_err(pyo3::exceptions::PyValueError::new_err)?;
    serde_json::to_string(&schedule)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Apply one compiled stem schedule to shaped source arrays. `source_map`
/// selects the source array for each output channel, which lets callers use
/// left/right/mono and already-shaped surround/height sends without copying
/// event interpolation into Python. `offset_frames` is absolute programme
/// time; `sample_rate` is the renderer rate and may differ from the compile
/// rate when the schedule is reused by a preview resampler.
#[pyfunction]
fn apply_movement_schedule<'py>(
    py: Python<'py>,
    schedule_json: &str,
    stem_index: usize,
    signals: Vec<PyReadonlyArray1<'py, f64>>,
    source_map: Vec<usize>,
    offset_frames: usize,
    sample_rate: u32,
    right: bool,
    channels: Vec<String>,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    if sample_rate == 0 || signals.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "movement application needs a sample rate and source arrays",
        ));
    }
    let schedule: MovementSchedule = serde_json::from_str(schedule_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    schedule
        .validate(channels.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let stem = schedule.stem(stem_index).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("movement schedule has no requested stem")
    })?;
    if source_map.len() != channels.len() || source_map.iter().any(|index| *index >= signals.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "movement source_map must select one source per output channel",
        ));
    }
    let views = signals
        .iter()
        .map(|signal| signal.as_array())
        .collect::<Vec<_>>();
    let frames = views[0].len();
    if views.iter().any(|signal| signal.len() != frames) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "movement source arrays must have equal lengths",
        ));
    }
    let mut output = vec![vec![0.0; frames]; channels.len()];
    let mut gains = vec![0.0; channels.len()];
    for frame in 0..frames {
        stem.sample_into_at(
            (offset_frames as f64 + frame as f64) * 1_000_000.0 / sample_rate as f64,
            right,
            &mut gains,
        );
        for channel in 0..channels.len() {
            output[channel][frame] = views[source_map[channel]][frame] * gains[channel];
        }
    }
    Ok(output
        .into_iter()
        .map(|values| PyArray1::from_vec(py, values))
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_movement_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_movement_features_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_movement_features, m)?)?;
    m.add_function(wrap_pyfunction!(validate_movement_sidecar, m)?)?;
    m.add_function(wrap_pyfunction!(compile_movement_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(apply_movement_schedule, m)?)?;
    m.add("MOVEMENT_FEATURE_VERSION", movement::FEATURE_VERSION)?;
    m.add("MOVEMENT_FEATURE_WINDOW_US", movement::FEATURE_WINDOW_US)?;
    m.add("MOVEMENT_EVENT_GRID_US", movement::EVENT_GRID_US)?;
    m.add("MOVEMENT_INTERPOLATION_US", movement::INTERPOLATION_US)?;
    Ok(())
}
